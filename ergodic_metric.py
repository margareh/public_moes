
import numpy as onp
import sys
import copy

from jax import vmap, jit, grad
import jax.numpy as np
from jax.lax import scan
from functools import partial

ERG_COEF = 1 # 1
REG_COEF = 0.05 # 3e-2
LENGTH_COEF = 1
BOUND_COEF = 1000 # 10
TRANSL_COEF = 0
ANG_COEF = 1
TARGET_V = 5 # in pixels
MIN_LENGTH = 5 # in pixels

# def fDyn(x, u): # dynamics of the robot
# 	xnew = x + np.tanh(u)
# 	return xnew, x

def fDiffDrive(x0, u):
	"""
	x0 = (x,y,theta)
	u = (v,w)
	"""
	x = x0 + np.array([np.cos(x0[2])*np.abs(u[0]), np.sin(x0[2])*np.abs(u[0]), 10*u[1]])
	return x, x0

def get_hk(k): # normalizing factor for basis function
	_hk = (2. * k + onp.sin(2 * k))/(4. * k)
	_hk = _hk.at[onp.isnan(_hk)].set(1.)
	return onp.sqrt(onp.prod(_hk))

def fk(x, k): # basis function
    return np.prod(np.cos(x*k))

def GetTrajXY(u, x0, all=False):
	"""
	"""
	xf, tr0 = scan(fDiffDrive, x0, u)
	if all:
		tr = copy.copy(tr0)
	else:
		tr = tr0[:,0:2] # take the first two dim
	return xf, tr

class ErgCalc(object):
	"""
	modified from Ian's Ergodic Coverage code base.
	"""
	def __init__(self, pdf, n_fourier=10, nPix=100):
		self.nPix = nPix
		# aux func
		self.fk_vmap = lambda _x, _k: vmap(fk, in_axes=(0,None))(_x, _k)

		# fourier indices
		k1, k2 = np.meshgrid(*[np.arange(0, n_fourier, step=1)]*2)
		k = np.stack([k1.ravel(), k2.ravel()]).T
		self.k = np.pi*k

		# lambda, the weights of different bands.
		self.lamk = (1.+np.linalg.norm(self.k/np.pi,axis=1)**2)**(-4./2.)

		# the normalization factor
		hk = []
		for ki in k:
		    hk.append(get_hk(ki))
		self.hk = np.array(hk)

		# compute phik
		X,Y = np.meshgrid(*[np.linspace(0,1,num=self.nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		phik = np.dot(vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k), pdf) #vmap(p)(_s)
		phik = phik/phik[0]
		self.phik = phik/self.hk

		# for reconstruction
		self.phik_recon = np.dot(self.phik, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		
		# to compute gradient func
		self.gradient = jit(grad(self.fourier_ergodic_loss))

		return

	def get_ck(self, tr):
		"""
		given a trajectory tr, compute fourier coeffient of its spatial statistics.
		k is the number of fourier coeffs.
		"""
		ck = np.mean(vmap(partial(self.fk_vmap, tr))(self.k), axis=1)
		ck = ck / self.hk
		return ck

	def fourier_ergodic_loss(self, u, x0):
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		# return np.sum(self.lamk*np.square(self.phik - ck)) \
		    #+ 1e-2 * np.mean(u[:,0]**2) + 3e-3*np.mean(u[:,1]**2)
		# return np.sum(self.lamk*np.square(self.phik - ck)) \
		# 	+ 3e-2 * np.mean(u**2) + np.mean((tr - np.array([0.5,0.5]))**10)
		return ERG_COEF * np.sum(self.lamk*np.square(self.phik - ck)) \
			+ REG_COEF * np.mean(u**2) + BOUND_COEF * np.sum(np.maximum(0, tr-1)**2 + np.maximum(0, -tr)**2)

	def spectral_decomposition(self,nPix=100): # some question to discuss
		phik1 = self.phik
		phik2 = self.phik
		for i in range(len(self.phik)):
		  if i < len(self.phik)/2:
		    phik1 = phik1.at[i].set(0)
		  else:
		  	phik2 = phik2.at[i].set(0)

		print("phik filtered: ", phik1)
		print("phik filtered 2: ", phik2)
		# phik2 = self.phik - phik1
		X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		pdf1 = np.dot(phik1, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		pdf2 = np.dot(phik2, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf1, pdf2

	def traj_stat(self, u, x0):
		"""
		"""
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		X,Y = np.meshgrid(*[np.linspace(0,1,num=self.nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		pdf = np.dot(ck, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf
	

class GPErgCalc(object):
	"""
	modified from public_moes, which was modified from Ian's Ergodic Coverage code base.
	"""
	def __init__(self, pdf, fourier_freqs=None, freq_vars=None, nPix=100, scale=1):

		# set target velocity based on number of pixels
		self.target_v = TARGET_V / nPix
		self.min_length = MIN_LENGTH / nPix

		self.nPix = nPix
		# aux func
		self.fk_vmap = lambda _x, _k: vmap(fk, in_axes=(0,None))(_x, _k)

		# get fourier frequencies
		if fourier_freqs is None:
			n_fourier=10

			# fourier indices
			k1, k2 = np.meshgrid(*[np.arange(0, n_fourier, step=1)]*2)
			k = np.stack([k1.ravel(), k2.ravel()]).T

		else:
			fourier_freqs = np.array(fourier_freqs)
			n_fourier = fourier_freqs.shape[0]
			k1 = np.tile(fourier_freqs[:,0], (n_fourier,1))
			k2 = np.tile(fourier_freqs[:,1], (n_fourier,1)).T
			k = np.stack([k1.ravel(), k2.ravel()]).T
			# k = np.array(fourier_freqs)

		self.k = np.pi*k

		# if freq_vars is None:
			# lambda, the weights of different bands.
		self.lamk = (1.+np.linalg.norm(self.k/(np.pi),axis=1)**2)**(-3./2.)
		if freq_vars is not None:
			# self.lamk = 1 / np.prod(np.sqrt(np.array(freq_vars)), axis=1)
			self.lamk = np.array(freq_vars) * self.lamk

		# the normalization factor
		hk = []
		for ki in k:
			hk.append(get_hk(ki))
		self.hk = np.array(hk)

		# compute phik
		X,Y = np.meshgrid(*[np.linspace(0,1,num=self.nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		phik = np.dot(vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k), pdf) #vmap(p)(_s)
		phik = phik/phik[0]
		self.phik = phik/self.hk

		# for reconstruction
		self.phik_recon = np.dot(self.phik, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		
		# to compute gradient func
		self.gradient = jit(grad(self.fourier_ergodic_loss))

		return

	def get_ck(self, tr):
		"""
		given a trajectory tr, compute fourier coeffient of its spatial statistics.
		k is the number of fourier coeffs.
		"""
		ck = np.mean(vmap(partial(self.fk_vmap, tr))(self.k), axis=1)
		ck = ck / self.hk
		return ck

	def fourier_ergodic_loss(self, u, x0, print_flag=False):
		erg_loss, length_loss, bound_loss, ang_loss, control_loss = self.loss_components(u, x0)
		if print_flag:
			print("LOSS: erg = {:4.4f}; control = {:4.4f}, {:4.4f}; length = {:4.4f}; boundary = {:4.4f}".format(erg_loss, control_loss, ang_loss, length_loss, bound_loss))
		return (ERG_COEF * erg_loss) + (REG_COEF * (TRANSL_COEF*control_loss+ANG_COEF*ang_loss)) + (LENGTH_COEF * length_loss) + (BOUND_COEF * bound_loss)
	
	def loss_components(self, u, x0):
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		erg_loss = np.sum(self.lamk*np.square(self.phik - ck))
		# control_loss = np.mean(u**2)
		diffs = tr[1:]-tr[0:-1]
		lengths = np.sum(np.square(diffs), axis=1)
		length_loss = np.mean(np.square(self.min_length - lengths))
		ang_loss = np.mean(u[:,1]**2, axis=0)
		control_loss = np.mean(np.square(u[:,0]-self.target_v), axis=0)
		bound_loss = np.sum(np.maximum(0, tr-1) + np.maximum(0, -tr))
		return erg_loss, length_loss, bound_loss, ang_loss, control_loss

	def spectral_decomposition(self,nPix=100): # some question to discuss
		phik1 = self.phik
		phik2 = self.phik
		for i in range(len(self.phik)):
		  if i < len(self.phik)/2:
		    phik1 = phik1.at[i].set(0)
		  else:
		  	phik2 = phik2.at[i].set(0)

		print("phik filtered: ", phik1)
		print("phik filtered 2: ", phik2)
		# phik2 = self.phik - phik1
		X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		pdf1 = np.dot(phik1, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		pdf2 = np.dot(phik2, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf1, pdf2

	def traj_stat(self, u, x0):
		"""
		"""
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		X,Y = np.meshgrid(*[np.linspace(0,1,num=self.nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		pdf = np.dot(ck, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf
