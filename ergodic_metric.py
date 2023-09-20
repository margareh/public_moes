
import numpy as onp
import sys
import copy

from jax import vmap, jit, grad
import jax.numpy as np
from jax.lax import scan
from functools import partial


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
		return np.sum(self.lamk*np.square(self.phik - ck)) \
			+ 3e-2 * np.mean(u**2) + 10 * np.sum(np.max(tr - 1,0)**2 + np.min(tr, 0)**2)

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
			k = np.array(fourier_freqs)

		self.k = np.pi*k

		if freq_vars is None:
			# lambda, the weights of different bands.
			self.lamk = (1.+np.linalg.norm(self.k/(np.pi*scale),axis=1)**2)**(-4./2.)
		else:
			self.lamk = 1 / np.prod(np.sqrt(np.array(freq_vars)), axis=1)

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
		return np.sum(self.lamk*np.square(self.phik - ck)) \
			+ 3e-2 * np.mean(u**2) + np.mean((tr - np.array([0.5,0.5]))**8)

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
