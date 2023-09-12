
import numpy as onp
import jax.numpy as np
# from jax.experimental import optimizers
from jax.example_libraries import optimizers

import matplotlib.pyplot as plt
import public_moes.ergodic_metric as ergodic_metric

GLOBAL_NUM_K = 0

def ErgCover(pdf, nA, s0, n_fourier, nPix, nIter, ifDisplay, u_init=None, stop_eps=-1, kkk=0):
	"""
	run ergodic coverage over a info map. Modified from Ian's code.
	return a list of control inputs.
	"""
	print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	erg_calc = ergodic_metric.ErgCalc(pdf, n_fourier, nPix)

	opt_init, opt_update, get_params = optimizers.adam(1e-3)

	# initial conditions
	x0 = np.array(s0[:3])
	# x0 = np.array([0,0,s0[0],s0[1]])
	u = np.zeros((nA,2))
	if u_init is not None:
		# print(u_init)
		u = np.array(u_init)
	# u = np.ones((nA,2))*0.1
	# u = 0.05*(onp.random.random((nA,2))*2-1) # each component range from [-1,1]*0.05
	opt_state = opt_init(u)
	log = []

	# if stop_eps > 0:
	# 	nIter = int(1e5) # set a large number, stop until converge.

	i = 0
	loss = 1e5
	while (i < nIter) and (loss >= stop_eps):
	# for i in range(nIter):
		g = erg_calc.gradient(get_params(opt_state), x0)
		opt_state = opt_update(i, g, opt_state)
		u = get_params(opt_state)
		log.append(erg_calc.fourier_ergodic_loss(u, x0).copy())

		# ## check for convergence
		# if i > 10 and stop_eps > 0: # at least 10 iterationss
		# 	if onp.abs(log[-1]) < stop_eps:
		# 		break

	if ifDisplay : # final traj
		plt.figure(figsize=(5,5))
		xf, tr = ergodic_metric.GetTrajXY(u, x0)
		X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
		plt.contourf(X, Y, erg_calc.phik_recon, levels=np.linspace(np.min(erg_calc.phik_recon), np.max(erg_calc.phik_recon),100), cmap='gray')
		# plt.scatter(tr[:,0],tr[:,1], c='r', marker="*:")
		plt.plot(tr[0,0],tr[0,1], "ro:")
		plt.plot(tr[:,0],tr[:,1], "r.:")
		plt.axis("off")
		plt.pause(1)
		plt.savefig("build/plot_traj/MOES-O2-nA_"+str(nA)+"_num_"+str(kkk)+".png", bbox_inches='tight',dpi=200)

	return get_params(opt_state), log, i


def GPErgCover(pdf, nA, s0, nPix, nIter, fourier_freqs=None, freq_vars=None, u_init=None, stop_eps=-1, pdf_gt=None, scale=1):
	"""
	run ergodic coverage over a info map. Modified from public_moes, which was modified from Ian's code.
	return a list of control inputs.
	"""
	if fourier_freqs is None:
		n_fourier = 10
	else:
		n_fourier = len(fourier_freqs)
	
	print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	erg_calc = ergodic_metric.GPErgCalc(pdf, fourier_freqs, freq_vars, nPix, scale)

	opt_init, opt_update, get_params = optimizers.adam(1e-3)

	# initial conditions
	x0 = np.array(s0[:3])
	# x0 = np.array([0,0,s0[0],s0[1]])
	u = np.zeros((nA,2))
	if u_init is not None:
		# print(u_init)
		u = np.array(u_init)
	# u = np.ones((nA,2))*0.1
	# u = 0.05*(onp.random.random((nA,2))*2-1) # each component range from [-1,1]*0.05
	opt_state = opt_init(u)
	log = []

	# if stop_eps > 0:
	# 	nIter = int(1e5) # set a large number, stop until converge.

	i = 0
	loss = 1e5 # start with high loss to pass first check
	while (i < nIter) and (loss >= stop_eps):
	# for i in range(nIter):
		g = erg_calc.gradient(get_params(opt_state), x0)
		opt_state = opt_update(i, g, opt_state)
		u = get_params(opt_state)
		erg_loss = erg_calc.fourier_ergodic_loss(u, x0).copy()
		loss = onp.abs(erg_loss)
		log.append(erg_loss)

		if i % 100 == 0:
			print("[INFO] Iteration {:d} of {:d}, ergodic metric is {:4.2f}".format(i, nIter, erg_loss))

		# ## check for convergence
		# if i > 10 and stop_eps > 0: # at least 10 iterationss
		# 	if onp.abs(log[-1]) < stop_eps:
		# 		break
		
		# increment counter
		i += 1

	if pdf_gt is not None:
		erg_gt = ergodic_metric.GPErgCalc(pdf_gt, None, None, nPix, 1).fourier_ergodic_loss(u, x0).copy()
	else:
		erg_gt = None

	return get_params(opt_state), np.array(log), i, erg_gt

