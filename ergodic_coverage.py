
import numpy as onp
import jax.numpy as np
# from jax.experimental import optimizers
# from jax.example_libraries import optimizers
import optax
import copy

import matplotlib.pyplot as plt
import public_moes.ergodic_metric as ergodic_metric
from public_moes.ergodic_metric import GetTrajXY

GLOBAL_NUM_K = 0

def ErgCover(pdf, nA, s0, n_fourier, nPix, nIter, ifDisplay, u_init=None, stop_eps=-1, kkk=0):
	"""
	run ergodic coverage over a info map. Modified from Ian's code.
	return a list of control inputs.
	"""
	print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	erg_calc = ergodic_metric.ErgCalc(pdf, n_fourier, nPix)

	# opt_init, opt_update, get_params = optimizers.adam(1e-3)
	optim = optax.adam(1e-3)

	# initial conditions
	x0 = np.array(s0[:3])
	# x0 = np.array([0,0,s0[0],s0[1]])
	u = np.zeros((nA,2))
	if u_init is not None:
		# print(u_init)
		u = np.array(u_init)
	# u = np.ones((nA,2))*0.1
	# u = 0.05*(onp.random.random((nA,2))*2-1) # each component range from [-1,1]*0.05
	# opt_state = opt_init(u)
	opt_state = optim.init(u)
	log = []

	# if stop_eps > 0:
	# 	nIter = int(1e5) # set a large number, stop until converge.

	i = 0
	loss = 1e5
	while (i < nIter) and (loss >= stop_eps):
	# for i in range(nIter):
		# g = erg_calc.gradient(get_params(opt_state), x0)
		# opt_state = opt_update(i, g, opt_state)
		# u = get_params(opt_state)
		g = erg_calc.gradient(u, x0)
		updates, opt_state = optim.update(g, opt_state)
		u = optax.apply_updates(u, updates)
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

	return u, log, i


def GPErgCover(pdf, nA, s0, nPix, nIter, fourier_freqs=None, freq_vars=None, u_init=None, stop_eps=-1, pdf_gt=None, scale=1, lr=5e-4):
	"""
	run ergodic coverage over a info map. Modified from public_moes, which was modified from Ian's code.
	return a list of control inputs.
	"""
	if fourier_freqs is None:
		n_fourier = 10
	else:
		n_fourier = len(fourier_freqs)

	# scale the pdf to [0,1]
	pdf_norm = (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf))
	# want all info in pdf to sum to 1
	# pdf_norm = pdf / np.sum(pdf)
	
	print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	erg_calc = ergodic_metric.GPErgCalc(pdf_norm, fourier_freqs, freq_vars, nPix, scale)

	# opt_init, opt_update, get_params = optimizers.adam(step_size=lr, b1=0.9, b2=0.999)
	optim = optax.adam(lr)

	# initial conditions
	x0 = np.array(s0[:3])
	# x0 = np.array([0,0,s0[0],s0[1]])
	u = np.zeros((nA,2))
	if u_init is not None:
		# print(u_init)
		u = np.array(u_init)
	# u = np.ones((nA,2))*0.1
	# u = 0.05*(onp.random.random((nA,2))*2-1) # each component range from [-1,1]*0.05
	# opt_state = opt_init(u)
	opt_state = optim.init(u)
	log = []

	# if stop_eps > 0:
	# 	nIter = int(1e5) # set a large number, stop until converge.

	i = 0
	loss = 1e5 # start with high loss to pass first check
	prev_loss = 0
	no_change_count = 0
	print_flag = False
	while (i < nIter) and (loss >= stop_eps):
	# for i in range(nIter):

		# stop if haven't changed for a set number of iters
		if no_change_count > 200:
			break

		# g = erg_calc.gradient(get_params(opt_state), x0)
		# opt_state = opt_update(i, g, opt_state)
		# u = get_params(opt_state)
		g = erg_calc.gradient(u, x0)
		updates, opt_state = optim.update(g, opt_state)
		u = optax.apply_updates(u, updates)

		print_flag = True if i % 100 == 0 else False
		erg_loss = erg_calc.fourier_ergodic_loss(u, x0, print_flag).copy()

		loss = onp.abs(erg_loss)
		log.append(erg_loss)
		
		if i > 0:
			if np.abs(loss - prev_loss) < 0.001:
				no_change_count += 1
			else:
				no_change_count = 0
		prev_loss = copy.copy(loss)

		if i % 100 == 0:
			print("[INFO] Iteration {:d} of {:d}, ergodic metric is {:4.2f}".format(i, nIter, erg_loss))		

		# ## check for convergence
		# if i > 10 and stop_eps > 0: # at least 10 iterationss
		# 	if onp.abs(log[-1]) < stop_eps:
		# 		break
		
		# increment counter
		i += 1

	# print final parameters of ergodic calculation
	print("Freqs:")
	print(erg_calc.k)
	print("Lambda: ")
	print(erg_calc.lamk)
	print("Phi:")
	print(erg_calc.phik)
	print("C:")
	_, tr = GetTrajXY(u, x0)
	print(erg_calc.get_ck(tr))

	# get the map predicted by the trajectory stats
	traj_map = erg_calc.traj_stat(u, x0)

	# get ergodic metric on ground truth map if available using usual frequencies
	if pdf_gt is not None:
		erg_gt = ergodic_metric.GPErgCalc(pdf_gt, None, None, nPix, 1).fourier_ergodic_loss(u, x0).copy()
	else:
		erg_gt = None

	# get loss components for final trajectory
	loss_components = erg_calc.loss_components(u, x0)

	return u, np.array(log), i, erg_gt, traj_map, loss_components

