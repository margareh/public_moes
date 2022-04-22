
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import datetime

def MOESSolveNSGA2(pbm, p_size, nIter, time_limit=-1):
	algorithm = NSGA2(
	    pop_size=p_size,
	    eliminate_duplicates=True
	)

	termination = get_termination("n_gen", nIter)

	if time_limit > 0:
		str_time_limit = str(datetime.timedelta(seconds=time_limit))
		print("[INFO] NSGA-II will terminate with time limit :", str_time_limit)
		termination = get_termination("time", str_time_limit)

	res = minimize(pbm,
	               algorithm,
	               termination,
	               seed=1,
	               save_history=True,
	               verbose=True)
	X = res.X
	erg_mat = res.F

	u_list = []
	for i in range(X.shape[0]):
		x = X[i]
		dx = x[0 : pbm.nA]
		dy = x[pbm.nA : 2*pbm.nA]
		u = np.stack([np.array(dx),np.array(dy)], axis=-1)
		u_list.append(u)
	
	return erg_mat, u_list
