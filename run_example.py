
import matplotlib.pyplot as plt

import common
import nsga
import scalarize
import bfs_o3

import numpy as np
import ergodic_metric


def run_o2():
  """
  """
  k = 1 # the k-th run. Need to run many times to observe the trend due to the randomness in NSGA and MOEA/D.

  ###### obj=2 test #########
  pbm_file = "sample_instances/MOES-O2-nA_"+str(100)+"_nGau_"+str(1)+"_pix_100_simple.pickle"
  ###### obj=2 test #########

  # ####### obj=3 test #########
  # pbm_file = "build/instances/MOES-O3-nA_"+str(100)+"_nGau_"+str(1)+"_pix_100_simple.pickle"
  # ####### obj=3 test #########
  
  problem = common.LoadProblem(pbm_file)
  problem.nA = 50

  total_iter = 1000
  n_scalar = 11

  print("********* Run *********")

  ###### obj=2 test #########

  erg_mat, u_list, pdf_list_scala, time_list, erg_list, iter_list = \
    scalarize.MOESSolveScalarize(problem, n_scalar, 10, total_iter, seqOptm=True, ifAdaptive=True)

  # erg_mat, u_list, pdf_list_scala, time_list, erg_list, iter_list = \
  #   scalarize.MOESSolveScalarize(problem, n_scalar, 10, total_iter, seqOptm=False, ifAdaptive=False)

  ###### obj=2 test #########

  #### temp plot traj time avg statistics.
  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  fig, axs = plt.subplots(2, 3)
  axs[0, 0].contourf(X, Y, problem.pdf1, levels=np.linspace(np.min(problem.pdf1), np.max(problem.pdf1),100), cmap='gray')
  axs[0, 0].set_title('First Info Map')

  axs[0, 1].contourf(X, Y, problem.pdf2, levels=np.linspace(np.min(problem.pdf2), np.max(problem.pdf2),100), cmap='gray')
  axs[0, 1].set_title('Second Info Map')

  kk = 2 # which traj to show
  axs[0, 2].contourf(X, Y, pdf_list_scala[kk], levels=np.linspace(np.min(problem.pdf2), np.max(problem.pdf2),100), cmap='gray')
  axs[0, 2].set_title('Scalarized Info Map')

  tj_stat = problem.calc1.traj_stat(u_list[kk], problem.s0)
  axs[1, 0].contourf(X, Y, tj_stat, levels=np.linspace(np.min(tj_stat), np.max(tj_stat),100), cmap='gray')
  _, tj = ergodic_metric.GetTrajXY(u_list[kk], problem.s0)
  axs[1, 0].plot(tj[:,0]*100,tj[:,1]*100, "r.--")
  axs[1, 0].set_title('Traj Time Avg. Stat.')

  axs[1, 1].plot(erg_mat[:,0], erg_mat[:,1], 'ro', alpha=0.5, label="Aggregation")
  axs[1, 0].set_title('Pareto-Optimal Front')

  plt.show()

  return 

def run_o3():
  """
  """
  k = 1 # the k-th run. Need to run many times to observe the trend due to the randomness in NSGA and MOEA/D.

  ####### obj=3 test #########
  pbm_file = "sample_instances/MOES-O3-nA_"+str(100)+"_nGau_"+str(1)+"_pix_100_simple.pickle"
  ####### obj=3 test #########
  
  problem = common.LoadProblem(pbm_file)
  problem.nA = 50

  total_iter = 1000

  print("********* Run *********")

  ####### obj=3 test #########
  bfs = bfs_o3.MOES_BFS_O3(0.2)
  erg_mat, u_list, pdf_list_scala, time_list, erg_list, iter_list = \
    bfs.solve(problem, n_basis=10, n_iter=total_iter, stop_eps=0.001, ifDisplay=False, delta=0.1)
  ####### obj=3 test #########

  #### temp plot traj time avg statistics.
  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  fig, axs = plt.subplots(2, 3)
  axs[0, 0].contourf(X, Y, problem.pdf1, levels=np.linspace(np.min(problem.pdf1), np.max(problem.pdf1),100), cmap='gray')
  axs[0, 0].set_title('First Info Map')

  axs[0, 1].contourf(X, Y, problem.pdf2, levels=np.linspace(np.min(problem.pdf2), np.max(problem.pdf2),100), cmap='gray')
  axs[0, 1].set_title('Second Info Map')

  kk = 2 # which traj to show
  axs[0, 2].contourf(X, Y, problem.pdf3, levels=np.linspace(np.min(problem.pdf3), np.max(problem.pdf3),100), cmap='gray')
  axs[0, 2].set_title('Third Info Map')

  tj_stat = problem.calc1.traj_stat(u_list[kk], problem.s0)
  axs[1, 0].contourf(X, Y, tj_stat, levels=np.linspace(np.min(tj_stat), np.max(tj_stat),100), cmap='gray')
  _, tj = ergodic_metric.GetTrajXY(u_list[kk], problem.s0)
  axs[1, 0].plot(tj[:,0]*100,tj[:,1]*100, "r.--")
  axs[1, 0].set_title('Traj Time Avg. Stat.')

  axs[1, 1].plot(erg_mat[:,0], erg_mat[:,1], 'ro', alpha=0.5, label="Aggregation")
  axs[1, 0].set_title('Pareto-Optimal Front')

  plt.show()

  return 


if __name__ == "__main__":

  run_o2()
  
  # run_o3()


