
import sys
import numpy as np

import jax.numpy as jnp
from jax import vmap
from ergodic_coverage import ErgCover
import ergodic_metric

import matplotlib.pyplot as plt
import time


def ScalarizeInfoMaps(info_maps, weights):
  """
  input info_maps can be either prob distribution or phik.
  Note: Arbitrary #obj.
  """
  if len(info_maps) != len(weights):
    sys.exit("[ERROR] ScalarizeInfoMaps, input info_maps and weights have diff len!")
  if abs(np.sum(weights)-1) > 1e-6:
    sys.exit("[ERROR] ScalarizeInfoMaps, input weights do not sum to one!")
  out = np.zeros(info_maps[0].shape)
  for i, info_map in enumerate(info_maps):
    out += info_map*weights[i]
  return out

def RunScalarizeMethodOnce(pbm, w, n_basis, n_iter, ifDisplay, u_init=None, stop_eps=1e-2, kkk=0):
  """
  w - a weight vector.
  pbm - the problem object.
  Note: Arbitrary #obj.
  """
  pdfs = [pbm.pdf1, pbm.pdf2]
  if len(w) == 3:
    pdfs = [pbm.pdf1, pbm.pdf2, pbm.pdf3]
  weighted_pdf = ScalarizeInfoMaps(pdfs, w)

  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  pdf = jnp.asarray(weighted_pdf.flatten())
  controls, ergs, iters = ErgCover(pdf, pbm.nA, pbm.s0, n_basis, pbm.pix, n_iter, ifDisplay, u_init, stop_eps, kkk)
  return controls, ergs, iters, weighted_pdf

def UniformGenerateWeights(n_weight, n_obj):
  """
  """
  out = list()
  if n_obj == 2:
    w1_list = np.linspace(0,1,n_weight)
    for w1 in w1_list:
      out.append( np.array( [w1,1-w1] ) )
  elif n_obj == 3:
    w_list = np.linspace(0,1,n_weight)
    for w1 in w_list:
      for w2 in w_list:
        if w1 + w2 > 1:
          break
        out.append( np.array( [1-w1-w2,w1,w2] ) )
  return out

def ErgodicDiff(calc1,calc2):
  """
  """
  lamk = calc1.lamk
  return np.sqrt( np.sum(lamk*np.square(calc1.phik - calc2.phik)) )

def AdaptiveGenerateWeights(pbm, n_obj, delta=0.1):
  """
  """
  out = list()
  if n_obj == 2:
    diff12 = ErgodicDiff(pbm.calc1, pbm.calc2)
    # print("[INFO] AdaptiveGenerateWeights, diff12 = ", diff12)
    dw12 = delta/diff12
    w1_list = np.linspace(0, 1, int(np.floor(1/dw12))+2) # at least two points
    for w1 in w1_list:
      out.append( np.array( [1-w1,w1] ) )
  elif n_obj == 3:
    diff12 = ErgodicDiff(pbm.calc1, pbm.calc2)
    dw12 = delta/diff12
    diff13 = ErgodicDiff(pbm.calc1, pbm.calc3)
    dw13 = delta/diff13
    # print("[INFO] AdaptiveGenerateWeights, diff12 = ", diff12, " diff13 = ", diff13)
    w1_list = np.linspace(0, 1, int(np.floor(1/dw12))+2)
    flip = False
    for w1 in w1_list:
      w2_list = np.linspace(0, 1, int(np.floor(1/dw13))+2)
      if flip:
        w2_list = np.linspace(1, 0, int(np.floor(1/dw13))+2)
      for w2 in w2_list:
        if w1 + w2 > 1:
          flip = not flip
          break
        out.append( np.array( [1-w1-w2,w1,w2] ) )
  return out

def MOESSolveScalarize(pbm, n_weight=11, n_basis=10, n_iter=1000, seqOptm=False, ifAdaptive=False, stop_eps=0.05, ifDisplay=False, delta=0.1):
  """
  n_weight defines the number of intervals to divide [0,1] for each weight component.
  """
  w1_list = np.linspace(0,1,n_weight)
  n_obj=2
  if hasattr(pbm, 'pdf3'):
    n_obj=3
  weight_list = UniformGenerateWeights(n_weight, n_obj)
  if ifAdaptive:
    weight_list = AdaptiveGenerateWeights(pbm, n_obj, delta)
  erg_metric_mat = np.zeros( (len(weight_list), n_obj) ) # should be two obj for now. @2021-12-18
  u_list = list()
  pdf_list = list()
  erg_list = list()
  time_list = list()
  iter_list = list()
  u_init = None

  # for j, w1 in enumerate(w1_list): ### assuming two objectives
  for j, weight in enumerate(weight_list):
    # weight = np.array( [w1,1-w1] )
    # print("weight = ", weight)
    if not seqOptm:
      u_init = None
    tnow = time.perf_counter()
    u, ergs, iters, weighted_pdf = RunScalarizeMethodOnce(pbm, weight, n_basis, n_iter, ifDisplay, u_init, stop_eps, j)
    used_time = time.perf_counter()-tnow
    time_list.append(used_time)
    u_list.append(u)
    erg_list.append(ergs)
    pdf_list.append(weighted_pdf)
    iter_list.append(iters)
    u_init = u_list[-1] # the solution from the last time.

    erg_vec = np.array([pbm.calc1.fourier_ergodic_loss(u, pbm.s0), pbm.calc2.fourier_ergodic_loss(u, pbm.s0)])
    if n_obj == 3:
      erg_vec = np.array([pbm.calc1.fourier_ergodic_loss(u, pbm.s0), pbm.calc2.fourier_ergodic_loss(u, pbm.s0), pbm.calc3.fourier_ergodic_loss(u, pbm.s0)])
    # print("Get erg_vec = ", erg_vec)
    erg_metric_mat[j,:] = erg_vec
  
  # print("[INFO] MOESSolveScalarize use time = ", time_list)
  return erg_metric_mat, u_list, pdf_list, time_list, erg_list, iter_list
