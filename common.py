

import pickle
import numpy as np
import copy

import pymoo.model.problem as pmo_p

import ergodic_metric
import distributions
from distributions import gaussianMixtureDistribution

import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import rfft, irfft, fft, fftfreq
import ergodic_metric
import jax.numpy as jnp


def SavePickle(obj, file_path):
    """
    Save a serie of tests as pickle file.
    """
    pickle_out = open(file_path,"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    return

def LoadPickle(file_path):
    """
    """
    pickle_in = open(file_path,"rb")
    obj = pickle.load(pickle_in)
    return obj

class MOESProblemO2(pmo_p.Problem):
    """
    define a problem.
    """
    def __init__(self):
        """
        """
        return

    def initFromDic(self, dic):
        self.init(dic["s0"], dic["nA"], dic["pdfs"], dic["nPixel"])

    def init(self, s0, nA, pdfs, pix):
        self.s0 = np.array([s0[0],s0[1],0]) # (x,y,theta)
        self.nA = nA
        # self.nA = 50 # nA # planning horizon
        self.pix = pix
        self.pdf1 = pdfs[0]
        self.pdf2 = pdfs[1]
        self.calc1 = copy.deepcopy(ergodic_metric.ErgCalc(self.pdf1.flatten(), 10, 100))
        self.calc2 = copy.deepcopy(ergodic_metric.ErgCalc(self.pdf2.flatten(), 10, 100))

        max_u = 0.05 # bound the control input for evolutionary computation
        xu = np.concatenate([max_u*np.ones(self.nA), max_u*np.ones(self.nA)])
        xl = -xu

        super().__init__(n_var=2*self.nA,
                         n_obj=2,
                         n_constr=4,
                         xl=xl, # just some bounds on controls
                         xu=xu,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        X - population
        """
        dx = x[0:self.nA]
        dy = x[self.nA:2*self.nA]
        u = np.stack([np.array(dx),np.array(dy)], axis=-1)

        f1 = self.calc1.fourier_ergodic_loss(u, self.s0).copy()
        f2 = self.calc2.fourier_ergodic_loss(u, self.s0).copy()

        out["F"] = [f1,f2]
        out["G"] = [0,0] # this means basically no constraints on control input.

        return

class MOESProblemO3(pmo_p.Problem):
    """
    define a problem.
    """
    def __init__(self):
        """
        """
        return

    def initFromDic(self, dic):
        self.init(dic["s0"], dic["nA"], dic["pdfs"], dic["nPixel"])

    def init(self, s0, nA, pdfs, pix):
        self.s0 = np.array([s0[0],s0[1],0]) # (x,y,theta)
        self.nA = nA
        # self.nA = 50 # nA # planning horizon
        self.pix = pix
        self.pdf1 = pdfs[0]
        self.pdf2 = pdfs[1]
        self.pdf3 = pdfs[2]
        self.calc1 = copy.deepcopy(ergodic_metric.ErgCalc(self.pdf1.flatten(), 10, 100))
        self.calc2 = copy.deepcopy(ergodic_metric.ErgCalc(self.pdf2.flatten(), 10, 100))
        self.calc3 = copy.deepcopy(ergodic_metric.ErgCalc(self.pdf3.flatten(), 10, 100))

        max_u = 0.05 # bound the control input for evolutionary computation
        xu = np.concatenate([max_u*np.ones(self.nA), max_u*np.ones(self.nA)])
        xl = -xu

        super().__init__(n_var=2*self.nA,
                         n_obj=2,
                         n_constr=4,
                         xl=xl, # just some bounds on controls
                         xu=xu,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        X - population
        """
        dx = x[0:self.nA]
        dy = x[self.nA:2*self.nA]
        u = np.stack([np.array(dx),np.array(dy)], axis=-1)

        f1 = self.calc1.fourier_ergodic_loss(u, self.s0).copy()
        f2 = self.calc2.fourier_ergodic_loss(u, self.s0).copy()
        f3 = self.calc3.fourier_ergodic_loss(u, self.s0).copy()

        out["F"] = [f1,f2,f3]
        out["G"] = [0,0] # this means basically no constraints on control input.

        return


def GenMOESProblemO2Simple(nA, pbm_file_name):
    """
    """
    s0 = np.array([.5, .5]) # initial state of robot
    # nA = 100 # number of actions
    pix = 100 # number of pixels for plotting
    k = 0
    save_dir = "build/instances/"

    mu = np.array([[.3, .3]])
    cov = np.array([[[.02, 0], [0, .02]]])
    pdf1 = gaussianMixtureDistribution(1, pix, mus=mu, covs=cov)

    mu = np.array([[.7, .7]])
    cov = np.array([[[.02, 0], [0, .02]]])
    pdf2 = gaussianMixtureDistribution(1, pix, mus=mu, covs=cov)
    
    # # bounds on linear velocities and angular velocities.
    # ub = np.concatenate([1*np.ones(nA), 1*np.ones(nA)])
    # lb = -ub

    dic = dict()
    dic["s0"] = s0
    dic["nA"] = nA
    dic["pdfs"] = [pdf1,pdf2]
    dic["nPixel"] = pix

    SavePickle(dic, pbm_file_name)

def GenMOESProblemO3Simple(nA, pbm_file_name):
    """
    three-objective problem
    """
    s0 = np.array([.5, .5, 0]) # initial state of robot, add orientation 0
    # nA = 100 # number of actions
    pix = 100 # number of pixels for plotting
    k = 0
    save_dir = "build/instances/"

    mu = np.array([[.3, .3]])
    cov = np.array([[[.02, 0], [0, .02]]])
    pdf1 = gaussianMixtureDistribution(1, pix, mus=mu, covs=cov)

    mu = np.array([[.7, .7]])
    cov = np.array([[[.02, 0], [0, .02]]])
    pdf2 = gaussianMixtureDistribution(1, pix, mus=mu, covs=cov)

    mu = np.array([[.2, .5]])
    cov = np.array([[[.01, 0], [0, .03]]])
    pdf3 = gaussianMixtureDistribution(1, pix, mus=mu, covs=cov)
    
    # # bounds on linear velocities and angular velocities.
    # ub = np.concatenate([1*np.ones(nA), 1*np.ones(nA)])
    # lb = -ub

    dic = dict()
    dic["s0"] = s0
    dic["nA"] = nA
    dic["pdfs"] = [pdf1,pdf2,pdf3]
    dic["nPixel"] = pix

    SavePickle(dic, pbm_file_name)
    return

def LoadProblem(instance_path):
    """
    """
    dic = LoadPickle(instance_path)
    pbm = MOESProblemO2()
    if len(dic['pdfs']) == 3:
        pbm = MOESProblemO3()
    pbm.initFromDic(dic)
    return pbm

def GenMOESProblemO2Random(nGau, k, nA, pbm_file_name):
    """
    """
    s0 = np.array([.5, .5]) # initial state of robot
    # nA = 100 # number of actions
    pix = 100 # number of pixels for plotting
    k = 0
    save_dir = "build/instances/"

    # generate two probability distributions
    pdf1 = gaussianMixtureDistribution(nGau, pix)
    pdf2 = gaussianMixtureDistribution(nGau, pix)

    # pdf1 = distributions.gauMixDistrib(nGau, pix)
    # pdf2 = distributions.gauMixDistrib(nGau, pix)

    dic = dict()
    dic["s0"] = s0
    dic["nA"] = nA
    dic["pdfs"] = [pdf1,pdf2]
    dic["nPixel"] = pix

    # problem = MOESProblemO2()
    # problem.initFromDic(dic)

    # file_name = "MOES-O2-nA_"+str(nA)+"_nGau_"+str(1)+"_pix_"+str(pix)+"_simple"
    SavePickle(dic, pbm_file_name)

