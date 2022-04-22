
import time
import queue
import sys
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d Pareto-front

import scalarize
import ergodic_metric

def constructTriangle(l1,l2,l3):
  """
  """
  if (l1+l2 < l3) or (l2+l3 < l1) or (l1+l3 < l2):
    print(" l123 = ", l1, l2, l3, " cannot form a triangle...")
    sys.exit("[ERROR] Given numbers cannot form a triangle!")
  pt1 = [0,0]
  pt2 = [l1,0]
  # law of consine
  theta = numpy.arccos( (l1**2 + l2**2 - l3**2) / (2*l1*l2) )
  print(" theta = ", theta)
  pt3 = [numpy.cos(theta)*l2, numpy.sin(theta)*l2]
  print("constructed triangle:", pt1, pt2, pt3)
  return pt1,pt2,pt3

def findAffineTF(pt1,pt2,pt3):
  """
  the an affine map x'=Ax+b, that can map each corner point of triangle pt1,pt2,pt3
  to a triangle (0,0),(0,1),(1,0)
  Note that b=(0,0) since pt1 is always zero.
  """
  if (pt1[0] != 0) or (pt1[1] != 0) or (pt2[1] != 0):
    sys.exit("[ERROR] input triangle pt1 is not (0,0), not implemented.")
  # a11*l1 + a12*0 = 1
  a11 = 1.0/pt2[0]
  # a21*l1 + a22*0 = 0
  a21 = 0
  # a11*pt3x + a12*pt3y = 0
  a12 = -a11*pt3[0]/pt3[1]
  # a21*pt3x + a22*pt3y = 1
  a22 = 1.0/pt3[1]
  # return A, b
  return numpy.array([[a11,a12],[a21,a22]]), numpy.array([0,0])

class MOES_BFS_O3(object):
  """
  only implement for three objectives
  """
  def __init__(self, step_size, adaptive=True):
    self.open_queue = queue.Queue()
    self.parent_dict = dict()
    self.ngh_x = [1,-1,0,0]
    self.ngh_y = [0,0,1,-1]
    self.step_size = step_size
    self.fig = None
    self.adaptive = adaptive

    ### each weight vector has a (tuple) id (int x, int y). Init ID is (0,0).
    self.A = []
    self.b = [] # affine transformation Ax+b.
    self.w_dict = dict() # map weight ID to the weight values, generated
    self.u_dict = dict() # map weight ID to the computed controls, expanded/closed

  def solve(self, pbm, n_basis=10, n_iter=1000, stop_eps=0.05, ifDisplay=False, delta=0.1):
    
    ###### init ############
    if not hasattr(pbm, 'pdf3'):
      sys.exit("[ERROR] input problem does not have pdf3, not a three-objective problem!")
    self.pbm = pbm
    self._initialize()

    if ifDisplay:
      self.fig = plt.figure(figsize=(8,8)) #subplots(2,2)
      ### plot-1
      self.ax1 = self.fig.add_subplot(2, 2, 1)
      ### plot-2
      self.ax2 = self.fig.add_subplot(2, 2, 2, projection='3d')
      self.ax2.set_xlabel('x')
      self.ax2.set_ylabel('y')
      self.ax2.set_zlabel('z')
      self.ax2.set_xlim([0,1])
      self.ax2.set_ylim([0,1])
      self.ax2.set_zlim([0,1])
      self.ax2.view_init(elev=40., azim=120)
      ### plot-3
      self.ax3 = self.fig.add_subplot(2, 2, 3)
      ### plot-4
      self.ax4 = self.fig.add_subplot(2, 2, 4)

    erg_list = list()
    u_list = list()
    time_list = list()
    iter_list = list()
    pdf_list = list()

    last_wID = -1 # for plot only
    last_erg_pt = -1 # for plot only

    ###### main loop ############
    kkk = 0
    while not self.open_queue.empty():
      wID = self.open_queue.get()
      pID = self.parent_dict[wID]
      w = self._getWeight(wID)
      print(">>>> pop wID = ", wID, " weight = ", w )
      # run ergodic coverage
      tnow = time.perf_counter()
      controls, ergs, iters, weighted_pdf = \
        scalarize.RunScalarizeMethodOnce(self.pbm, w, n_basis, n_iter, ifDisplay=False, u_init=self.u_dict[pID], stop_eps=stop_eps, kkk=0)
      used_time = time.perf_counter()-tnow
      # store data
      self.u_dict[wID] = controls
      u_list.append(controls)
      val1 = self.pbm.calc1.fourier_ergodic_loss(controls, self.pbm.s0)
      val2 = self.pbm.calc2.fourier_ergodic_loss(controls, self.pbm.s0)
      val3 = self.pbm.calc3.fourier_ergodic_loss(controls, self.pbm.s0)
      erg_list.append([val1,val2,val3])
      time_list.append(used_time)
      iter_list.append(iters)
      pdf_list.append(weighted_pdf)
      # generate neighboring weights
      nghs = self._neighbor(wID)
      new_ngh = list()
      for wID2 in nghs:
        if wID2 in self.w_dict:
          # already generated
          continue
        self.w_dict[ wID2 ] = self._getWeight(wID2)
        self.open_queue.put(wID2)
        self.parent_dict[wID2] = wID
        new_ngh.append(wID2)
        print(">>>> wID2 = ", wID2, " weight = ", self.w_dict[ wID2 ] )


      if ifDisplay:

        self.ax1.plot([self.pt1[0],self.pt2[0]], [self.pt1[1],self.pt2[1]], "k--")
        self.ax1.plot([self.pt2[0],self.pt3[0]], [self.pt2[1],self.pt3[1]], "k--")
        self.ax1.plot([self.pt3[0],self.pt1[0]], [self.pt3[1],self.pt1[1]], "k--")

        if last_wID != -1:
          self.ax1.plot(self.step_size*last_wID[0], self.step_size*last_wID[1], "bo")
        self.ax1.plot(self.step_size*wID[0], self.step_size*wID[1], "ro")
        for wID2 in new_ngh:
          self.ax1.plot(self.step_size*wID2[0], self.step_size*wID2[1], "go")
        self.ax1.axis('equal')

        if last_erg_pt != -1:
          self.ax2.scatter(last_erg_pt[0], last_erg_pt[1], last_erg_pt[2], c='b')
        self.ax2.scatter(val1, val2, val3, c='r')

        self.ax3.clear()
        xf, tr = ergodic_metric.GetTrajXY(controls, self.pbm.s0)
        X,Y = numpy.meshgrid(*[numpy.linspace(0,1,num=self.pbm.pix)]*2)
        self.ax3.contourf(X, Y, weighted_pdf, levels=numpy.linspace(numpy.min(weighted_pdf), numpy.max(weighted_pdf),100), cmap='gray')

        self.ax4.clear()
        self.ax4.contourf(X, Y, weighted_pdf, levels=numpy.linspace(numpy.min(weighted_pdf), numpy.max(weighted_pdf),100), cmap='gray')
        self.ax4.plot(tr[0,0],tr[0,1], "ro:")
        self.ax4.plot(tr[:,0],tr[:,1], "r.:")
        # plt.savefig("build/plot_traj/MOES-O3-nA_"+str(self.pbm.nA)+"_num_"+format(kkk, '05d')+".png", bbox_inches='tight', dpi=200)

        plt.draw()
        plt.pause(1)

      kkk += 1 # episode
      last_wID = wID
      last_erg_pt = [val1,val2,val3]
    # end while loop

    # post-process
    erg_mat = numpy.array(erg_list)
    return erg_mat, u_list, pdf_list, time_list, [], iter_list

  def _neighbor(self, wID):
    """
    """
    out = list()
    for i in range(len(self.ngh_x)):
      wID2 = (wID[0] + self.ngh_x[i], wID[1] + self.ngh_y[i])
      if self._isValid(wID2):
        out.append(wID2)
    return out

  def _getWeight(self, wID):
    """
    """
    x = numpy.array([self.step_size*wID[0], self.step_size*wID[1]])
    x0 = self.A.dot(x) + self.b
    w0 = 1-numpy.sum(x0)
    return numpy.array([x0[0],w0,x0[1]])

  def _isValid(self, wID):
    """
    """
    w = self._getWeight(wID)
    if numpy.sum(w>=0) < 3 or numpy.sum(w<=1) < 3:
      return False
    return True

  def _initialize(self):
    """
    """
    ### compute ergodic metrics between info maps, which defines a triangle.
    self.erg12 = scalarize.ErgodicDiff(self.pbm.calc1, self.pbm.calc2)
    self.erg23 = scalarize.ErgodicDiff(self.pbm.calc2, self.pbm.calc3)
    self.erg31 = scalarize.ErgodicDiff(self.pbm.calc3, self.pbm.calc1)

    print(">>> computed ergs = ", self.erg12, self.erg23, self.erg31)

    ### find the transformation
    # this TF should take a weight ID (e.g. (3,4)), return a weight vector (w1,w2,w3).
    self.pt1, self.pt2, self.pt3 = constructTriangle(self.erg12, self.erg23, self.erg31)
    self.A, self.b = findAffineTF(self.pt1,self.pt2,self.pt3)

    ### add to OPEN, init some data
    # w0 = tuple([6,1]) # for that video
    w0 = tuple([0,0])
    self.open_queue.put(w0)
    self.w_dict[w0] = self._getWeight(w0)
    self.parent_dict[w0] = -1
    self.u_dict[-1] = None

    return

