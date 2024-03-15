import numpy as np
from sympy import *
from cvxpy import *

# This function computes the sigmoid function used in the computation of the "smooth" optimal transition
def sigmoid_function(tt):

  ss = 1/(1 + np.exp(-tt/100))            # ss = 1/1+e^-tt
  ds = ss*(1-ss)                          # ds = d/dt ss(tt)

  return ss, ds

# This function computes the reference position and velocity used in the computation of the reference signals using the sigmoid function
def reference_position(tt, p0, pT, T):

  # In the definition of the function we use:
  # - tt time instant
  # - p0 initial position
  # - pT final position
  # - T time horizon

  pp = p0 + sigmoid_function(tt - T/2)[0]*(pT - p0)         # reference position 
  vv = sigmoid_function(tt - T/2)[1]*(pT - p0)              # reference velocity

  return pp, vv

# This function computes the K,P matrices by exploiting the Difference Riccati Equation and has the possibility to augment the state or not depending on the inputs.
# Then it computes the trajectories of the linearization Delta_x and Delta_u  
def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qq = None, rr = None, qqf = None):

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  # Check for affine terms

  augmented = False

  if qq is not None or rr is not None or qqf is not None:
    augmented = True
    #print("Augmented term!")

  if augmented:
    if qq is None:
      qq = np.zeros(ns)

    if rr is None:
      rr = np.zeros(ni)

    if qqf is None:
      qqf = np.zeros(ns)

    # Check sizes

    try:  
      na, la = qq.shape
    except:
      qq = qq[:,None]
      na, la = qq.shape

    try:  
      nb, lb = rr.shape
    except:
      rr = rr[:,None]
      nb, lb = rr.shape

    if na != ns:
        print("State affine term does not match states dimension")
        exit()
    if nb != ni:
        print("Input affine term does not match inputs dimension")
        exit()
    if la == 1:
        qq = qq.repeat(TT, axis=1)
    if lb == 1:
        rr = rr.repeat(TT, axis=1)

  # Build matrices

  if augmented:

    KK = np.zeros((ni, ns + 1, TT))
    PP = np.zeros((ns+1, ns+1, TT))

    QQ = np.zeros((ns + 1, ns + 1, TT))
    QQf = np.zeros((ns + 1, ns + 1))
    SS = np.zeros((ni, ns + 1, TT))
    RR = np.zeros((ni, ni, TT))                   # Must be positive definite

    AA = np.zeros((ns + 1, ns + 1, TT))
    BB = np.zeros((ns + 1, ni, TT))

    # Augmented matrices
    for tt in range(TT):

      # Cost

      QQ[1:, 0, tt] = 0.5 * qq[:,tt]
      QQ[0, 1:, tt] = 0.5 * qq[:,tt].T
      QQ[1:, 1:, tt] = QQin[:, :, tt]

      RR[:, :, tt] = RRin[:, :, tt]

      SS[:, 0, tt] = 0.5 * rr[:, tt]
      SS[:,1:,tt] = SSin[:, :, tt]

      # System

      AA[0, 0, tt] = 1
      AA[1:, 1:, tt] = AAin[:, :, tt]
      BB[1:, :, tt] = BBin[:, :, tt]

    QQf[1:, 0] = 0.5 * qqf
    QQf[0, 1:] = 0.5 * qqf.T
    QQf[1:, 1:] = QQfin

    # Systems trajectory
    xx = np.zeros((ns + 1, TT))
    uu = np.zeros((ni, TT))
    # Augmented state
    xx[0, :].fill(1)
    xx[1:,0] = x0

  else:
    KK = np.zeros((ni, ns, TT))
    PP = np.zeros((ns, ns, TT))

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:,0] = x0
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
        + (BBt.T@PPtp@AAt + SSt).T@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt + SSt)
  
  # Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]

    # Check positive definiteness

    MM = RRt + BBt.T@PPtp@BBt

    if not np.all(np.linalg.eigvals(MM) > 0):

      # Regularization
      print('regularization at tt = {}'.format(tt))
      MM += 0.5*np.eye(ni)

    KK[:,:,tt] = -np.linalg.inv(MM)@(BBt.T@PPtp@AAt + SSt)


  

  for tt in range(TT - 1):
    # Trajectory

    uu[:, tt] = KK[:,:,tt] @ xx[:, tt]
    xx_p = AA[:,:,tt] @ xx[:,tt] + BB[:,:,tt] @ uu[:, tt]

    xx[:,tt+1] = xx_p.squeeze()

    if augmented:
      xxout = xx[1:,:]
    else:
      xxout = xx

    uuout = uu

  return KK, PP, xxout, uuout
    