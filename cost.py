import numpy as np
from sympy import *

import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

# Cost Matrices
QQt = np.array(((1000, 0, 0, 0), (0, 100, 0, 0), (0, 0, 1000, 0), (0, 0, 0, 100)))
RRt = 10*np.eye(ni)
QQTt = 10*QQt

# This function computes the cost function and its gradients with respect to the states and the input 
def cost_function(xx, uu, xxref, uuref):
  
  xx = xx[:,None]
  uu = uu[:,None]
  
  xxref = xxref[:,None]
  uuref = uuref[:,None]

  ll_stage = 1/2*((xx - xxref).T @ QQt @ (xx - xxref)) + 1/2*((uu - uuref).T @ RRt @ (uu - uuref))   # stage cost
  ll_term = 1/2*((xx - xxref).T @ QQTt @ (xx - xxref))                                               # terminal cost

  lx = QQt @ (xx - xxref)                           # gradient of the stage cost with respect to x      
  lu = RRt @ (uu - uuref)                           # gradient of the stage cost with respect to u
  lTx = QQTt @ (xx - xxref)                         # gradient of the terminal cost
    
  return ll_stage, lx, lu, lTx, ll_term 

# This function computes the Regularized matrices that will be used for the Newton algorithm 
def Regularization():

  dlxx = np.zeros((ns,ns))
  dlTxx = np.zeros((ns,ns))

  dlxx = QQt                          # hessian of the stage cost with respect to x 
  dluu = RRt                          # hessian of the stage cost with respect to u 
  dlxu = np.zeros((ni,ns))            # hessian of the stage cost with respect to x and u 
  dlTxx = QQTt                        # hessian of the terminal cost with respect to x 

  return dlxx, dluu, dlxu, dlTxx

# This function computes the Regularized matrices that will be used for the trajectory tracking task
def Regularization_tracking():

  dlxx = np.zeros((ns,ns))
  dlTxx = np.zeros((ns,ns))

  dlxx = np.array(((10, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1000, 0), (0, 0, 0, 100)))
  dluu = 0.1*RRt
  dlxu = np.zeros((ni,ns))
  dlTxx = 100*QQTt   

  return dlxx, dluu, dlxu, dlTxx
