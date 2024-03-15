import numpy as np
import sympy as sp

ns = 4                              # number of states
ni = 1                              # number of inputs


# Dynamics parameters

mm = 0.2                            # leg mass
mh = 2                              # body mass
aa = 0.7
bb = 0.5
ll = aa+bb                          # leg length
gg = 9.81

dt = 1e-3                           # discretization gamma - Forward Euler

# xx1 = theta_sw 
# xx2 = theta_sw_dot
# xx3 = theta_st
# xx4 = theta_st_dot
# uu = tau

# This function computes the update of the discretized dynamics
def statedot(xx1, xx2, xx3, xx4, uu):  

  c31 = np.cos(xx3-xx1)
  s1 = np.sin(xx1)
  s31 = np.sin(xx3-xx1)
  s3 = np.sin(xx3)

  x1dot = xx1 + dt*xx2

  x2dot = xx2 + dt*((1/(mm*bb**2))*(uu - mm*ll*bb*s31*xx4**2 - mm*bb*gg*s1 + mm*ll*bb*c31*((1/(((mh+mm)*ll**2 + mm*aa**2) - mm*ll**2*c31**2))*(-uu - mm*ll*bb*s31*xx2**2 - (mh*ll + mm*aa + mm*ll)*gg*s3 + (ll/bb)*c31*(uu - mm*ll*bb*s31*xx4**2 - mm*bb*gg*s1)))))

  x3dot = xx3 + dt*xx4

  x4dot = xx4 + dt*((1/(((mh+mm)*ll**2 + mm*aa**2) - mm*ll**2*c31**2))*(-uu - mm*ll*bb*s31*xx2**2 - (mh*ll + mm*aa + mm*ll)*gg*s3 + (ll/bb)*c31*(uu - mm*ll*bb*s31*xx4**2 - mm*bb*gg*s1)))

  return x1dot, x2dot, x3dot, x4dot


# We define the variables as symbols using the SymPy package
xx1 = sp.symbols('xx1')
xx2 = sp.symbols('xx2')
xx3 = sp.symbols('xx3')
xx4 = sp.symbols('xx4')
uu = sp.symbols('uu')  

c31 = sp.cos(xx3-xx1)
s1 = sp.sin(xx1)
s31 = sp.sin(xx3-xx1)
s3 = sp.sin(xx3)

x1dot = xx1 + dt*xx2

x2dot = xx2 + dt*((1/(mm*bb**2))*(uu - mm*ll*bb*s31*xx4**2 - mm*bb*gg*s1 + mm*ll*bb*c31*((1/(((mh+mm)*ll**2 + mm*aa**2) - mm*ll**2*c31**2))*(-uu - mm*ll*bb*s31*xx2**2 - (mh*ll + mm*aa + mm*ll)*gg*s3 + (ll/bb)*c31*(uu - mm*ll*bb*s31*xx4**2 - mm*bb*gg*s1)))))

x3dot = xx3 + dt*xx4

x4dot = xx4 + dt*((1/(((mh+mm)*ll**2 + mm*aa**2) - mm*ll**2*c31**2))*(-uu - mm*ll*bb*s31*xx2**2 - (mh*ll + mm*aa + mm*ll)*gg*s3 + (ll/bb)*c31*(uu - mm*ll*bb*s31*xx4**2 - mm*bb*gg*s1)))

# In order to compute the gradients and the hessians of the function, we used the Lambdify tool that takes the     # symbolic matrices created with SymPy and allow us to substitute numerical values inside 

##################################### Computation of the symbolic Gradients ########################################

xxdot = [x1dot, x2dot, x3dot, x4dot]
xx = [xx1, xx2, xx3, xx4] 

# Gradient of the function with respect to x
fx = []
fx1_lambd = []

for jj in range(ns):
  fx_temp2 = []
  fx_lambd = []
  for ii in range(ns):
    fx_temp = sp.diff(xxdot[jj], xx[ii])
    fx_lambd_temp = sp.lambdify((xx, uu), fx_temp, "numpy")
    fx_temp2.append(fx_temp)
    fx_lambd.append(fx_lambd_temp)

  fx1_lambd.append(fx_lambd)
  fx.append(fx_temp2)


# Gradient of the function with respect to u
fu = []
fu_lambd = []

for ii in range(ns):
  fu_temp = sp.diff(xxdot[ii], uu)
  fu_lambd_temp =sp.lambdify((xx, uu), fu_temp, "numpy")
  fu.append(fu_temp)
  fu_lambd.append(fu_lambd_temp)

##################################### Computation of the symbolic Hessians ########################################

# Hessian of the function with respect to x
hessianx_lambd = []

for kk in range(ns):
  fkk = []
  for jj in range(ns):
    fjj = []
    for ii in range(ns):
      fii = sp.diff(fx[kk][jj], xx[ii])
      hessianx_lambd_temp = sp.lambdify((xx, uu), fii, "numpy")
      fjj.append(hessianx_lambd_temp)    
    fkk.append(fjj)
  hessianx_lambd.append(fkk)

# Hessian of the function with respect to u
hessianu_lambd = []

for ii in range(ns):
  hessianu_temp = sp.diff(fu[ii], uu)
  hessianu_lambd_temp = sp.lambdify((xx, uu), hessianu_temp, "numpy")
  hessianu_lambd.append(hessianu_lambd_temp)

# Hessian of the function with respect to x and u
hessianxu_lambd = []

for jj in range(ns):
  hessianxu_temp2 = []
  for ii in range(ns):
    hessianxu_temp = sp.diff(fx[jj][ii], uu)
    hessianxu_lambd_temp = sp.lambdify((xx, uu), hessianxu_temp, "numpy")
    hessianxu_temp2.append(hessianxu_lambd_temp)
  hessianxu_lambd.append(hessianxu_temp2)      
        
# This function substitutes the numerical values inside the gradients and the hessians  
def dynamics(xxe1, xxe2, xxe3, xxe4, uue):

  # Gradients substitution
  fxx = np.zeros((ns,ns))
  xxe = [xxe1, xxe2, xxe3, xxe4]

  for jj in range(ns):
    for ii in range(ns):
      fx_subs = fx1_lambd[jj][ii](xxe, uue)
      fxx[ii,jj] = fx_subs


  fuu = np.zeros((ni,ns))

  for ii in range(ns):
    fu_subs = fu_lambd[ii](xxe, uue)
    fuu[:,ii] = fu_subs

  # Hessians substitution
  hessianxx = np.zeros((ns,ns,ns))

  for kk in range(ns):
    for jj in range(ns):
      for ii in range(ns):
        hessianx_subs = hessianx_lambd[kk][jj][ii](xxe, uue) 
        hessianxx[kk,jj,ii] = hessianx_subs

  hessianuu = np.zeros((ni, ns))

  for ii in range(ns):
    hessianu_subs = hessianu_lambd[ii](xxe, uue)
    hessianuu[:,ii] = hessianu_subs

  hessianuu = hessianuu.T

  hessianxxuu = np.zeros((ns,ns))

  for jj in range(ns):
    for ii in range(ns):
      hessianxu_subs = hessianxu_lambd[jj][ii](xxe, uue)
      hessianxxuu[ii,jj] = hessianxu_subs

  hessianxxuu = hessianxxuu.T

  return fxx, fuu, hessianxx, hessianuu, hessianxxuu