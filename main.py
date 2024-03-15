import numpy as np
import dynamics as dyn
from sympy import *
from cvxpy import *
import matplotlib.pyplot as plt
import utils as utl
import cost as cost

# first equilibrium
xx1 = 0
xx3 = 0
xx2 = 0
xx4 = 0
uue = 0
dist = 0.21

# second equlibrium
xx3s = 0
xx1s = 40                                       # degrees
xx2s = 0
xx4s = 0
uues = 0
dists = 0.42

tf = 3.5
iterations = 100

ns = dyn.ns                                     # number of states
ni = dyn.ni                                     # number of inputs 
dt = dyn.dt                                     # delta (discretization)
mm = dyn.mm
mh = dyn.mh
gg = dyn.gg
aa = dyn.aa
bb = dyn.bb
ll_dyn = dyn.ll

TTt = int(tf/dt)                                # period

xx = np.zeros((ns, TTt, iterations))
uu = np.zeros((ni, TTt, iterations))
x0 = np.zeros((ns,1))

xxref = np.zeros((ns, TTt))
uuref = np.zeros((ni, TTt))

tt = 0

ll = np.zeros((iterations))
lambdat = np.zeros((ns, TTt, iterations))

# Armijo parametes
gamma_0 = 1                                      # initial stepsize
cc = 0.5
bbeta = 0.7
armijo_iterations = 20                           # number of Armijo iterations

descent = np.zeros((iterations))
term_cond = 1e-7

delta_uu = np.zeros((ni, TTt, iterations))

visu_armijo = False
visu_animation = True

############################################# TASK 1 & TASK 2 ######################################################


for tt in range(TTt):    

    # step reference    
    #xxref[0,int(TTt/2):] = np.ones((1,int(TTt/2)))*np.deg2rad(xx1s)                         
    
    # sigmoid reference
    xxref[0,tt] = utl.reference_position(tt,np.deg2rad(xx1),np.deg2rad(xx1s),TTt)[0]     
    xxref[1,tt] = utl.reference_position(tt,np.deg2rad(xx1),np.deg2rad(xx1s),TTt)[1]        
    xxref[2,tt] = utl.reference_position(tt,np.deg2rad(xx3),np.deg2rad(xx3s),TTt)[0]
    xxref[3,tt] = utl.reference_position(tt,np.deg2rad(xx3),np.deg2rad(xx3s),TTt)[1]     

    # equilibrium input
    uuref[:,tt] = 0.5*(mm*bb*gg*np.sin(xxref[0,tt]) - (mh*ll_dyn + mm*aa + mm*ll_dyn)*gg*np.sin(xxref[2,tt]))              

for i in range(iterations-1):
    # Initialization 
    ll[i] = 0
    
    AAt = np.zeros((ns,ns,TTt))
    BBt = np.zeros((ns,ni,TTt))
    
    hessianx = np.zeros((ns,ns,ns,TTt))
    hessianu = np.zeros((ns,ni,TTt))
    hessianxu = np.zeros((ns,ns,TTt))
    
    lx = np.zeros((ns,TTt))
    lu = np.zeros((ni,TTt))

    QQt = np.zeros((ns,ns,TTt))
    RRt = np.zeros((ni,ni,TTt))
    SSt = np.zeros((ni,ns,TTt)) 

    # Computation of the cost and its gradients
    for tt in range(TTt-1):    
        ll_stage, gradllx, gradllu = cost.cost_function(xx[:,tt,i], uu[:,tt,i], xxref[:,tt], uuref[:,tt])[0:3]
        ll[i] += ll_stage
        lx[:,tt] = gradllx.squeeze()
        lu[:,tt] = gradllu.squeeze()

    ll_term = cost.cost_function(xx[:,TTt-1,i], uu[:,TTt-1,i], xxref[:,TTt-1], uuref[:,TTt-1])[4]
    lTx = cost.cost_function(xx[:,TTt-1,i], uu[:,TTt-1,i], xxref[:,TTt-1], uuref[:,TTt-1])[3]
    
    ll[i] += ll_term 

    # Backward computation of the lambda
    lambdat[:, TTt-1, i] = lTx.squeeze()                # LambdaT

    for tt in reversed(range(TTt-1)):
        fx, fu, hessianx[:,:,:,tt], hessianu[:,:,tt], hessianxu[:,:,tt] = dyn.dynamics(xx[0,tt, i], xx[1,tt,i], xx[2,tt,i], xx[3,tt,i], uu[:,tt,i])
        AAt[:,:,tt] = fx.T
        BBt[:,:,tt] = fu.T

        lambdat_temp = lx[:,tt] + (AAt[:,:,tt].T @ lambdat[:, tt+1, i]) 
        lambdat[:,tt,i] = lambdat_temp.squeeze()

    # Cost matrices definition
    QQT = cost.Regularization()[3]
    
    for tt in range(TTt-1):
       # Here we perform a tensor product between the hessians coming from the dynamics and the lambda vector
       hessian_temp1 = hessianx[0, 0:4, 0:4, tt] * lambdat[0, tt+1, i]
       hessian_temp2 = hessianx[1, 0:4, 0:4, tt] * lambdat[1, tt+1, i]
       hessian_temp3 = hessianx[2, 0:4, 0:4, tt] * lambdat[2, tt+1, i]
       hessian_temp4 = hessianx[3, 0:4, 0:4, tt] * lambdat[3, tt+1, i]

       hessianx_new = hessian_temp1 + hessian_temp2 + hessian_temp3 + hessian_temp4

       hessianxu_temp1 = hessianxu[0, 0:4, tt] * lambdat[0, tt+1, i]
       hessianxu_temp2 = hessianxu[1, 0:4, tt] * lambdat[1, tt+1, i]
       hessianxu_temp3 = hessianxu[2, 0:4, tt] * lambdat[2, tt+1, i]
       hessianxu_temp4 = hessianxu[3, 0:4, tt] * lambdat[3, tt+1, i]

       hessianxu_new = hessianxu_temp1 + hessianxu_temp2 + hessianxu_temp3 + hessianxu_temp4

       hessianu_new = hessianu[:,:,tt].T @ lambdat[:,tt+1,i]

       QQt[:,:,tt] = cost.Regularization()[0] + hessianx_new    
       RRt[:,:,tt] = cost.Regularization()[1] + hessianu_new
       SSt[:,:,tt] = cost.Regularization()[2] + hessianxu_new 

    # affine terms
    qqt = lx
    rrt = lu
    qqT = lTx.squeeze()

    # Delta_u computation through the LQR solver 
    delta_uu[:,:,i] = utl.ltv_LQR(AAt, BBt, QQt, RRt, SSt, QQT, TTt, x0.squeeze(), qqt, rrt, qqT)[3]

    # Computation of the descent direction 
    for tt in (range(TTt)):        
       descent[i] += (lu[:,tt] + BBt[:,:,tt].T @ lambdat[:,tt,i] ).T @ delta_uu[:,tt,i]        

    ###################################### gamma selection - ARMIJO ################################################
    
    stepsizes = []                            
    costs_armijo = []

    ggamma = gamma_0

    for ii in range(armijo_iterations):

        xx_temp = np.zeros((ns, TTt))
        uu_temp = np.zeros((ni, TTt))

        xx_temp[:,0] = x0.squeeze()  
        
        for tt in range(TTt-1):
            uu_temp[:,tt] = uu[:,tt,i] + ggamma*delta_uu[:,tt,i]
            xx_temp[0,tt+1],xx_temp[1,tt+1],xx_temp[2,tt+1],xx_temp[3,tt+1] = dyn.statedot(xx_temp[0,tt], xx_temp[1,tt], xx_temp[2,tt], xx_temp[3,tt], uu_temp[:,tt])
    
        ll_stage_temp = 0

        for tt in range(TTt-1):            
            temp_cost = cost.cost_function(xx_temp[:,tt], uu_temp[:,tt], xxref[:,tt], uuref[:,tt])[0]            
            ll_stage_temp += temp_cost 

        ll_term_temp = cost.cost_function(xx_temp[:,TTt-1], uu_temp[:,TTt-1], xxref[:,TTt-1], uuref[:,TTt-1])[4] 
        ll_armijo = ll_stage_temp + ll_term_temp 

        stepsizes.append(ggamma)            # save the gamma
        costs_armijo.append(ll_armijo)      # save the cost associated to the gamma

        if ll_armijo > ll[i]+cc*ggamma*descent[i]:
           ggamma = bbeta * ggamma         # update the gamma
        else:
            print('Armijo stepsize = {}'.format(ggamma))
            break

    # Armijo plot

    if visu_armijo:

        steps = np.linspace(0,1,int(1e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):

            step = steps[ii]
            
            # temp solution update
            xx_temp = np.zeros((ns,TTt))
            uu_temp = np.zeros((ni,TTt))

            xx_temp[:,0] = x0.squeeze()

            for tt in range(TTt-1):
                uu_temp[:,tt] = uu[:,tt,i] + step*delta_uu[:,tt,i]
                xx_temp[0,tt+1],xx_temp[1,tt+1],xx_temp[2,tt+1],xx_temp[3,tt+1] = dyn.statedot(xx_temp[0,tt], xx_temp[1,tt], xx_temp[2,tt], xx_temp[3,tt], uu_temp[:,tt])
            
            # temp_cost computation
            ll_stage_temp = 0
            temp_cost = 0
            ll_armijo = 0 

            for tt in range(TTt-1): 
                temp_cost = cost.cost_function(xx_temp[:,tt], uu_temp[:,tt], xxref[:,tt], uuref[:,tt])[0]          
                ll_stage_temp = temp_cost + ll_stage_temp  
 
            ll_term_temp = cost.cost_function(xx_temp[:,TTt-1], uu_temp[:,TTt-1], xxref[:,TTt-1], uuref[:,TTt-1])[4] 
           
            ll_armijo = ll_stage_temp + ll_term_temp 

            costs[ii] = ll_armijo


        plt.figure(1)
        plt.clf()

        plt.plot(steps, costs, color='g', label='$l(\\mathbf{u}^k + stepsize*d^k)$')
        plt.plot(steps, ll[i] + descent[i]*steps, color='r', label='$l(\\mathbf{u}^k) + stepsize*\\nabla l(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, ll[i] + cc*descent[i]*steps, color='g', linestyle='dashed', label='$l(\\mathbf{u}^k) + stepsize*c*\\nabla l(\\mathbf{u}^k)^{\\top} d^k$')

        plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()

        plt.show()

    # Implementation of the Newton's Method
    xx_temp = np.zeros((ns,TTt))
    uu_temp = np.zeros((ni,TTt))

    xx_temp[:,0] = x0.squeeze()

    # Dynamics integration 
    for tt in range(TTt-1):
        uu_temp[:,tt] = uu[:,tt,i] + ggamma*delta_uu[:,tt,i]
        xx_temp[0,tt+1],xx_temp[1,tt+1],xx_temp[2,tt+1],xx_temp[3,tt+1] = dyn.statedot(xx_temp[0,tt], xx_temp[1,tt], xx_temp[2,tt], xx_temp[3,tt], uu_temp[:,tt])
    xx[:,:,i+1] = xx_temp
    uu[:,:,i+1] = uu_temp

    # Termination condition

    print('Iteration = {}\t Descent = {}\t Cost = {}'.format(i, descent[i], ll[i]))

    descent_temp = np.abs(descent[i])

    if descent_temp <= term_cond:
        iterations = i
        break

# Definition of the optimal trajectory (for plots)
xx_star = xx[:,:,iterations-1]
uu_star = uu[:,:,iterations-1]
uu_star[:,-1] = uu_star[:,-2] 

################################################## TASK 3 ##########################################################

# Initialization of the tracking variables
xx_track = np.zeros((ns,TTt))
delta_x = np.zeros((ns,TTt))
uu_track = np.zeros((ni,TTt))
xx_opt = xx[:,:,i]
uu_opt = uu[:,:,i]
xx_temp = np.zeros((ns,TTt))

# Change of the initial condition for tracking purposes
x0_track = [np.deg2rad(10), np.deg2rad(10), np.deg2rad(5), np.deg2rad(5)]

# Linearization of the nonlinear system about the optimal trajectory
for tt in range(TTt-1):
    fx, fu = dyn.dynamics(xx_opt[0,tt], xx_opt[1,tt], xx_opt[2,tt], xx_opt[3,tt], uu_opt[:,tt])[0:2]
    AAt[:,:,tt] = fx.T
    BBt[:,:,tt] = fu.T

# Cost matrices definition
QQT = cost.Regularization_tracking()[3]
    
for tt in range(TTt-1):

    QQt[:,:,tt] = cost.Regularization_tracking()[0]     
    RRt[:,:,tt] = cost.Regularization_tracking()[1] 
    SSt[:,:,tt] = cost.Regularization_tracking()[2] 

# Computation of the K matrix
KKt = utl.ltv_LQR(AAt, BBt, QQt, RRt, SSt, QQT, TTt, x0_track)[0]


xx_temp[:,0] = x0_track


for tt in range(TTt-1):
  uu_track[:,tt] = uu_opt[:,tt] + KKt[:,:,tt] @ (xx_temp[:,tt] - xx_opt[:,tt])
  xx_temp[0,tt+1],xx_temp[1,tt+1],xx_temp[2,tt+1],xx_temp[3,tt+1] = dyn.statedot(xx_temp[0,tt], xx_temp[1,tt], xx_temp[2,tt], xx_temp[3,tt], uu_track[:,tt])

xx_track[:,:] = xx_temp


####################################################### PLOTS ######################################################

# Descent direction's plot
plt.figure('descent direction')
plt.plot(np.arange(iterations), np.abs(descent[:iterations]))
plt.xlabel('$k$')
plt.ylabel('||$\\nabla l(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.show()

# Cost's plot
plt.figure('cost')
plt.plot(np.arange(iterations), ll[:iterations])
plt.xlabel('$k$')
plt.ylabel('$l(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.show()

# Optimal trajectory plot
tt_hor = np.linspace(0,tf,TTt)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
axs[0].plot(tt_hor, xxref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
axs[1].plot(tt_hor, xxref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, xx_star[2,:], linewidth=2)
axs[2].plot(tt_hor, xxref[2,:], 'g--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$x_3$')

axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
axs[3].plot(tt_hor, xxref[3,:], 'g--', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$x_4$')

axs[4].plot(tt_hor, uu_star[0,:],'r', linewidth=2)
axs[4].plot(tt_hor, uuref[0,:], 'r--', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$u$')
axs[4].set_xlabel('time')

plt.show()

# Tracking trajectory plot
tt_hor = np.linspace(0,tf,TTt)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_track[0,:], linewidth=2)
axs[0].plot(tt_hor, xx_star[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_track[1,:], linewidth=2)
axs[1].plot(tt_hor, xx_star[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, xx_track[2,:], linewidth=2)
axs[2].plot(tt_hor, xx_star[2,:], 'g--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$x_3$')

axs[3].plot(tt_hor, xx_track[3,:], linewidth=2)
axs[3].plot(tt_hor, xx_star[3,:], 'g--', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$x_4$')

axs[4].plot(tt_hor, uu_track[0,:],'r', linewidth=2)
axs[4].plot(tt_hor, uu_star[0,:], 'r--', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$u$')
axs[4].set_xlabel('time')
  
plt.show()

################################################## ANIMATION ######################################################

import matplotlib.animation as animation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator) 
import matplotlib.patches as ptc

dt = 0.001

time = np.arange(len(tt_hor))*dt

if visu_animation:
    
  fig = plt.figure()  
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1.2, 0.5))
  ax.set_title("Bipedal Robot Animation")
  ax.grid()
  center = 0,0
  radius = 0.05
  Mh = ptc.Circle(center, radius, color = 'green', label = 'mh')
  ax.add_patch(Mh)
      
  ax.set_yticklabels([])
  ax.set_xticklabels([])

  line0, = ax.plot([], [], '-', lw=2, c='k', label='Leg1')
  line1, = ax.plot([], [], '-', lw=2, c='k', label='Leg2')
  point2, = ax.plot([], [], 'o', lw=2, c='g', label='m')
  point3, = ax.plot([], [], 'o', lw=2, c='g', label='m')

  lineref, = ax.plot([], [], '*-', lw=2, c='g',dashes=[2, 2], label='Reference')

  time_template = 't = %.1f s'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
  fig.gca().set_aspect('equal', adjustable='box')

  # Subplot
  left, bottom, width, height = [0.64, 0.65, 0.2, 0.2]
  ax2 = fig.add_axes([left, bottom, width, height])
  ax2.xaxis.set_major_locator(MultipleLocator(2))
  ax2.yaxis.set_major_locator(MultipleLocator(0.25))
  ax2.set_xticklabels([])
  
  ax2.grid(which='both')
  ax2.plot(time, xx_star[0],c='b')
  ax2.plot(time, xxref[0], color='g', dashes=[2, 1])

  point1, = ax2.plot([], [], 'o', lw=2, c='b')
  
  def init():
      line0.set_data([], [])
      line1.set_data([], [])
      point2.set_data([], [])
      point3.set_data([], [])
      lineref.set_data([], [])

      point1.set_data([], [])
      
      time_text.set_text('')
      return line0,line1, lineref, time_text, point1, point2, point3

  def animate(i):
      # Trajectory
      thisx0 = [0, np.sin(xx_star[0, i])]
      thisy0 = [0, -np.cos(xx_star[0, i])]
      line0.set_data(thisx0, thisy0)

      thisx2 = [0, np.sin(xx_star[0, i])/2]
      thisy2 = [0, -np.cos(xx_star[0, i])/2]
      point2.set_data(thisx2, thisy2)

      thisx1 = [0, np.sin(xx_star[2, i])]
      thisy1 = [0, -np.cos(xx_star[2, i])]
      line1.set_data(thisx1, thisy1)

      thisx3 = [0, np.sin(xx_star[2, i])/2]
      thisy3 = [0, -np.cos(xx_star[2, i])/2]
      point3.set_data(thisx3, thisy3)

      # Reference
      thisxref = [0, np.sin(xxref[0, -1])]
      thisyref = [0, -np.cos(xxref[0, -1])]
      lineref.set_data(thisxref, thisyref)

      point1.set_data(i*dt, xx_star[0, i])      

      time_text.set_text(time_template % (i*dt))
      return line0, line1, lineref, time_text, point1, point2, point3

  ani = animation.FuncAnimation(fig, animate, TTt, interval=1, blit=True, init_func=init)
  ax.legend(loc="lower left")

  plt.show()