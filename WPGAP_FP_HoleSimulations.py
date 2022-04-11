# from deap import base, creator, tools, algorithms
from dedalus import public as de
from dedalus.extras.plot_tools import plot_bot_2d
from dedalus.extras import *
import numpy as np
import scipy
from scipy import integrate, ndimage
import pickle
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd

import time as timeski
import os

#Suppress most Dedalus output
de.logging_setup.rootlogger.setLevel('ERROR')

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'arial'

rcParams.update({'font.size': 12})

###################################################################
# Simulation Functions
###################################################################

#Scale list between 0 and 1
def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

# steady state ODE
def ssODE(y,t,params):
    u,v,g,G = y
    b,gamma,n,delta,e,c,d = params
    
    du = b*v+v*gamma*u**n/(1+u**n)-delta*u-e*G*u
    dv = -du
    
    dG = c*u*g-d*G
    dg = - dG
    
    derivs = [du,dv,dg,dG]
    return derivs

#determine homogenous SS using ssODE
def homogenousSS(u,v,g,G,params):

    y0 = (u,v,g,G)
    t_sim = (0,1000)
    odeSoln = integrate.odeint(ssODE,y0,t_sim,args=(params,),mxstep=1000000) 
      
    return(odeSoln[1])

def logistic_decay(x,p_min,p_max,k,x0=1.75):  
    #positive k -> decay
    p_amp = p_max - p_min
    return p_min+p_amp/(1+np.exp(k*(x-x0)))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    phi[phi<0]+=2*np.pi

    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(-x, y)

from scipy import interpolate

#given values on a grid of polar coordinates, return values on an evenly spaced, cartesian coordinates grid
def cartesian_coordinate_interpolation(z, r_grid,phi_grid,max_r=4,interp_n = 1000j):
    X,Y = pol2cart(r_grid,phi_grid)

    grid_x, grid_y = np.mgrid[-max_r:max_r:interp_n, -max_r:max_r:interp_n]
    original_grid = np.vstack([X.flatten(),Y.flatten()]).T
    grid_z0 = interpolate.griddata(original_grid, z.flatten(), (grid_x, grid_y), method='nearest')
    
    return(grid_z0)

#given values on a grid of cartesian coordinates, return values on a given polar coordinates grid
def polar_coordinate_interpolation(z,x_grid,y_grid, grid_r,grid_phi):
    RHO,PHI = cart2pol(x_grid,y_grid)

    original_grid = np.vstack([RHO.flatten(),PHI.flatten()]).T
    grid_z0 = interpolate.griddata(original_grid, z.flatten(), (grid_r, grid_phi), method='nearest')
    
    return(grid_z0)

def WPGAP_Disk(params,disk_r = 2, max_r = 4,title='None'):
    c_s,c_p,c_m,gamma_s,gamma_p,gamma_m,e,d = params
    
    b = 0.0002
    delta = 0.04

    #Bases:names,modes,intervals,dealiasing
    phi_basis=de.Fourier('p',256,interval=(0,2*np.pi),dealias=3/2)
    r_basis=de.Chebyshev('r',128,interval=(0,max_r),dealias=3/2)
    #Domain:bases,datatype
    domain=de.Domain([phi_basis,r_basis],float)
    phi, r = domain.grids(scales=1)

    c = domain.new_field(name='c')
    gamma = domain.new_field(name='gamma')

    mu_F = disk_r
    c['g'] = logistic_decay(r[0],c_s,c_p,c_m,x0=mu_F) 
    gamma['g'] = logistic_decay(r[0],gamma_s,gamma_p,gamma_m,x0=mu_F)
        
        
    c.meta['p']['constant'] = True
    gamma.meta['p']['constant'] = True

    T = 4.04
    Tg = 10
    # b = 0.1
    # c = 1
    # delta = 1
    # e = 1
    # gamma = 2
    n = 2
    # d = 1
    Du = .004 #.005 
    Dv = 100*Du
    DG = 100*Du #40
    Dg = 100*Du

    # bp = np.min(b['g']) 
    # cp = np.min(c['g'])

    params = [b,gamma_s,n,delta,e,c_s,d]
#     params = [b_SS,gamma_SS,n,delta,e,c_SS,d]
    u0,v0,g0,G0 = homogenousSS(T/2,T/2, Tg/2, Tg/2,params)


    # Specify problem
    problem = de.IVP(domain, variables=['u', 'v','ur','vr','G','g','Gr','gr'])

    problem.parameters['gamma'] = gamma
    problem.parameters['b'] = b
    problem.parameters['n'] = n
    problem.parameters['u0'] = u0
    problem.parameters['v0'] = v0
    problem.parameters['G0'] = G0
    problem.parameters['g0'] = g0
    problem.parameters['c'] = c
    problem.parameters['dd'] = d
    problem.parameters['delta'] = delta
    problem.parameters['e'] = e
    problem.parameters['Tg'] = Tg

    problem.parameters['Du'] = Du
    problem.parameters['Dv'] = Dv
    problem.parameters['DG'] = DG
    problem.parameters['Dg'] = Dg

    problem.substitutions['f(u,v,G)'] = 'b*v+v*gamma*u**n/(1+u**n)-delta*u-e*G*u'
    problem.substitutions['minf(u,v,G)'] = '-f(u,v,G)'
    problem.substitutions['fg(u,G,g)'] = 'c*u*g-dd*G'
    problem.substitutions['minfg(u,G,g)'] = '-fg(u,G,g)'


    problem.add_equation("r**2*dt(u)-r**2*Du*dr(ur)-r*Du*dr(u)-Du*dp(dp(u))=r**2*f(u,v,G)")
    problem.add_equation("r**2*dt(v)-r**2*Dv*dr(vr)-r*Dv*dr(v)-Dv*dp(dp(v))=r**2*minf(u,v,G)")
    problem.add_equation("r**2*dt(G)-r**2*DG*dr(Gr)-r*DG*dr(G)-DG*dp(dp(G))=r**2*fg(u,G,g)")
    problem.add_equation("r**2*dt(g)-r**2*Dg*dr(gr)-r*Dg*dr(g)-Dg*dp(dp(g))=r**2*minfg(u,G,g)")

    problem.add_equation("ur-dr(u)=0")
    problem.add_equation("vr-dr(v)=0")
    problem.add_equation("Gr-dr(G)=0")
    problem.add_equation("gr-dr(g)=0")

    #Reflective boundary conditions

    problem.add_bc("left (ur) = 0")
    problem.add_bc("right (ur) = 0")
    problem.add_bc("left (vr) = 0")
    problem.add_bc("right (vr) = 0")
    problem.add_bc("left (Gr) = 0")
    problem.add_bc("right (Gr) = 0")
    problem.add_bc("left (gr) = 0")
    problem.add_bc("right (gr) = 0")


    # Pick a timestepper
    ts = de.timesteppers.RK443 #222
    # Build solver
    solver = problem.build_solver(ts)
    # Set integration limits
    solver.stop_wall_time = np.inf
    solver.stop_sim_time = np.inf
    solver.stop_iteration = np.inf
    # Set initial conditions
    u = solver.state ['u']
    v = solver.state['v']
    G = solver.state ['G']
    g = solver.state['g']
    
    #these required slighty more noise than usual?
    urand = 0.3*v0*np.random.rand(*u['g'].shape)

    u['g'] = u0 + urand
    v['g'] = v0 - urand
    G['g'] = G0*np.ones(G['g'].shape)
    g['g'] = g0*np.ones(g['g'].shape)


    phi, r = domain.grids(scales=domain.dealias)
    phi = np.vstack((phi,2*np.pi))
    phi,r = np.meshgrid(phi,r)

    solver.stop_iteration = 10000

    dt =  0.25 #0.1
    nonan = True
    not_steady = True
    prev_state = np.zeros((256*3//2,128*3//2))
    # Main loop chceking stopping criteria
    while solver.ok and nonan and not_steady:
        # Step forward
        solver.step(dt)

        if solver.iteration % 50 == 0:
            if np.count_nonzero(np.isnan(u['g'])) > 0 or np.min(u['g']) < 0 :
                return('Numerical Error')
                nonan = False  
                
        if solver.iteration% 50 ==0:
            curr_state = np.array(u['g'])
            if np.max(np.abs(curr_state-prev_state)) < 10e-4:
                print(np.max(np.abs(curr_state-prev_state)))
                print('Steady state at t = %.2f'%(np.round(solver.iteration*dt,2)))
                not_steady = False
            else: prev_state = np.array(u['g'])

#         if solver.iteration %100  == 0:
#             z = np.vstack((u['g'],u['g'][0])).T
# #             z = np.vstack((u['g']+v['g'],u['g'][0]+v['g'][0])).T
#             fig = plt.figure()
#             # ax = Axes3D(fig)

#             plt.subplot(projection="polar")

#             plt.pcolormesh(phi,r, z,vmin=0., vmax = 1,shading='auto')

#             plt.plot(phi, r, color='k', ls='none') 
#             plt.title('t={}'.format(np.round(solver.iteration*dt,2)))
#             plt.xticks([])
#             plt.yticks([])
#             plt.colorbar()
            
#             index = solver.iteration//100
#             if index < 10:
#                 plt.savefig(save_name+'_00{}.png'.format(index),dpi=300)
#                 plt.close()
#             elif index < 100:
#                 plt.savefig(save_name+'_0{}.png'.format(index),dpi=300)
#                 plt.close()
#             elif index < 1000:
#                 plt.savefig(save_name+'_{}.png'.format(index),dpi=300)
#                 plt.close()

                
    z = np.vstack((u['g'],u['g'][0])).T
    fig = plt.figure(figsize=(4,4)) #figsize = (18,6))
    # ax = Axes3D(fig)

    plt.subplot(projection="polar")

    plt.pcolormesh(phi,r,z,shading='auto',vmin=0.2, vmax = .8)

    plt.plot(phi, r, color='k', ls='none') 
#         plt.legend(['t={}'.format(curr_t)],framealpha=0)
    plt.xticks([])
    plt.yticks([])
    if title != 'None':
        plt.title(title)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('[u]',rotation=0)
    
    # plt.savefig('HoleFigures/HoleRadii%.2f_4.pdf'%(disk_r),bbox_inches='tight',dpi=300)
    # plt.savefig('HoleFigures/HoleRadii%.2f_4.png'%(disk_r),bbox_inches='tight',dpi=300)
    plt.close('all')

    return [u['g'].T,v['g'].T]


# MCMC Parameter Set
c_s = 0.1
gamma_s = 0.0005

c_p,c_m,gamma_p,gamma_m,e,d = [ 1.58414687, 12.89118133,  0.96046689, 2.04555275,  3.13967828, 4.2980938]
        
#hole simulations when decay constants negative (thus, inverted)
param_set = [c_s, c_p,-c_m,gamma_s,gamma_p,-gamma_m,e,d]

phi_basis=de.Fourier('p',256,interval=(0,2*np.pi),dealias=3/2)
r_basis=de.Chebyshev('r',128,interval=(0,4),dealias=3/2)
#Domain:bases,datatyp
domain=de.Domain([phi_basis,r_basis],float)
phi, r = domain.grids(scales=domain.dealias)
phi_grid,r_grid = np.meshgrid(phi,r)

us = []
labels = []
nums = []

import sys
radii = np.float(sys.argv[-1])

for i in range(10):
    u,v = WPGAP_Disk(param_set,disk_r=radii,title='Hole R = %.2f'%radii)
    z0 = cartesian_coordinate_interpolation(u, r_grid, phi_grid)

    curr_mask = z0 > (np.max(u)-np.min(u))/2
    l,n = ndimage.label(curr_mask)

    us.append(u)
    labels.append(l)
    nums.append(n)

pickle.dump(us,open( "hole_us_%.2f.pickle"%radii, "wb" ))
pickle.dump(labels,open( "hole_labels_%.2f.pickle"%radii, "wb" ) )
pickle.dump(np.array([all_radii, nums]),open( "hole_nums_%.2f.pickle"%radii, "wb" ) )





