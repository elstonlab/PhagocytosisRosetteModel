import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import stats, integrate
import pickle
from dedalus import public as de
from dedalus.extras.plot_tools import plot_bot_2d
from dedalus.extras import *
import matplotlib.pyplot as plt
import seaborn as sns

np.seterr(over='ignore');
#Suppress most Dedalus output
de.logging_setup.rootlogger.setLevel('ERROR')

import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
matplotlib.rcParams['font.sans-serif']="Arial"
matplotlib.rcParams['font.family']="sans-serif"
# rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams.update({'font.size': 16})

#need to scale by GTPase concentration scaling factor (200)
AVG_SF = pickle.load(open("PickleFiles/Avg_R2_noscale_128.pickle", "rb"))*200
STD_SF = pickle.load(open("PickleFiles/STDev_R2_noscale_128.pickle", "rb"))*200

#Scale list between 0 and 1
def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

# steady state ODE
def ssODE(y,t,params):
    u,v,g,G = y
    b,gamma,n,delta,e,c,d,K = params
    
    du = b*v+v*gamma*u**n/(K**n+u**n)-delta*u-e*G*u
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

def WPGAP(params,plot_bool=False,title='None',mu_F = 1.75):
    c_s,c_p,c_m,gamma_s,gamma_p,gamma_m,e,d = params
    
    # Fixed parameters
    b = 0.002
#     c_s = 1/200
#     gamma_s = 0.005
    delta = 0.4
    
    #Bases:names,modes,intervals,dealiasing
    phi_basis=de.Fourier('p',256,interval=(0,2*np.pi),dealias=3/2)
    r_basis=de.Chebyshev('r',128,interval=(0,4),dealias=3/2)
    #Domain:bases,datatype
    domain=de.Domain([phi_basis,r_basis],float)
    phi, r = domain.grids(scales=1)

    c = domain.new_field(name='c')
    gamma = domain.new_field(name='gamma')

    
    c['g'] = logistic_decay(r[0],c_s,c_p,c_m,x0=mu_F) 
    gamma['g'] = logistic_decay(r[0],gamma_s,gamma_p,gamma_m,x0=mu_F)

        
    c.meta['p']['constant'] = True
    gamma.meta['p']['constant'] = True

    T = 808
    Tg = 10
    n = 2
    K = 200

    Du = .04 
    Dv = 100*Du
    DG = 100*Du #40
    Dg = 100*Du

    params = [b,np.min(gamma['g']),n,delta,e,np.min(c['g']),d,K]
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
    problem.parameters['K'] = K
    problem.parameters['Tg'] = Tg

    problem.parameters['Du'] = Du
    problem.parameters['Dv'] = Dv
    problem.parameters['DG'] = DG
    problem.parameters['Dg'] = Dg

    problem.substitutions['f(u,v,G)'] = 'b*v+v*gamma*u**n/(K**n+u**n)-delta*u-e*G*u'
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
    ts = de.timesteppers.RK443 #443
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
    # u_seed = pickle.load( open( "PR_9Dots.pickle", "rb" ) )
    u_seed = pickle.load( open( "PickleFiles/PR_256x128.pickle", "rb" ) )[2]
    u_seed_norm = u_seed/np.max(u_seed)
    urand = 0.1*v0*np.random.rand(*u['g'].shape) + 0.1*v0*u_seed_norm

    u['g'] = u0+urand
    v['g'] = v0-urand
    G['g'] = G0*np.ones(G['g'].shape)
    g['g'] = g0*np.ones(g['g'].shape)

    solver.stop_iteration = 400

    dt =  0.025 
    nonan = True
    # Main loop chceking stopping criteria
    while solver.ok and nonan:
        # Step forward
        solver.step(dt)

        if solver.iteration % 50 == 0:
            if np.count_nonzero(np.isnan(u['g'])) > 0 or np.min(u['g']) < 0 :
                return('Numerical Error')
                nonan = False  
                
    usim = u['g'].T
            
#     phi, r = domain.grids(scales=domain.dealias)
#     phi = np.vstack((phi,2*np.pi))
#     phi,r = np.meshgrid(phi,r)
#     z = np.vstack((u['g'],u['g'][0])).T
#     fig = plt.figure(figsize=(4,4)) #figsize = (18,6))
#     # ax = Axes3D(fig)

#     plt.subplot(projection="polar")

#     plt.pcolormesh(phi,r,z,shading='auto',vmin=0.2*200, vmax = .8*200)

#     plt.plot(phi, r, color='k', ls='none') 
# #         plt.legend(['t={}'.format(curr_t)],framealpha=0)
#     plt.xticks([])
#     plt.yticks([])
#     if title != 'None':
#         plt.title(title)
#     cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
#     cbar.ax.get_yaxis().labelpad = 15
#     cbar.ax.set_ylabel('[u]',rotation=0)

    # plt.savefig('../Figures/Sweep_%s.png'%title,bbox_inches='tight',dpi=300)
    
    avg_curr = []
    q1_std, q2_std , q3_std , q4_std,q5_std, q6_std , q7_std , q8_std = [],[],[],[],[],[],[],[]
    inc = len(usim[0])//8
    for i in range(len(usim)):
        avg_curr.append(np.mean(usim[i]))
        q1_std.append(np.std(usim[i][0:inc]))
        q2_std.append(np.std(usim[i][inc:2*inc]))
        q3_std.append(np.std(usim[i][2*inc:3*inc]))
        q4_std.append(np.std(usim[i][3*inc:4*inc]))
        q5_std.append(np.std(usim[i][4*inc:5*inc]))
        q6_std.append(np.std(usim[i][5*inc:6*inc]))
        q7_std.append(np.std(usim[i][6*inc:7*inc]))
        q8_std.append(np.std(usim[i][7*inc:8*inc]))
    q1_std = np.array(q1_std)
    q2_std = np.array(q2_std)
    q3_std = np.array(q3_std)
    q4_std = np.array(q4_std)
    q5_std = np.array(q5_std)
    q6_std = np.array(q6_std)
    q7_std = np.array(q7_std)
    q8_std = np.array(q8_std)
    
    avg_curr = np.array(avg_curr)
    
    def mse(A,B):
        A = A[~np.isnan(A)]
        B = B[~np.isnan(B)]
        return(np.mean(np.subtract(A,B)**2))
        
    def mae(A,B):
        A = A[~np.isnan(A)]
        B = B[~np.isnan(B)]
        return(np.mean(np.abs(np.subtract(A,B))))
    
    try:
        total_error = np.sum([mae(AVG_SF[1],avg_curr),
                            mae(STD_SF[1],q1_std),mae(STD_SF[1],q2_std),mae(STD_SF[1],q3_std),mae(STD_SF[1],q4_std),
                            mae(STD_SF[1],q5_std),mae(STD_SF[1],q6_std),mae(STD_SF[1],q7_std),mae(STD_SF[1],q8_std)])
    except: 
        print('Error')
        return(avg_curr, q1_std)

                
    return [np.array(u['g'].T), total_error]

c_s = 1/200
gamma_s = 0.0005

c_p,c_m,gamma_p,gamma_m,e,d = [ 15.8414687/200, 12.89118133,  9.6046689, 2.04555275,  31.3967828, 42.980938]

param_set = [c_s, c_p,c_m,gamma_s,gamma_p,gamma_m,e,d]

## Sweep individual parameters

param_name = 'c_max'
param_i = 1
vals = np.linspace(8/200,26/200,11)

# param_name = 'c_km' 
# param_i = 2
# vals = np.linspace(10,18,15)

# param_name = 'gam_max'
# param_i = 4
# vals = np.linspace(8.5,11,11)

#param_name = 'gam_km'
#param_i = 5
#vals = np.linspace(1.7,2.3,11)

#param_name = 'e'
#param_i = 6
#vals = np.linspace(24,36,11)

#param_name = 'd'
#param_i = 7
#vals = np.linspace(36,50,11)

errors = []
for val in vals:
    curr_param_set = np.copy(param_set)
    curr_param_set[param_i] = val

    try:
        u,error =WPGAP(curr_param_set,plot_bool=True,title='%s%.2f'%(param_name,val))
        errors.append(error)
    except: 
        print('Error %s = %.2f'%(param_name,val))
        errors.append(100)

# plt.figure(figsize=(4,3))
# plt.scatter(vals,errors)
# plt.ylabel('Score')
# plt.xlabel(param_name)
# plt.ylim([0,60])
# plt.savefig('../Figures/Sweep_Errors%s.png'%(param_name),bbox_inches='tight',dpi=300)

# pickle.dump(np.array([vals,errors]),open('../Figures/%s_errors.pickle'%param_name,'wb'))

