# import required packages
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import stats, integrate
import pickle
import os
from datetime import datetime
from dedalus import public as de
from dedalus.extras.plot_tools import plot_bot_2d
from dedalus.extras import *
from pymcmcstat import MCMC
from pymcmcstat import MCMC, structures, plotting, propagation
from pymcmcstat.plotting import MCMCPlotting
from pymcmcstat.chain import ChainStatistics
import matplotlib.pyplot as plt
import seaborn as sns

np.seterr(over='ignore');
#Suppress most Dedalus output
de.logging_setup.rootlogger.setLevel('ERROR')

#need to scale by GTPase concentration scaling factor (200)
AVG_SF = pickle.load(open("PickleFiles/Avg_R2_noscale.pickle", "rb"))*200
STD_SF = pickle.load(open("PickleFiles/STDev_R2_noscale.pickle", "rb"))*200

SF_radii = AVG_SF[0]
SF_data = np.vstack([AVG_SF[1], [STD_SF[1] for i in range(8)]]).T


### Functions for simulations
def scale(A):
    #Scale list between 0 and 1
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

def WPGAP(params):
    c_p,c_m,gamma_p,gamma_m,e,d = params
    
    #Fixed parameters
    b = 0.002
    c_s = 1/200
    gamma_s = 0.005
    delta = 0.4
    
    #Bases:names,modes,intervals,dealiasing
    phi_basis=de.Fourier('p',64,interval=(0,2*np.pi),dealias=3/2)
    r_basis=de.Chebyshev('r',64,interval=(0,4),dealias=3/2)
    #Domain:bases,datatype
    domain=de.Domain([phi_basis,r_basis],float)
    phi, r = domain.grids(scales=1)

    mu_F = 1.75

    c = domain.new_field(name='c')
    gamma = domain.new_field(name='gamma')

    c['g'] = logistic_decay(r[0],c_s,c_p,c_m,x0=mu_F) 
    c.meta['p']['constant'] = True

    gamma['g'] = logistic_decay(r[0],gamma_s,gamma_p,gamma_m,x0=mu_F)
    gamma.meta['p']['constant'] = True

    T = 808
    Tg = 10
    n = 2
    K = 200

    Du = .04
    Dv = 100*Du
    DG = 100*Du #40
    Dg = 100*Du

    params = [b,gamma_s,n,delta,e,c_s,d,K]
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
    ts = de.timesteppers.RK222 #443
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
    u_seed = pickle.load( open( "PickleFiles/PR_9Dots.pickle", "rb" ) )
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
                
    return np.array([avg_curr,q1_std, q2_std , q3_std , q4_std,q5_std, q6_std , q7_std , q8_std]).T

#MSE score function to score each parameter set
def WPGAP_mse(params,mc_data):
    def mse(A,B):
        A = A[~np.isnan(A)]
        B = B[~np.isnan(B)]
        return(np.mean(np.subtract(A,B)**2))
    
    sf_avg = np.array(mc_data.ydata).T[0]
    sf_std = np.array(mc_data.ydata).T[1]
    
    sim_dists = WPGAP(params)
    if np.shape(sim_dists) != (96,9) :
        return(np.ones(9)*np.inf)
    
    sim_avg = np.array(sim_dists.T[0])
    sim_std = np.array(sim_dists.T[1:])
    
    mse_avg = [mse(sf_avg,sim_avg)]
    mse_std = [mse(sf_std,sim_std[i]) for i in range(8)]
    

    return np.hstack([mse_avg,mse_std])

# initialize MCMC object
mcstat = MCMC.MCMC()


mcstat.data.add_data_set(x=np.arange(0,96),
                         y=SF_data,
                         user_defined_object=SF_radii)

# add model parameters and set prior value and constraints
mcstat.parameters.add_model_parameter(name='c_p', theta0=11.5/200, minimum=0)
mcstat.parameters.add_model_parameter(name='c_m', theta0=13.29, minimum=1,maximum=50)
mcstat.parameters.add_model_parameter(name='gamma_p', theta0=9.5, minimum=0)
mcstat.parameters.add_model_parameter(name='gamma_m', theta0=1.99, minimum=1,maximum=50)
mcstat.parameters.add_model_parameter(name='e', theta0=32.5, minimum=0)
mcstat.parameters.add_model_parameter(name='d', theta0=43.0, minimum=0)

now = datetime.now()
dt_string = now.strftime("%Y%m%d")
rint = str(int(np.random.rand(1)*10e10))

save_folder = 'MCMCRun015_Logistic_Best_Fixed_Oct_Cont_'+dt_string
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
save_json = 'MCMCRun'+dt_string+'_'+rint+'.json'
save_file = save_folder+'/'+save_json

    
# Generate options
mcstat.simulation_options.define_simulation_options(
    nsimu=5.0e3, updatesigma=True, 
    verbosity=False,save_to_json=True,
    save_lightly=False,savedir = save_folder,results_filename = save_json, waitbar=False )
# Define model object:
mcstat.model_settings.define_model_settings(
    sos_function=WPGAP_mse,
    nbatch = 9,
    sigma2=0.01**2,S20=0.015*np.ones(9),N0=0.015*np.ones(9))


# Run simulation
mcstat.run_simulation()
# # Rerun starting from results of previous run
#mcstat.simulation_options.nsimu = int(2.0e3)
#mcstat.run_simulation(use_previous_results=True)

#results = mcstat.simulation_results.results
#chain = results['chain']
#s2chain = results['s2chain']
#names = results['names']
