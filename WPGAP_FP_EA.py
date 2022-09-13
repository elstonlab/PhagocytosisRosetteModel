#!/usr/bin/env python


from deap import base, creator, tools, algorithms
from dedalus import public as de
import numpy as np
import scipy
from scipy import integrate
import pickle
from math import sqrt

import time as timeski
import os

#Suppress most Dedalus output
de.logging_setup.rootlogger.setLevel('ERROR')

###################################################################
#EA PARAMS
###################################################################

number_of_runs = 1
number_of_generations = 400
number_of_individuals = 50
mutation_rate = 0.3
crossover_rate = 0.5
number_of_params = 6
filename = '012522_Seeded'


###################################################################
# Simulation Functions
###################################################################

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

def WPGAP(params):
	c_p,c_m,gamma_p,gamma_m,e,d = params
	
	b = 0.002
	c_s = 1
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
	DG = 100*Du
	Dg = 100*Du

	# bp = np.min(b['g']) 
	# cp = np.min(c['g'])

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
	problem.parameters['Tg'] = Tg
	problem.parameters['K'] = K

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
	ts = de.timesteppers.RK222
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

	phi, r = domain.grids(scales=domain.dealias)
	phi = np.vstack((phi,2*np.pi))
	phi,r = np.meshgrid(phi,r)

	solver.stop_iteration = 400

	dt = 0.025
	nonan = True
	# curr_t = 0
	# Main loop chceking stopping criteria
	while solver.ok and nonan:
		# Step forward
		solver.step(dt)
		# curr_t += dt

		if solver.iteration % 10 == 0:
			if np.count_nonzero(np.isnan(u['g'])) > 0 or np.min(u['g']) < 0 :
				return('Numerical Error')
				nonan = False        

	return u['g'].T

###################################################################
# EA Functions
###################################################################

def make_conversion_matrix(number_of_params):
	# want easily savable matrix to hold this info
	# interp boolean, interp range (min,max), power boolean, power number (y)
	arr_IandP = np.zeros((5,number_of_params))
	# Set all interp booleans to 1 - everything is going to be interpreted
	arr_IandP[0,:] = 1
	# Set all power booleans to 1 - everything is in the form of powers
	arr_IandP[3,:] = 1
	# Set all power numbers to 10 - everything has a base of 10
	arr_IandP[4,:] = 10
	# Set minimums and maximums for all parameters. 
	# c_p,c_m,gamma_p,gamma_m,e,d
	
	minimums = np.array([-3,0,-3.3,0,-5,-5])

	maximums = np.array([1,1.5,2,1.5,2,2])

	for i in range(len(minimums)):
		arr_IandP[1,i] = minimums[i] #interp_range_min
		arr_IandP[2,i] = maximums[i] #interp_range_max

	return arr_IandP

#converts parameters from an EA individual using the conversion matrix
def convert_individual(ea_individual, conversion_matrix, number_of_params):
	# copy and get len of individual
	arr_params_conv = np.zeros(number_of_params)#np.copy(arr_parameters)
	len_ind = len(ea_individual)

	# Interp:
	for idx in np.nonzero(conversion_matrix[0])[0]:
		ea_val = ea_individual[idx]
		r_min = conversion_matrix[1][idx]
		r_max = conversion_matrix[2][idx]
		arr_params_conv[idx] = np.interp(ea_val, (0,1), (r_min, r_max))

	# Exponentiate:
	for idx in np.nonzero(conversion_matrix[3])[0]:
		ea_val = arr_params_conv[idx]
		base_val = conversion_matrix[4][idx]
		arr_params_conv[idx] = np.power(base_val, ea_val)

	# arr_params_conv[-4:] = np.round(arr_params_conv[-4:],0)

	return arr_params_conv

#Score function to score each parameter set
def ScoreFxn(learned_params):
	
	arr_params_IP = convert_individual(learned_params, arr_conversion_matrix, number_of_params)
	usim = WPGAP(arr_params_IP)
	if type(usim) == str:
		return 10e15

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

	# Training Data
	#need to scale by GTPase concentration scaling factor (200)
	AVG_SF = pickle.load(open("PickleFiles/Avg_R2_noscale.pickle", "rb"))*200
	STD_SF = pickle.load(open("PickleFiles/STDev_R2_noscale.pickle", "rb"))*200
	#error = np.mean(np.mean(np.abs(avg_curr-AVG_SF)) + np.mean(np.abs(std_curr-STD_SF)))
	error = np.mean(np.mean(np.abs(avg_curr-AVG_SF[1]))+
		np.mean(np.abs(q1_std-STD_SF[1]))+np.mean(np.abs(q3_std-STD_SF[1]))+
		np.mean(np.abs(q3_std-STD_SF[1]))+np.mean(np.abs(q4_std-STD_SF[1]))+
		np.mean(np.abs(q5_std-STD_SF[1]))+np.mean(np.abs(q6_std-STD_SF[1]))+
		np.mean(np.abs(q7_std-STD_SF[1]))+np.mean(np.abs(q8_std-STD_SF[1])))
	return error

#helper 
def scorefxn_helper(individual):
	return ScoreFxn(individual),


###################################################################
#Functions To Save Data
###################################################################

def strip_filename(fn):
	#input = full path filename
	#output = filename only
	#EX input = '/home/iammoresentient/phd_lab/data/data_posnegfb_3cellsum.pickled'
	#EX output = 'data_posnegfb_3cellsum'
	if '/' in fn:
		fn = fn.split('/')[-1]
	fn = fn.split('.')[0]
	return fn


def add_info(fn, gens, inds, mutationrate, crossoverrate):
	#input = filename only
	#output = date + filename + EA data
	# EX input = 'data_posnegfb_3cellsum'
	# EX output = '170327_data_posnegfb_3cellsum_#g#i#m#c'

	#get current date:
	cur_date = timeski.strftime('%y%m%d')
	# setup EA data:
	ea_data = str(gens) + 'g' + str(inds) + 'i' + str(int(mutationrate*100)) + 'm' + str(int(crossoverrate*100)) + 'c'
	#put it all together:
	#new_fn = cur_date + '_' + fn + '_' + ea_data
	new_fn = cur_date + '_' + os.path.basename(fn).split('.')[0].split('_')[-1] + '_' + ea_data
	return new_fn

def get_filename(val):
	filename_base = dir_to_use + '/' + stripped_name + '_'
	if val < 10:
		toret = '000' + str(val)
	elif 10 <= val < 100:
		toret = '00' + str(val)
	elif 100 <= val < 1000:
		toret = '0' + str(val)
	else:
		toret = str(val)
	return filename_base + toret + '.pickled'

# Run Code
arr_conversion_matrix = make_conversion_matrix(number_of_params)

stripped_name = strip_filename(filename)
informed_name = add_info(stripped_name, number_of_generations, number_of_individuals, mutation_rate, crossover_rate)
fn_to_use = informed_name
dir_to_use = os.getcwd() + '/' + stripped_name

#check if dir exists and make
if not os.path.isdir(dir_to_use):
	os.makedirs(dir_to_use)
	# print('Made: ' + dir_to_use)
	# and make README file:
	fn = dir_to_use + '/' + 'output.txt'
	file = open(fn, 'w')

	# write pertinent info at top
	file.write('OUTPUT\n\n')
	file.write('Filename: ' + stripped_name + '\n')
	file.write('Directory: ' + dir_to_use + '\n')
	file.write('Data file: ' + filename + '\n\n')
	file.write('Generations: ' + str(number_of_generations) + '\n')
	file.write('Individuals: ' + str(number_of_individuals) + '\n')
	file.write('Mutation rate: ' + str(mutation_rate) + '\n')
	file.write('Crossover rate: ' + str(crossover_rate) + '\n')
	file.write('\n\n\n\n')

	#write script to file
	#script_name = os.getcwd() + '/' + 'EA_1nf1pf.py'
	script_name = os.path.basename(__file__)#__file__)
	open_script = open(script_name, 'r')
	write_script = open_script.read()
	file.write(write_script)
	open_script.close()

	file.close()

###################################################################
#LOOP: EVOLUTIONARY ALGORITHM + SAVE DATA
###################################################################

for i in range(number_of_runs):
	###################################################################
	#EVOLUTIONARY ALGORITHM
	###################################################################

	#TYPE
	#Create minimizing fitness class w/ single objective:
	creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
	#Create individual class:
	creator.create('Individual', list, fitness=creator.FitnessMin)

	#TOOLBOX
	toolbox = base.Toolbox()
	#Register function to create a number in the interval [1-100?]:
	#toolbox.register('init_params', )
	#Register function to use initRepeat to fill individual w/ n calls to rand_num:
	toolbox.register('individual', tools.initRepeat, creator.Individual,
					 np.random.random, n=number_of_params)
	#Register function to use initRepeat to fill population with individuals:
	toolbox.register('population', tools.initRepeat, list, toolbox.individual)

	#GENETIC OPERATORS:
	# Register evaluate fxn = evaluation function, individual to evaluate given later
	toolbox.register('evaluate', scorefxn_helper)
	# Register mate fxn = two points crossover function
	toolbox.register('mate', tools.cxTwoPoint)
	# Register mutate by swapping two points of the individual:
	toolbox.register('mutate', tools.mutPolynomialBounded,
					 eta=0.1, low=0.0, up=1.0, indpb=0.2)
	# Register select = size of tournament set to 3
	toolbox.register('select', tools.selTournament, tournsize=3)

	#EVOLUTION!
	pop = toolbox.population(n=number_of_individuals)
	hof = tools.HallOfFame(1)

	stats = tools.Statistics(key = lambda ind: [ind.fitness.values, ind])
	stats.register('all', np.copy)

	# using built in eaSimple algo
	pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate,
									   mutpb=mutation_rate,
									   ngen=number_of_generations,
									   stats=stats, halloffame=hof,
									   verbose=False)
	# print(f'Run number completed: {i}')

	###################################################################
	#MAKE LISTS
	###################################################################
	# Find best scores and individuals in population
	arr_best_score = []
	arr_best_ind = []
	for a in range(len(logbook)):
		scores = []
		for b in range(len(logbook[a]['all'])):
			scores.append(logbook[a]['all'][b][0][0])
		#print(a, np.nanmin(scores), np.nanargmin(scores))
		arr_best_score.append(np.nanmin(scores))
		#logbook is of type 'deap.creator.Individual' and must be loaded later
		#don't want to have to load it to view data everytime, thus numpy
		ind_np = np.asarray(logbook[a]['all'][np.nanargmin(scores)][1])
		ind_np_conv = convert_individual(ind_np, arr_conversion_matrix, number_of_params)
		arr_best_ind.append(ind_np_conv)
		#arr_best_ind.append(np.asarray(logbook[a]['all'][np.nanargmin(scores)][1]))


	# print('Best individual is:\n %s\nwith fitness: %s' %(arr_best_ind[-1],arr_best_score[-1]))

	###################################################################
	#PICKLE
	###################################################################
	arr_to_pickle = [arr_best_score, arr_best_ind]

	counter = 0
	filename = get_filename(counter)
	while os.path.isfile(filename) == True:
		counter += 1
		filename = get_filename(counter)

	pickle.dump(arr_to_pickle, open(filename,'wb'))


