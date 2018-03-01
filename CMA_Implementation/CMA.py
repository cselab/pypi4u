import ConfigParser 
import importlib
import math
import numpy as np 
import matplotlib
import cma #import cma package
import matplotlib.pyplot as plt

'''
[MODEL]
Number of model parameters = 3
model file = model_function.py
data file = data.txt 

[PRIORS]
# Set prior distribution
# prior distributions uniform normal


P1 = normal 4 2
P2 = normal 1 2
P3 = uniform 0 5
error_prior = uniform 0 5

[log-likelihood]
# error either proportional or constant
error = proportional
'''

config_common_par = ConfigParser.ConfigParser()
config_common_par.read('common_parameters.par')

num_parameters = config_common_par.getint('MODEL', 'Number of model parameters') #reading in the number of parameters that the model function has

model_filename = config_common_par.get('MODEL', 'model file') #reading in the filename of the model function 
model_filename = model_filename.split('.')[0]
module = importlib.import_module(model_filename, 'model_function') #importing model function as module 

data_filename = config_common_par.get('MODEL', 'data file')
data_array = np.loadtxt("%s" %data_filename) 
t_data = data_array[:,0]
y_data = data_array[:,1]


prior_set = []
for i in range(num_parameters):
	try:
		prior = config_common_par.get('PRIORS', 'P%s' %(i+1))
		prior = prior.split(' ')
		if prior[0] == 'uniform':
			uni_lower = float(prior[1])
			uni_upper = float(prior[2])
			prior_set.append([prior[0], uni_lower, uni_upper])
		elif prior[0] == 'normal':
			normal_mu = float(prior[1])
			normal_var = float(prior[2])
			prior_set.append([prior[0], normal_mu, normal_var])
		else:
			print("Unexpected error - Unknown Prior Distribution for prior %s" %(i+1))
			raise()
	except ConfigParser.NoOptionError:
		print("error occured when defining prior %s " %(i+1))
		raise()


try:
	error_prior = config_common_par.get('PRIORS', 'error_prior')
	error_prior = error_prior.split(' ')
	if prior[0] == 'uniform':
		uni_lower = float(error_prior[1])
		uni_upper = float(error_prior[2])
		prior_set.append([error_prior[0], uni_lower, uni_upper])
	elif prior[0] == 'normal':
		normal_mu = float(error_prior[1])
		normal_var = float(error_prior[2])
		prior_set.append([error_prior[0], normal_mu, normal_var])
	else:
		print("Unexpected error - Unknown Error Prior Distribution for prior")
		raise()
except ConfigParser.NoOptionError:
	print("error occured when defining the error prior")
	raise()

error_types = ['proportional', 'constant']
try: 
	error_type = config_common_par.get('log-likelihood', 'error') 
	if(error_type not in error_types):
		print "unknown error type: " + error_type
		raise()
except ConfigParser.NoOptionError:
	print 'error is not defined at all, see section [log-likelihood]'
	raise()



'''
[PARAMETERS]
#defining the parameters for CMA 

bounds = 0 10 #upper and lower bound, the parameters must be within these bounds 
x_0 = 5 5 5 5 #starting point (first two elements are theta) and then error
sigma_0 = 5 #initial standard deviation
'''

config_cma_par = ConfigParser.ConfigParser()
config_cma_par.read('CMA_parameters.par')

bounds = config_cma_par.get('PARAMETERS', 'bounds')
lower_bound = float(bounds.split(' ')[0])
upper_bound = float(bounds.split(' ')[1])

x_loading = config_cma_par.get('PARAMETERS', 'x_0')
x_0 = np.zeros(num_parameters+1)
for i in range(num_parameters+1):
	x_0[i] = float(x_loading.split(' ')[i])

sigma_0 = config_cma_par.get('PARAMETERS', 'sigma_0')
sigma_0 = float(sigma_0.split(' ')[0])



def sigma_func(error_type, sigma_estimator, time_instance): #defining both the proporitonal and constant error 
	if (error_type == 'constant'):
		return sigma_estimator
	elif (error_type == 'proportional'):
		return sigma_estimator * time_instance



def log_uniform_prior(uni_lower, uni_upper):
	return math.log(1.0/(uni_upper-uni_lower))

def log_normal_prior(mean, sigma, estimator):
	return math.log((1.0/(math.sqrt(2*math.pi)*sigma)))+(-0.5*((((estimator - mean)/sigma))**2)) #log of normal pdf 

def log_total_prior(prior_set, estimators, time_instance):
	log_total_prior = 0
	for i in range(len(prior_set)):
		if (prior_set[i][0] == 'uniform'):
			log_total_prior += log_uniform_prior(prior_set[i][1],prior_set[i][2])
		elif (prior_set[i][0] == 'normal'):
			log_total_prior += log_normal_prior(prior_set[i][1], math.sqrt(prior_set[i][2]), estimators[i])
	return log_total_prior



def ln_normal_probability_function(y, mean, sigma, theta): #evaluates my normal probability function for a given y, mean, and theta and sigma
	return math.log((1.0/(math.sqrt(2*math.pi)*sigma)))+(-0.5*((((y - mean)/sigma))**2)) #log of normal pdf 

def maximum_likelihood_func_ln(y, time_mesh, estimators, error_type, prior_set): #defining my maximum likelihood function, which is a function of my estimators theta and sigma (the parameters I want to determine)
	theta = estimators[0:-1]
	sigma_estimator = estimators[-1]
	likelihood_output = 0 
	for i in range(len(time_mesh)):
		mean = module.model_function(theta, time_mesh[i])
		sigma = sigma_func(error_type, sigma_estimator, time_mesh[i])
		likelihood_output = likelihood_output + ln_normal_probability_function(y[i],mean,sigma,theta) + log_total_prior(prior_set, estimators, time_mesh[i])
	return likelihood_output 


es = cma.CMAEvolutionStrategy(x_0, sigma_0, {'bounds': [lower_bound, upper_bound]}) #optim instance is generated with starting point x0 = (0)^T and initial standard deviation sigma0 = 1
while not es.stop(): #iterate
	solutions = es.ask() #ask delivers new candidate solutions, solutions is a list or array of candidate solution points
	es.tell(solutions, [-1*maximum_likelihood_func_ln(y_data, t_data, i, error_type, prior_set) for i in solutions]) #tell updates the optim instance by passing the respective function values
	es.logger.add() #append some logging data from CMAEvolutionStrategy class instance es
	es.disp() #displays selected data from the class 


res = es.result
np.savetxt("estimators.txt", res[0][:], newline='\n')

es.result_pretty() #print results 
es.logger.plot() #plots the results 


matplotlib.pyplot.show('hold')









