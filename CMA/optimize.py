# *
# *  optimize.py
# *  PyPi4U
# *
# *  Authors:
# *     Paul Aurel Diederichs  - daurel@ethz.ch
# *     Georgios Arampatzis - arampatzis@collegium.ethz.ch
# *     Panagiotis Chatzidoukas
# *  Copyright 2018 ETH Zurich. All rights reserved.
# *

import sys
sys.path.insert(0, '..')
import configparser
import importlib
import math
import numpy as np
import matplotlib
import cma #import cma package
import matplotlib.pyplot as plt


def sigma_func(error_type, sigma_estimator, mean): #defining both the proporitonal and constant error
	if (error_type == 'constant'):
		return sigma_estimator
	elif (error_type == 'proportional'):
		return abs(sigma_estimator * mean)


def log_uniform_prior(uni_lower, uni_upper): #uniform prior
	return math.log(1.0/(uni_upper-uni_lower))

def log_normal_prior(mean, sigma, estimator): #defining the log of the normal prior
	return math.log((1.0/(math.sqrt(2*math.pi)*sigma)))+(-0.5*((((estimator - mean)/sigma))**2)) #log of normal pdf

def log_total_prior(prior_set, estimators, time_instance): #the sum of all log priors
	log_total_prior = 0
	for i in range(len(prior_set)):
		if (prior_set[i][0] == 'uniform'):
			log_total_prior += log_uniform_prior(prior_set[i][1],prior_set[i][2])
		elif (prior_set[i][0] == 'normal'):
			log_total_prior += log_normal_prior(prior_set[i][1], math.sqrt(prior_set[i][2]), estimators[i])
	return log_total_prior


def ln_normal_probability_function(y, mean, sigma, theta): #evaluates my normal probability function for a given y, mean, and theta and sigma
	return math.log((1.0/(math.sqrt(2*math.pi)*sigma)))+(-0.5*((((y - mean)/sigma))**2)) #log of normal pdf

def maximum_likelihood_func_ln(y, time_mesh, estimators, error_type, prior_set, model_filename): #defining my maximum likelihood function, which is a function of my estimators theta and sigma (the parameters I want to determine)
	module = importlib.import_module(model_filename, 'model_function') #importing model function as module
	theta = estimators[0:-1] #model parameter estimators
	sigma_estimator = estimators[-1] #error estimator
	likelihood_output = 0
	for i in range(len(time_mesh)):
		mean = module.model_function(theta, time_mesh[i]) #calculating the mean using the model function for the given estimators
		sigma = sigma_func(error_type, sigma_estimator, mean) #calculating sigma using the definition of the error
		likelihood_output = likelihood_output + ln_normal_probability_function(y[i],mean,sigma,theta) + log_total_prior(prior_set, estimators, time_mesh[i])
	return likelihood_output

def CMA_method(x_0, sigma_0, y_data, t_data, error_type, prior_set, lower_bound, upper_bound, model_filename):
	es = cma.CMAEvolutionStrategy(x_0, sigma_0, {'bounds': [lower_bound, upper_bound]}) #optim instance is generated with starting point x0 = (0)^T and initial standard deviation sigma0 = 1
	while not es.stop(): #iterate
		estiomators = es.ask() #ask delivers new candidate estimatior, estimators is a list or array of candidate estimator points
		es.tell(estiomators, [-1*maximum_likelihood_func_ln(y_data, t_data, estimator, error_type, prior_set,model_filename) for estimator in estiomators]) #tell updates the optim instance by passing the respective function values
		es.logger.add() #append some logging data from CMAEvolutionStrategy class instance es
		es.disp() #displays selected data from the class


	res = es.result
	np.savetxt("CMA_estimators.txt", res[0][:], newline='\n')

	es.result_pretty() #print results
	es.logger.plot() #plots the results


	matplotlib.pyplot.show('hold')
