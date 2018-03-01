from model_function import model_function
import numpy as np 
import math


def synthetic_data(time_mesh, theta_0, sigma_0): #creating synthetic data from model with given sigma and given theta_0 and sigma_0
	syntehtic_eval = []
	mu = 0 # mean and standard deviation
	for i in range(len(time_mesh)):
		epsilon = np.random.normal(mu, sigma_0) #generating random variables to function as noise from a normal distribution 
		syntehtic_eval.append(model_function(theta_0,time_mesh[i]) + epsilon) #evaluating model 
	return syntehtic_eval

theta_0 = [4,1,2]
sigma_0 = 1 #defining sigma (adds white noise to my synthetic data)
time_mesh = np.arange(0.01, 4, 0.01)

y = synthetic_data(time_mesh, theta_0, sigma_0) 

np.savetxt("data.txt", np.c_[time_mesh, y])