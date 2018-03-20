import configparser 
import numpy as np 


def read_in():
	config_common_par = configparser.ConfigParser()
	config_common_par.read('common_parameters.par')

	num_parameters = config_common_par.getint('MODEL', 'Number of model parameters') #reading in the number of parameters that the model function has

	model_filename = config_common_par.get('MODEL', 'model file') #reading in the filename of the model function 
	model_filename = model_filename.split('.')[0]

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
		except configparser.NoOptionError:
			print("error occured when defining prior %s " %(i+1))
			raise()

	try:
		error_prior = config_common_par.get('PRIORS', 'error_prior')
		error_prior = error_prior.split(' ')
		if error_prior[0] == 'uniform':
			uni_lower = float(error_prior[1])
			uni_upper = float(error_prior[2])
			prior_set.append([error_prior[0], uni_lower, uni_upper])
		elif error_prior[0] == 'normal':
			normal_mu = float(error_prior[1])
			normal_var = float(error_prior[2])
			prior_set.append([error_prior[0], normal_mu, normal_var])
		else:
			print("Unexpected error - Unknown Error Prior Distribution for prior")
			raise()
	except configparser.NoOptionError:
		print("error occured when defining the error prior")
		raise()

	error_types = ['proportional', 'constant']
	try: 
		error_type = config_common_par.get('log-likelihood', 'error') 
		if(error_type not in error_types):
			print ("unknown error type: " + error_type)
			raise()
	except configparser.NoOptionError:
		print ('error is not defined at all, see section [log-likelihood]')
		raise()


	config_cma_par = configparser.ConfigParser()
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


	return (x_0, sigma_0, y_data, t_data, error_type, prior_set, lower_bound, upper_bound, model_filename)
	
