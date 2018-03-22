import configparser
import numpy as np
import re
import sys
try:
    from priors import *
except:
    pass

class Parameters:
    def __init__(self, options):
        self.options = options
        self.set_defaults()

    def set_defaults(self):
        """ Set default values to all member variables """

        # Set default optimization options
        self.options.MaxIter = 1000
        self.options.Tol = 1e-10
        self.options.display = 1
        self.options.Step = 1e-5
        self.prior_type = 0     # uniform = 0 , gaussian = 1

    def read_settings_common(self):
        config_common = configparser.ConfigParser()

        config_common.read("common_parameters.par")

        try:
            self.dimension = int(config_common['MODEL'][
                                            'Number of model parameters'])
            self.model_file = (config_common['MODEL'][
                                            'model file'])
            self.data_file = (config_common['MODEL'][
                                            'data file'])
#            self.alpha = float(config_common['log-likelihood'][
#                                            'alpha'])
#            self.beta = float(config_common['log-likelihood'][
#                                            'beta'])
#            self.gamma = float(config_common['log-likelihood'][
#                                            'gamma'])
            self.error_type = (config_common['log-likelihood']['error'])
            if self.error_type == 'constant':
                self.alpha = 0
                self.beta = 1
            elif self.error_type == 'proportional':
                self.alpha = 1
                self.beta = 0
            # else : raise
            self.gamma = 0
        except:
            print("Error occurred while reading configuration parameters. ")
            raise
        re_expr = re.compile(
                "\s[+-]?(?=\d*)(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?")
        self.priors = np.full(self.dimension+1, None)
        for i in range(self.dimension+1):
                try:
                    if i < self.dimension:
                        line = config_common['PRIORS']['P'+str(i+1)]
                    else:
                        line = config_common['PRIORS']['error_prior']
                except:
                    print("P"+str(i+1) + " or error prior was not found"
                          + " in configuration file.")
                params = re_expr.findall(line)
                if (re.match("uniform", line, re.IGNORECASE)) is not None:
                    self.priors[i] = UniformPrior()
                    self.priors[i].set_bounds(float(params[0]),
                                              float(params[1]))
                elif (re.match("normal", line, re.IGNORECASE)) is not None:
                    self.priors[i] = NormalPrior()
                    self.priors[i].set_distribution(float(params[0]),
                                                    float(params[1]))
                elif (re.match(
                            "lognormal", line, re.IGNORECASE)) is not None:
                    self.priors[i] = LogNormalPrior()
                    self.priors[i].set_distribution(float(params[0]),
                                                    float(params[1]))
                elif (re.match(
                        "truncated_normal", line, re.IGNORECASE)) is not None:
                    self.priors[i] = TruncatedNormalPrior()
                    self.priors[i].set_distribution(float(params[0]),
                                                    float(params[1]),
                                                    float(params[2]),
                                                    float(params[3]))
                else:
                    assert False, ("Prior type for P" + str(i+1) +
                                   " not recognised.")
        self.error_prior = self.priors[self.dimension]
        self.priors = np.array(self.priors)
        self.dimension = self.dimension + 1
        #self.print_data()

    def read_settings_tmcmc(self):
        config_tmcmc = configparser.ConfigParser()
        config_tmcmc.read("tmcmc_parameters.par")

        try:
            self.burn_in = int(config_tmcmc['SIMULATION SETTINGS'][
                                            'burn_in'])
            self.PopSize = int(config_tmcmc['SIMULATION SETTINGS'][
                                            'pop_size'])
            self.tolCOV = float(config_tmcmc['SIMULATION SETTINGS'][
                                            'tol_cov'])
            self.bbeta = float(config_tmcmc['SIMULATION SETTINGS'][
                                            'bbeta'])
            self.MaxStages = int(config_tmcmc['SIMULATION SETTINGS'][
                                            'max_stages'])
            self.seed = int(config_tmcmc['SIMULATION SETTINGS'][
                                            'seed'])
        except:
            print("Error occurred while reading configuration parameters. ")
            raise
        self.Num = np.full(self.MaxStages, self.PopSize)

    def print_data(self):
        print(vars(self))
        return None

def read_CMA(num_parameters):
    config_cma_par = configparser.ConfigParser()
    config_cma_par.read('cma.par')

    bounds = config_cma_par.get('PARAMETERS', 'bounds')
    lower_bound = float(bounds.split(' ')[0])
    upper_bound = float(bounds.split(' ')[1])

    x_loading = config_cma_par.get('PARAMETERS', 'x_0')
    x_0 = np.zeros(num_parameters+1)
    for i in range(num_parameters+1):
        x_0[i] = float(x_loading.split(' ')[i])

    sigma_0 = config_cma_par.get('PARAMETERS', 'sigma_0')
    sigma_0 = float(sigma_0.split(' ')[0])
    return (x_0, sigma_0, lower_bound, upper_bound)

def read_data(data_filename):
    data_array = np.loadtxt("%s" %data_filename)
    t_data = data_array[:,0]
    y_data = data_array[:,1]
    return (t_data, y_data)


def read_in():
    config_common_par = configparser.ConfigParser()
    config_common_par.read('../model.par')

    num_parameters = config_common_par.getint('MODEL', 'Number of model parameters') #reading in the number of parameters that the model function has

    model_filename = config_common_par.get('MODEL', 'model file') #reading in the filename of the model function
    model_filename = model_filename.split('.')[0]

    data_filename = config_common_par.get('MODEL', 'data file')
    (t_data, y_data) = read_data(data_filename)

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
    (x_0, sigma_0, lower_bound, upper_bound) = read_CMA(num_parameters)

    return (x_0, sigma_0, y_data, t_data, error_type, prior_set, lower_bound, upper_bound, model_filename)
