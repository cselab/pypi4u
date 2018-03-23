import configparser
import numpy as np
import re
import sys

sys.path.insert(0, './engines/TMCMC')

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

    def read_settings_common(self, model_folder ):
        config_common = configparser.ConfigParser()

        config_common.read( model_folder + "model.par" )

        try:
            self.dimension = int(config_common['MODEL']['Number of model parameters'])
            self.model_file = (config_common['MODEL']['model file'])
            self.data_file = model_folder + (config_common['MODEL']['data file'])

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

    
    
    
    def read_settings_tmcmc( self, model_folder ):
        
        config_tmcmc = configparser.ConfigParser()
        config_tmcmc.read( model_folder + "tmcmc.par" )

        try:

            tmp = re.split(r'\s*', config_tmcmc['SIMULATION SETTINGS']['burn_in'] )
            self.burn_in = int( tmp[0] )

            tmp = re.split(r'\s*', config_tmcmc['SIMULATION SETTINGS']['pop_size'])
            self.PopSize = int(tmp[0])
            
            tmp = re.split(r'\s*', config_tmcmc['SIMULATION SETTINGS']['tol_cov'])
            self.tolCOV = float(tmp[0])
            
            tmp = re.split(r'\s*', config_tmcmc['SIMULATION SETTINGS']['bbeta'])
            self.bbeta = float(tmp[0])
            
            tmp = re.split(r'\s*', config_tmcmc['SIMULATION SETTINGS']['max_stages'])
            self.MaxStages = int(tmp[0])
            
            tmp = re.split(r'\s*', config_tmcmc['SIMULATION SETTINGS']['seed'])
            self.seed = int(tmp[0])

        except:
            print("Error occurred while reading configuration parameters. ")
            raise
        self.Num = np.full(self.MaxStages, self.PopSize)

    def print_data(self):
        print(vars(self))
        return None

def read_CMA( num_parameters, model_folder ):
    
    x_0 = np.zeros(num_parameters+1)

    config_cma_par = configparser.ConfigParser()
    config_cma_par.read( model_folder + 'cma.par')
    
    bounds = config_cma_par.get('PARAMETERS', 'bounds')
    tmp = re.split(r'\s*', bounds );
    lower_bound = float( tmp[0] )
    upper_bound = float( tmp[1] )

    x_loading = config_cma_par.get('PARAMETERS', 'x_0')
    tmp = re.split(r'\s*',x_loading);
    for i in range(num_parameters+1):
        x_0[i] = float( tmp[i] )

    sigma_0 = config_cma_par.get('PARAMETERS', 'sigma_0')
    tmp = re.split(r'\s*', sigma_0 );
    sigma_0 = float( tmp[0] )
    return (x_0, sigma_0, lower_bound, upper_bound)




def read_data(data_filename):
    data_array = np.loadtxt("%s" %data_filename)
    t_data = data_array[:,0]
    y_data = data_array[:,1]
    return (t_data, y_data)


def read_in( model_folder ):
    
    config_common_par = configparser.ConfigParser()
    config_common_par.read( model_folder + 'model.par')

    tmp = config_common_par.get('MODEL', 'Number of model parameters') 
    tmp = re.split(r'\s*', tmp );
    num_parameters = int(tmp[0])

    tmp = config_common_par.get('MODEL', 'model file') 
    tmp = re.split(r'\s*', tmp );
    model_filename = tmp[0]

    tmp = model_folder + config_common_par.get('MODEL', 'data file')
    data_filename = re.split(r'\s*', tmp )[0];
    (t_data, y_data) = read_data(data_filename)

    prior_set = []
    for i in range(num_parameters):
        try:
            tmp = config_common_par.get('PRIORS', 'P%s' %(i+1))
            prior = re.split(r'\s*', tmp );
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
        tmp = config_common_par.get('PRIORS', 'error_prior')
        error_prior = re.split(r'\s*', tmp );
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
        tmp = config_common_par.get('log-likelihood', 'error')
        error_type = re.split(r'\s*', tmp )[0];
        if(error_type not in error_types):
            print ("unknown error type: " + error_type)
            raise()
    except configparser.NoOptionError:
        print ('error is not defined at all, see section [log-likelihood]')
        raise()
    (x_0, sigma_0, lower_bound, upper_bound) = read_CMA( num_parameters, model_folder )

    return (x_0, sigma_0, y_data, t_data, error_type, prior_set, lower_bound, upper_bound, model_filename)
