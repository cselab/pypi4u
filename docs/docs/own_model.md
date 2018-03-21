# Set up own model

## Model Function

The example model function is defined as following: 

![equation](http://latex.codecogs.com/gif.latex?f%28t%2C%5Ctheta_1%2C%5Ctheta_2%2C%5Ctheta_3%29%3Dt%5Ccdot%5Ctheta_3%5Ccdot%5Ccos%28%5Ctheta_1%5Ccdot%20t%29%20&plus;%20%5Ctheta_2%5Ccdot%5Csin%28t%29) 

Therefore, the first argument of the function, the theta vector, needs to be a vector of size three, as there are three model parameters. The resulting function definition is as following: 

```
import math

def model_function(theta, time): #evaluates my model function for a given theta and time
	return time*theta[2]*math.cos(theta[0]*time) + theta[1]*math.sin(time)
```

Both the CMA-ES and the TMCMC implementation call this python function.  

## Parameters

Both CMA-ES and TMCMC enable the user to tweak the algorithms by changing some parameters. We split the parameters in a common_parameters.par, CMA_parameters.par and TMCMC.par file, with the common, the CMA specific and TMCMC specific parameters, respectively. 

### Common Parameters
```
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
error_prior = uniform 0 2

[log-likelihood]
# error either proportional or constant
error = constant
```

**[MODEL]** - The model function comprises three parameters; therefore the number of model parameters was set to three. Additionally, the relative paths to the python model function and to the data file are given. 

**[PRIORS]** - In priors you can encode your prior beliefs about the parameters. Follow the following formats:

* normal distributed parameter i: Pi = normal mean variance
* uniformly distributed parameter i: Pi = normal lower_bound upper_bound

In this exemplary case, the prior for the first parameter was taken to be a normal probability distribution with a mean of 4 and a variance of 2. The prior of the second parameter is also a normal probability distribution, but with a mean of 1 and a variance of 2. The third prior was set to a uniform probability distribution with a minimum of 0 and maximum of 5. Finally, the error prior was defined to be a uniform distribution with a minimum of 0 and maximum of 2. 

**[log-likelihood]** - The synthetic data was produced by corrupting the function evaluations with constant noise, which originated from a normal distribution with a mean of 0 and a variance of 1 (![equation](http://latex.codecogs.com/gif.latex?%5Cvarepsilon%20%5Csim%20%5Cmathcal%7BN%7D%28%5C0%2C1%29)). Therefore, the error is set equal to a constant in the log-likelihood section of the common parameters. 

### CMA-ES Implementation
To be able to implement the CMA-ES algorithm the CMA parameters must still be defined.  

```
[PARAMETERS]
#defining the parameters for CMA 

bounds = 0 10 #upper and lower bound, the parameters must be within these bounds 
x_0 = 5 5 5 5 #starting point, initial guess for the theta vector (the last entry of the vector corresponds to the guess of the error term)
sigma_0 = 5 #initial standard deviation
```

In this example all parameters lie within the bound [0,10]. Furthermore, the rule of thumb is applied to obtain an initial starting guess for the theta vector. Finally, the initial standard deviation of the CMA-ES alogrithm was defined to be 5. 
