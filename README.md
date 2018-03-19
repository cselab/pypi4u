# pypi4u

pypi4u is a python based project that provides a TMCMC and covariance matrix adaptation evolution strategy implementation (CMA-ES) to uncertainty quantification and parameter estimation. The CMA-ES implementation uses the covariance matrix adaptation evolution strategy to determine the maximum of the posterior probability distribution, which is defined as following: 

![equation](http://latex.codecogs.com/gif.latex?p%28hypothesis%7Cdata%2CI%29%20%5Cpropto%20p%28data%7Chypothesis%2CI%29%5Ctimes%20p%28hypothesis%7CI%29)

The TMCMC implementation directly generates samples from the probability function using a markov chain. The generated samples can then be used to determine the stochastic mean and variance. The stochasitc mean of the multivariate distribution can be equated to the most-likely parameters/estimators that define the trend of the data. 


## Getting Started
The covariance matrix adaptation evolution strategy (CMA-ES) implementation requires python 2.7. Furthermore, the following python packages need to be installed: 

* cma 2.5.3 - https://pypi.python.org/pypi/cma
* numpy
* ConfigParser
* matplotlib
* importlib

## Example - DEMO 
The following demo 


## How it Works
The following section explains the structure of the project and how the provided code can be used to make estimations of model parameters. This is then further illustrated by the proceeding example. 

## Common Parameters
Both implementations access a common paramter file, named `common_parameters.par`. The common parameter file, which needs to be filled out by the user, 

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

### Generation of Synthetic Data 
Synthetic data was generated from a predefined model function:

![equation](http://latex.codecogs.com/gif.latex?f%28t%2C%5Ctheta_1%2C%5Ctheta_2%2C%5Cthetat_3%29%3Dt%5Ccdot%5Ctheta_2%5Ccdot%5Ccos%28%5Ctheta_1%5Ccdot%20t%29%20&plus;%20%5Ctheta_1%5Ccdot%5Csin%28t%29) 

The model parameters were set equal to

![equation](http://latex.codecogs.com/gif.latex?%5Ctheta_1%20%3D%204%2C%20%5Ctheta_2%3D1%2C%20%5Ctheta_3%3D2)

The function was then evaluated for

![equation](http://latex.codecogs.com/gif.latex?t%20%3D%20%5B0.2%2C%200.4%2C%20%5Chdots%2C%204.0%5D)

Additionally, random noise is introduced by simply adding epsilon to the function evaluations.T The sum of the terms forms 

![equation](http://latex.codecogs.com/gif.latex?y_i%20%3D%20f%28t_i%2C%5Ctheta_1%2C%5Ctheta_2%2C%5Ctheta_3%29&plus;%5Cvarepsilon)

where epsilon equates to 

![equation](http://latex.codecogs.com/gif.latex?%5Cvarepsilon%20%5Csim%20%5Cmathcal%7BN%7D%28%5C0%2C1%29)

 Consequently, all obtained function evaluations are independently and identically distributed, following a normal distribution with a standard deviation of one. The synthetic data is stored in a text document `data.txt`, which lists the input value *t* and the corresponding function value *f*. Both approaches use the synthetic data and the function definiton f to approximate the values of the thetas and epsilon. 











