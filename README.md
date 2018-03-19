# pypi4u

pypi4u is a python based project that provides a TMCMC and covariance matrix adaptation evolution strategy implementation (CMA-ES) to uncertainty quantification and parameter estimation. The CMA-ES implementation uses the covariance matrix adaptation evolution strategy to determine the maximum of the posterior probability distribution, which is defined as following: 




## Getting Started
The covariance matrix adaptation evolution strategy (CMA-ES) implementation requires python 2.7 to be installed on the machine. Additionally, the following python packages need to be installed: 

* cma 2.5.3 https://pypi.python.org/pypi/cma
* numpy
* ConfigParser
* matplotlib
* importlib


## Example - DEMO 
The following demo 


## How it works 

In the following section an example implementation of both methods is shown  

Synthetic data was generated from a predefined model function:

![equation](http://latex.codecogs.com/gif.latex?f%28t%2C%5Ctheta_1%2C%5Ctheta_2%2C%5Cthetat_3%29%3Dt%5Ccdot%5Ctheta_2%5Ccdot%5Ccos%28%5Ctheta_1%5Ccdot%20t%29%20&plus;%20%5Ctheta_1%5Ccdot%5Csin%28t%29) 

The model parameters were set equal to

![equation](http://latex.codecogs.com/gif.latex?%5Ctheta_1%20%3D%204%2C%20%5Ctheta_2%3D1%2C%20%5Ctheta_3%3D2)

and epsilon was equated to

![equation](http://latex.codecogs.com/gif.latex?%5Cvarepsilon%20%5Csim%20%5Cmathcal%7BN%7D%28%5C0%2C1%29)

The function was then evaluated for

![equation](http://latex.codecogs.com/gif.latex?t%20%3D%20%5B0.2%2C%200.4%2C%20%5Chdots%2C%204.0%5D).

The epsilon introduces random noise to the data set that is normally distributed and has a standard devation of one. Consequently, all obtained function evaluations are independently and identically distributed. 







