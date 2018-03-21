# Getting started

Both algorithms are implemented in Python 3 and require the installation of additional packages.
To download the current code, please visit the github [repository](https://github.com/cselab/pypi4u) and clone it. 

## Installation
Both implementations rely on mostly well-known packages as, e.g. numpy. To use the code, please install the required packages via "pip3 install *package*". 


**CMA-ES** requires the packages

* cma 2.5.3 - https://pypi.python.org/pypi/cma,
* numpy,
* ConfigParser,
* and matplotlib.



**TMCMC** requires the packages

* numpy,
* scipy, 
* matplotlib,
* and configparser.

## Running an Example

PyPi4u comes with a ready-to-run example including a model, synthetic data and predefined parameters. 

### Model and Synthetic Data

The model is given by the equation 

![equation](http://latex.codecogs.com/gif.latex?f%28t%2C%5Ctheta_1%2C%5Ctheta_2%2C%5Ctheta_3%29%3Dt%5Ccdot%5Ctheta_3%5Ccdot%5Ccos%28%5Ctheta_1%5Ccdot%20t%29%20&plus;%20%5Ctheta_2%5Ccdot%5Csin%28t%29).


The model parameters were set equal to ![equation](http://latex.codecogs.com/gif.latex?%5Ctheta_1%20%3D%204%2C%20%5Ctheta_2%3D1%2C%20%5Ctheta_3%3D2) and the function was evaluated for ![equation](http://latex.codecogs.com/gif.latex?t%20%3D%20%5B0.2%2C%200.4%2C%20%5Chdots%2C%204.0%5D). Random noise is introduced by simply adding epsilon to the function evaluations (constant error). The sum of the terms forms 

![equation](http://latex.codecogs.com/gif.latex?y_i%20%3D%20f%28t_i%2C%5Ctheta_1%2C%5Ctheta_2%2C%5Ctheta_3%29&plus;%5Cvarepsilon)

where epsilon equates to ![equation](http://latex.codecogs.com/gif.latex?%5Cvarepsilon%20%5Csim%20%5Cmathcal%7BN%7D%28%5C0%2C1%29)

Consequently, all obtained function evaluations are independently and identically distributed, following a normal distribution with a variance of one. The synthetic data is stored in a text document `data.txt`, which lists the input value *t* and the corresponding function value *f*. Both approaches use the synthetic data and the function definition *f* to approximate the values of the thetas and epsilon. 

## Play with Parameters

The parameters for the CMA-ES and TMCMC algorithms can be set in the common_parameters.par, CMA.par and the TMCMC.par files. A more detailed description of the parameters and how to set them, can be found in the following. For the example, they are already set in a feasable fashion.

## How to run the Example

Run `CMA_implementation.py` or `sequential_tmcmc.py` to execute the CMA implementation or TMCMC implementation, respectively. On execution of the CMA implementation a text file named `CMA_estimators.txt` is created, in which the values of the estimators are stored. For the TMCMC implementation, "curgen_db_***.dat" files are generated, corresponding to the different generations. The results can be plotted by "plotting.py". 

