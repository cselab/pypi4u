'''	*
 	*  CMA_implementation.py
 	*  pypi4u
 	*
 	*  Created by Paul Aurel Diederichs on 01/01/18.
 	*  Copyright 2018 ETH Zurich. All rights reserved.
	*
 	*'''

from read_in import read_in 
from cma_method import CMA_method

parameters = read_in()
CMA_method(*parameters)


