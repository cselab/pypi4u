#!/usr/bin/env python

# *
# *  CMA.py
# *  PyPi4U
# *
# *  Authors:
# *     Paul Aurel Diederichs  - daurel@ethz.ch
# *     Georgios Arampatzis - arampatzis@collegium.ethz.ch
# *     Panagiotis Chatzidoukas
# *  Copyright 2018 ETH Zurich. All rights reserved.
# *

import sys

sys.path.insert(0, './engines/common')
from read_in import read_in

sys.path.insert(0, './engines/CMA')
from optimize import CMA_method

sys.path.insert(0, './engines/TMCMC')
from sample import tmcmc




model_folder = "model_1/"

#parameters = read_in( model_folder )
#sys.path.insert(0, model_folder )
#CMA_method(*parameters, model_folder)


sys.path.insert(0, model_folder )
tmcmc( model_folder )
