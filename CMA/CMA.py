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
sys.path.insert(0, '../common')
from read_in import read_in
from optimize import CMA_method

parameters = read_in()
CMA_method(*parameters)
