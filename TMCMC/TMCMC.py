# *
# *  sequential_tmcmc.py
# *  PyPi4U
# *
# *  Authors:
# *     Philipp Mueller  - muellphi@ethz.ch
# *     Georgios Arampatzis - arampatzis@collegium.ethz.ch
# *     Panagiotis Chatzidoukas
# *  Copyright 2018 ETH Zurich. All rights reserved.
# *
import argparse
import sys
sys.path.insert(0, '../common')
sys.path.insert(0, '../')
import numpy as np
from scipy import optimize, random, stats
import matplotlib.pyplot as plt
from read_in import Parameters
import argparse
from math import exp, log
import configparser
from importlib import import_module
sys.path.insert(0, 'TMCMC')
from priors import *
from plotting import plot_theta
from random_auxiliary import *


class LogLikelihood:
    """ """
    def __init__(self, model_function, data_file, parameters):
        self.alpha = parameters.alpha
        self.beta = parameters.beta
        self.gamma = parameters.gamma
        self.parameters = parameters

        # Load model function
        try:
            self.model = import_module(model_function)
            self.m_func = self.model.model_function
        except:
            print("Model function could not been loaded.")
            raise

        # Load data
        try:
            self.data = np.loadtxt(data_file)
        except:
            print("Error occurred during reading data file.")
            raise

    def __call__(self, model_params):
        sum = 0
        self.sigma = model_params[self.parameters.dimension-1]
        volatility = (self.beta * self.sigma)**2

        for t in range(len(self.data)):
            f_ti_v = self.m_func(model_params, self.data[t, 0])

            # Recalculate volatility if a proportional error is assumed
            if self.gamma != 0 and self.alpha != 0:
                volatility = ((self.alpha * abs(f_ti_v) ** self.gamma +
                               self.beta) * self.sigma)**2
            elif self.alpha != 0:
                volatility = ((self.alpha + self.beta) * self.sigma)**2
            sum += - (self.data[t, 1] - f_ti_v)**2 / (2*volatility)
            sum += - 0.5 * log(2*np.pi*volatility)

        return sum


class Sort:
    def __init__(self, idx, sel, F):
        self.idx = idx
        self.nsel = sel
        self.F = F


class GenerationDB:
    def __init__(self):
        self.point = None
        self.entry = None

        self.F = 0.0
        self.nsel = -1
        self.queue = -1
        self.entries = 0

    def init(self, parameters):
        self.entry = np.empty(parameters.PopSize + 1, dtype=object)

    def update(self, point, F, parameters):
        if self.entry is None:
            self.init(parameters)

        pos = self.entries
        self.entries += 1
        self.entry[pos] = GenerationDB()

        if (self.entry[pos].point is None):
            self.entry[pos].point = point.copy()
            self.entry[pos].F = F

    def print_size(self):
        print("=======")
        print("CURGEN_DB [size= " + str(self.entries) + " ]")
        print("=======")


class OptimOptions:
    def __init__(self):
        self.MaxIter = None
        self.Tol = 1e-10
        self.display = 1
        self.Step = 1e-6


class RunInfo:
    def __init__(self):
        return

    def init_runinfo(self, parameters):
        self.CoefVar = np.zeros(parameters.MaxStages + 1, dtype=np.float)
        self.p = np.zeros(parameters.MaxStages + 1, dtype=np.float)
        self.currentuniques = np.zeros(parameters.MaxStages, dtype=np.float)
        self.logselection = np.zeros(parameters.MaxStages, dtype=np.float)
        self.acceptance = np.zeros(parameters.MaxStages, dtype=np.float)
        self.SS = np.zeros((parameters.dimension, parameters.dimension),
                           dtype=np.float)
        self.meantheta = np.zeros((parameters.MaxStages, parameters.dimension),
                                  dtype=np.float)
        self.Gen = 0
        self.CoefVar[0] = 10

    def save_runinfo(self):
        return None

    def load_runinfo(self):
        return None




def init_chaintask(in_tparam, parameters, curgen_db, loglikelihood):
    """ Evaluate function values F(c) = Posterior(c) """
    point = in_tparam.copy()
    fpoint = loglikelihood(point)
    curgen_db.update(point, fpoint, parameters)


def prepare_newgen(nchains, leaders, curgen_db, parameters, runinfo):
    """ DOCUMENTATION """

    n = curgen_db.entries
    fj = np.empty(n, dtype=np.float)
    sel = np.zeros(n, dtype=np.int)

    for i in range(n):
        fj[i] = curgen_db.entry[i].F
    calculate_statistics(fj, parameters=parameters, runinfo=runinfo,
                         curgen_db=curgen_db, sel=sel)
    newchains = 0
    for i in range(n):
        if sel[i] != 0:
            newchains += 1

    ldi = 0       # leader index
    for i in range(n):
        # Check if selected by normalized plausability weights
        if sel[i] != 0:
            idx = i
            for p in range(parameters.dimension):
                leaders[ldi].point[p] = curgen_db.entry[idx].point[p]
            leaders[ldi].F = curgen_db.entry[idx].F
            leaders[ldi].nsel = sel[i]
            ldi += 1

    curgen_db.entries = 0

    print("calculate statistics: newchains = " + str(newchains))

    return newchains




def calculate_statistics(flc, parameters, runinfo, curgen_db, sel):
    """ Calculate annealing constang p_{j+1} s.t. COV of
        {f(D|M,theta)}^{p_{j+1} - p_j} is within tolCOV. """
    display = parameters.options.display
    tolCOV = parameters.tolCOV
    CoefVar = runinfo.CoefVar
    p = runinfo.p
    Num = parameters.Num
    logselection = runinfo.logselection
    Step = parameters.options.Step
    tol = parameters.options.Tol
    Gen = runinfo.Gen
    maxIter = parameters.options.MaxIter
    n = curgen_db.entries
    fmin, xmin, conv = 0, 0, 0

# Estimate p_{j+1} such that COV of objlog is lower than a prescribed threshold
    if conv == 0:
        method = 'Nelder-Mead'
        options = {
            'disp': False,
            'maxiter': maxIter,
            'xatol': tol,
            'return_all': True,
            'fatol': tol}
        res = optimize.minimize(
            obj_log_p, p[Gen], method=method, args=(
                flc, p[Gen], tolCOV), options=options)
        xmin = res.x
        fmin = res.fun
        conv = res.success
        #print(
        #    "fminsearch: conv = " +
        #    str(conv) +
        #    " xmin = " +
        #    str(xmin) +
        #    " fmin = " +
        #    str(fmin))

    j = Gen + 1

    if (conv != 0 and (xmin > p[Gen])):
        p[j] = xmin
        CoefVar[j] = fmin
    else:
        p[j] = p[Gen] + 0.1 * Step
        CoefVar[j] = CoefVar[Gen]

    if (p[j] > 1):
        p[j] = 1
        Num[j] = parameters.PopSize

    flcp = np.empty(n, dtype=np.float)
    for i in range(n):
        flcp[i] = flc[i] * (p[j] - p[j - 1])

    fjmax = np.max(flcp)
    weight = np.zeros(n, dtype=np.float)
    for i in range(n):
        weight[i] = np.exp(flcp[i] - fjmax)

    sum_weight = np.sum(weight)

    # calculate normalized weights and save to q
    q = np.empty(n, dtype=np.float)
    for i in range(n):
        q[i] = weight[i] / sum_weight

    # if (display):
    #     print("runinfo_q - normalized weights" + str(q))

    runinfo.logselection[Gen] = np.log(sum_weight) + fjmax - np.log(n)
    #if (display):
    #    print("logselection \n" + str(logselection[0:Gen+1]))
    #    print("\n")
    #    print("\n")
    CoefVar[Gen] = np.std(q) / np.mean(q)

    #if(display):
    #    print("CoefVar  \n" + str(CoefVar[0:Gen+1]))
    #    print("\n")
    #    print("\n")

    N = 1

    samples = n

    nselections = samples

    for i in range(samples):
        sel[i] = 0


    # Draw nselections from K with probabilites q = normalized weights
    # selected samples are distributed as f_{j+1}
    nn = multinomialrand(nselections, q)
    for i in range(n):
        sel[i] += nn[i]

    #if (display):
    #    print("SEL = " + str(sel))

    mean_of_theta = np.zeros(parameters.dimension, dtype=np.float)
    for i in range(parameters.dimension):
        for j in range(n):
            mean_of_theta[i] += curgen_db.entry[j].point[i] * q[j]

        runinfo.meantheta[Gen][i] = mean_of_theta[i]
    meanv = np.empty(parameters.dimension)
    for i in range(parameters.dimension):
        meanv[i] = mean_of_theta[i]

    for i in range(parameters.dimension):
        for j in range(parameters.dimension):
            s = 0
            for k in range(n):
                s += q[k] * (curgen_db.entry[k].point[i] - meanv[i]) * \
                    (curgen_db.entry[k].point[j] - meanv[j])
            runinfo.SS[i][j] = s
            runinfo.SS[j][i] = s

    #if (display):
    #    print("runinfo.SS = \n" + str(runinfo.SS))


def obj_log_p(x, fj, pj, tol):
    """Function to calculate cov given sample likelihoods and annealing
        stage."""
    fjmax = np.max(fj)
    q = np.exp((fj - fjmax) * (x - pj))
    q = q / np.sum(q)
    CoefVar = (np.std(q) / np.mean(q) - tol) ** 2  # result
    #print(
    #    "   pj = %.16f" % pj +
    #    "   x = %.16f" % x +
    #    "   f(x) = %.16f" % CoefVar,
    #    "   tol = " + str(tol))
    return CoefVar


def chaintask(in_tparam, pnsteps, out_tparam, winfo, runinfo, parameters,
              curgen_db, loglikelihood):
    """Initialize Markov Chain"""
    nsteps = pnsteps
    gen_id = winfo[0]
    chain_id = winfo[1]
    burn_in = parameters.burn_in

    leader = np.zeros(parameters.dimension, dtype=np.float)

    for i in range(parameters.dimension):
        leader[i] = in_tparam[i]  # get leader
    loglik_leader = out_tparam[0]  # and their function value
    pj = runinfo.p[runinfo.Gen]

    for step in range(nsteps+burn_in):
        # Compute candidate by drawing from normal distribution
        # centered at leader with covariance of S
        candidate = propose_candidate(leader, parameters, runinfo)
        loglik_candidate = loglikelihood(candidate)

        logprior_candidate = logpriorpdf(candidate, n=parameters.dimension,
                                         parameters=parameters)
        logprior_leader = logpriorpdf(leader, n=parameters.dimension,
                                      parameters=parameters)
        # without exp, with log in logpriorpdf and fitfun
        L = (logprior_candidate - logprior_leader) + (loglik_candidate - loglik_leader) * pj
        
        #print( logprior_candidate, logprior_leader, loglik_candidate, loglik_leader )
        #print(L)
        #print("---------")

        if( L < -100 ):
            L = 0
        elif( L > 0):
            L = 1
        else:
            L = exp( L )


        if (uniformrand(0, 1) < L):  # Accept candidate with probability L
            leader = candidate
            loglik_leader = loglik_candidate
            if step >= burn_in:     # Discard first burn_in runs
                curgen_db.update(leader, loglik_candidate, parameters)
        else:   # Discard candidate and add current leader with probability 1-L
            if step >= burn_in:
                curgen_db.update(leader, loglik_leader, parameters)
    return


def dump_curgen_db(Gen, parameters, curgen_db):
    """Print theta and lik to curgen_db_GEN.txt file. This file can be used
        for plotting."""
    with open("curgen_db_" + "{0:0=3d}".format(Gen) + ".txt", "w") as f:
        for pos in range(curgen_db.entries):
            for i in range(parameters.dimension):
                f.write(str(curgen_db.entry[pos].point[i]) + " ")
            f.write(str(curgen_db.entry[pos].F) + "\n")


def logpriorpdf(theta, n, parameters):
    res = 0
    for i in range(n):
        res += parameters.priors[i].logpriorpdf(theta[i])
    return res

#@profile
def propose_candidate(leader, parameters, runinfo):
    """ Sample a candidate from the multivariate_normal, centered
        at the chain's leader with variance bSS. """
    bSS = np.empty((parameters.dimension, parameters.dimension), dtype=np.float)
    for i in range(parameters.dimension):
        for j in range(parameters.dimension):
            bSS[i][j] = parameters.bbeta * runinfo.SS[i][j]

    # Generate random numbers until RN inside parameters range
    #while(True):
    #    flag = 0
    candidate = np.random.multivariate_normal(leader, bSS)
    #    for i in range(parameters.dimension):
    #        assert (np.isnan(candidate[i]) is not False), "Nan in candidate point! - Something went wrong!!!!" + str(candidate[i])           
    #        if ((candidate[i] < parameters.priors[i].lower_bound) or (candidate[i] > parameters.priors[i].upper_bound)):
    #            flag = 1
    #            break
    #    if (flag == 0):
    #        break
    
    return candidate


def tmcmc():

    options = OptimOptions()
    parameters = Parameters(options)
    parameters.read_settings_common()
    parameters.read_settings_tmcmc()
    curgen_db = GenerationDB()
    runinfo = RunInfo()
    runinfo.init_runinfo(parameters)



    loglikelihood = LogLikelihood(parameters.model_file, parameters.data_file,
                                  parameters)
    # Set random seed
    if parameters.seed != -1:
        np.random.seed(parameters.seed)

    out_tparam = np.zeros(parameters.PopSize, dtype=np.float)
    winfo = np.zeros(4, dtype=np.int)
    in_tparam = np.zeros(parameters.dimension, dtype=np.float)

    nchains = parameters.Num[0]
    curgen_db.entries = 0

    # Randomly select nchains starting points c from prior pdf,
    # calculate function value F(c) from posterior distribution, and
    # put results in curgen_db
    for i in range(int(nchains)):
        winfo[0] = runinfo.Gen
        winfo[1] = i
        for d in range(parameters.dimension):
            in_tparam[d] = parameters.priors[d].sample()
        init_chaintask(in_tparam, parameters, curgen_db, loglikelihood)
    #curgen_db.print_size()

    # dump curgen database for plotting
    dump_curgen_db(runinfo.Gen, parameters, curgen_db)

    leaders = np.empty(parameters.PopSize, dtype=object)
    for i in range(parameters.PopSize):
        leaders[i] = GenerationDB()
        leaders[i].point = np.empty(parameters.dimension, dtype=np.float)

    nchains = prepare_newgen(nchains=nchains, leaders=leaders,
                             curgen_db=curgen_db, parameters=parameters,
                             runinfo=runinfo)
    runinfo.Gen += 1


    while runinfo.Gen < parameters.MaxStages:

        print("======================================")

        print("Generation = " + str(runinfo.Gen) + "\np = " + str(runinfo.p[runinfo.Gen]))

        for i in range(nchains):
            winfo[0] = runinfo.Gen
            winfo[1] = i
            in_tparam = np.zeros(parameters.dimension)
            for p in range(parameters.dimension):
                in_tparam[p] = leaders[i].point[p]
            nsteps = leaders[i].nsel
            out_tparam[0] = leaders[i].F
            chaintask(in_tparam=in_tparam, pnsteps=nsteps,
                      out_tparam=out_tparam,
                      winfo=winfo, runinfo=runinfo, parameters=parameters,
                      curgen_db=curgen_db, loglikelihood=loglikelihood)
        #curgen_db.print_size()
        dump_curgen_db(runinfo.Gen, parameters, curgen_db)
        nchains = prepare_newgen(nchains, leaders, curgen_db, parameters=parameters,
                                 runinfo=runinfo)
        if runinfo.p[runinfo.Gen] == 1:
            break

        runinfo.Gen += 1

    #plot_theta("curgen_db_" + "{0:0=3d}".format(runinfo.Gen-1) + ".txt", False)


if __name__ == '__main__':
    tmcmc()
