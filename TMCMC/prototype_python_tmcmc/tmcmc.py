import argparse, sys
import numpy as np
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
 
'''
Define a standard normal gaussian with mean 0 std 1
sample it with TMCMC - estimate the posterior mean and stds.
'''
 
'''
Likelihood function - fitness  -> Here normal of zero mean var std
'''
def lnprob(x, var):
    return -0.5*x.size*(np.log(2.0*np.pi)+np.log(var)) - 0.5*np.sum( x**2 / var**2)
 
'''
Prior distribution
'''
def lnprior(x, privar):
    if  ( (any(x) < -privar) or (any(x)> privar) ) : return -np.inf
    return np.log(1/(2.0*privar))
 
'''
MCMC transition kernel: here simple diffusion :
Can be also langevin when derivatives are available, or can be Hamiltonian etc etc, or can be stochastic newton if we also have hessians
'''
def propose(x, cov_ss):
    return np.random.multivariate_normal(x,cov_ss)
 
'''
Function to calculate cov given sample likelihoods and annhealing stage.
'''
def Objlogp(runinfo, x, tol):
    fj = runinfo[0] #likelihood values
    pj = runinfo[2]
    fjmax = np.max(fj)
    q = np.exp( (fj-fjmax)*(x-pj) )
    q = q/np.sum(q)
    return (np.std(q)/np.mean(q) - tol)**2
 
'''
Do nesessary postprocessing in the likelihood evaluations of 1 stage in order to be able to get the new weights,
calculate the evidence, and also calculate the annealing schedule dynamically
'''
def chain_statistics(runinfo, sys_para):
    # runinfo = { lnlik, lnpri, pj, x, curgen }
    #   sys_para = (ndim, bbeta2, nwalkers, tolcov, nsteps, iplot, print_chain)
 
    #first : find the next exponent to be in the right tolcov
    fit=[Objlogp(runinfo,ptest,sys_para[3]) for ptest in np.logspace(np.log10(runinfo[2]),np.log10(1.5),10000)]
    idx = np.argmin(fit)
    ptest = np.logspace(np.log10(runinfo[2]),np.log10(1.5),10000)
    pnew= ptest[idx]
    if pnew>1.0: pnew = 1.0
    # need to do the sum-log trick in order to avoid overflows
    tmp = np.max( runinfo[0]* (pnew- runinfo[2]) )
    w = np.exp( runinfo[0]*( (pnew- runinfo[2]) ) - tmp )
    # calculate partial evidence
    #w = w*sys_para[2]
    lnEv = np.log(np.sum(w))+tmp-np.log(sys_para[2])
    #runinfo.w = runinfo.w .* runinfo.Ns;
    #% calculate partial ln(evidence)
    #runinfo.S_lnEv(runinfo.gen) = log(sum(runinfo.w))+tmp-log(sys_para.N_s);
 
    # normalize the weights
    w = w/w.sum()
    # calculate covariance matrix for MCMC proposals in next stage
    # remove weighted mean
    #repmat(a, m, n) is tile(a, (m, n)).
    cov_ss = runinfo[3] - np.tile(np.dot(w.T,runinfo[3]),(runinfo[3].shape[0],1) )
    # weighted sample cov
    cov_ss = np.dot(cov_ss.T, (cov_ss*np.tile(np.array([w]).T,(1,sys_para[0] ) ) ) )
    # guarantee symmetry of cov. matrix & scale with beta^2
    cov_ss = (sys_para[1]*0.5)*( cov_ss + cov_ss.T)
    # final covariance matrix should be in shape of #ndim x ndim
    return w, cov_ss, pnew, lnEv
 
 
 
 
'''
Dump some stuff - here the ending nwalker samples of the algorithm
'''
def output_diagnostics (runinfo, sys_para, lnev):
     # runinfo = { lnlik, lnpri, pj, x, curgen}
    #   sys_para = (ndim, bbeta2, nwalkers, tolcov, nsteps, iplot, print_chain)
  iplot=sys_para[5]
  print_chain = sys_para[6]
  if (iplot==1 and runinfo[2]==1.0):
    x_f = runinfo[3]
    df = pd.DataFrame(x_f)
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.savefig("final_samples" + ".png")
  if runinfo[2] >= 1.0:
    x_f = runinfo[3]
    print 'The mean of the final samples is :', np.mean(x_f,axis=0)
    print 'The std of the final samples is :' , np.std(x_f,axis=0)
    print 'The model log(Evidence) is :',lnev.sum(), ', Theoretical value:', -x_f.shape[1]*np.log(10)
    print_chain = 1
    if print_chain == 1:
      f = open("final"+"_samples.dat", "w")
      f.close()
    if print_chain == 1:
      f = open("final"+"_samples.dat", "a")
      for k in range(x_f.shape[0]):
            f.write("{0:6d} {1:s}\n".format(k, " ".join(map(str,x_f[k]))+" "+str(runinfo[0][k])))
      f.close()
 
'''
Given the resampled filtered points, do nsteps of MCMC to rejuvenate the particle filter seeds
'''
def move_mcmc_seeds (p0, sys_para, cov_ss, runinfo ):
    # runinfo = { lnlik, lnpri, pj, x, curgen }
    #   sys_para = (ndim, bbeta2, nwalkers, tolcov, nsteps, iplot, print_chain)
    # candidate points
    # for all seeds
    var =1.0
    prior_range=5.0
    nsteps= sys_para[4]
    x_new = runinfo[3] # copy original sampled points before mcmc steps
    lnlik_final= runinfo[0] # same, place holder for the final likelihood values
    lnpri_final= runinfo[1] # same, place holder for the final prior values
# here should be the despot distributed evaluation - the first for loop should be task distributed
    for i in xrange(p0.shape[0]):
        for k in xrange(nsteps):  #this inner loop is sequential mcmc steps - should reside in local worker
          if k==0:
            x_old = p0[i,:]
            lnlik_old= runinfo[0][i]  #likelihood from current evaluation
            lnpri_old= runinfo[1][i]  # prior from current evaluation
          #propose a new candidate point using x_old as mean and cov_ss as proposal matrix
          x_cand = propose(x_old, cov_ss)
          # evaluate the prior and likelihood at the new point 
          lnlik_cand=lnprob(x_cand , var)
          lnpri_cand=lnprior(x_cand , prior_range)
          #evaluate the acceptance ratio
          r = np.exp(runinfo[2]*(lnlik_cand - lnlik_old) + (lnpri_cand - lnpri_old) )
          #Metropolize!!!
          r = np.min([1,r])
          state = np.random.multinomial(1,np.array([r,1-r]) )
          if state[0]==0: #in this case we failed - the point is rejected
            x_old = x_old
            lnlik_old = lnlik_old
            lnpri_old = lnpri_old
          else : #if we actually succeeded in our proposed point!
            x_old = x_cand
            lnlik_old = lnlik_cand
            lnpri_old = lnpri_cand
        #copy the final values of the burnin steps only
        lnlik_final[i] = lnlik_old
        lnpri_final[i] = lnpri_old
        x_new[i,:] = np.array(x_old)
    return lnlik_final, lnpri_final, x_new
       
 
 
 
if __name__ == "__main__":
 
  parser = argparse.ArgumentParser(description='Transitional Markov Chain Monte Carlo posterior sampling algorithm - improved sampling with 0 bias version.')
  parser.add_argument('--dimensions', type=int, default=2, help='count of dimensions.')
  parser.add_argument('--beta', type=float, default=0.08, help='sigma for the target Normal distribution .')
  parser.add_argument('--population', type=int, default=100, help='population size of points per iteration.')
  parser.add_argument('--seed', type=int, default=12345, help='random seed.')
  parser.add_argument('--tolcov', type=float, default=1.0, help='tolerance for the coefficient of variation of the weights for annealing')
  parser.add_argument('--nmcmcsteps', type=int, default= 1, help='number of burnin steps used in the MCMC steps within the rejuvenation BASIS steps');
  parser.add_argument('--iplot', type=int, default=0, help='Plot a pandas plot matrix of the posterior?.')
  parser.add_argument('--print_chain', type=int, default=0, help='print the final samples to a file.')
 
  args = parser.parse_args()
 
  #number of dims , number of parallel walkers to use : #walkers need to be at least 2x the dim
  ndim, nwalkers = args.dimensions, args.population
  bbeta2 = args.beta
  tolcov = args.tolcov
  nsteps = args.nmcmcsteps
  var = 1.0
  #init seed
  np.random.seed(args.seed)
  # do not produce an ascii file with all thesamples
  print_chain = 0
  # initial positioning of the tmcmc seeds
  prior_range = 10.0
  p0 = np.array([prior_range* (np.random.rand(ndim)-0.5) for i in xrange(nwalkers)])
  # initial annealing exponent pj - temperature schedule
  pj=1e-8
  #current generation marker - we start at 0
  curgen = 0;
  lnlik = np.zeros((nwalkers,1))
  lnpri = np.zeros((nwalkers,1))
  lnlik=lnlik[:,-1]
  lnpri=lnpri[:,-1]    
  lnev = np.zeros((100,1))
 
  sys_para = (ndim, bbeta2, nwalkers, tolcov, nsteps, args.iplot, args.print_chain)
 
## MAIN LOOP OF THE algorithm
  while pj<1 :
    # This is a very cheap computation, can stay on master worker
    # This is the prior sampling - only in generation 0
    if curgen==0:
      for i in xrange(p0.shape[0]):
        lnpri[i] = lnprior(p0[i,:] , prior_range/2.0)
      # here should be the despot distributed evaluation
      for i in xrange(p0.shape[0]):
        lnlik[i] = lnprob(p0[i,:] , var)
    # the following stuff now is statistics in the main process
 
    # tuple that contains all the important stuff needed for the statistics computation
    x = p0
    if curgen==0:  runinfo = ( lnlik, lnpri, pj, x, curgen) #need to define it for the first prior loop , does not exist so far!
    w,cov_ss,pnew, lnev[curgen] = chain_statistics(runinfo, sys_para) #get the weights and the covariance matrix for the mcmc steps
    #print 'Pjold',pj,'Pjnew',pnew,'lnev',lnev[curgen],'Ev tot',lnev.sum()
    pj = pnew
    # resamble the current chains
    len_chain=np.random.multinomial(sys_para[2],w)
    # new walker positions are now only the guys that did not get filtered out
    #new_seeds = x[len_chain>0,:]
    filt_seeds = np.zeros([1,ndim])
    filt_lik = np.zeros([1,1])
    filt_pri = np.zeros([1,1])
    for i in range(x.shape[0]):
      filt_seeds = np.vstack([filt_seeds , np.tile(x[i,:],(len_chain[i],1))])
      filt_lik = np.vstack([filt_lik , np.tile(lnlik[i],(len_chain[i],1))])
      filt_pri = np.vstack([filt_pri , np.tile(lnpri[i],(len_chain[i],1))])
 
    p0 = filt_seeds[1:,]
    lnlik = filt_lik[1:]
    lnpri = filt_pri[1:]
    lnlik = lnlik[:,-1]
    lnpri = lnpri[:,-1]
    # update using the new filtered values
    runinfo = ( lnlik, lnpri, pj, p0, curgen)
    # now , start an m-step mcmc chain from each one of the p0 points (many of them are at this point the same)
    lnlik,lnpri,p0 = move_mcmc_seeds(p0, sys_para, cov_ss, runinfo )
    curgen=curgen+1
    print 'Finished generation',curgen,' and annealing exponent is :', pj
  output_diagnostics( runinfo, sys_para, lnev)
