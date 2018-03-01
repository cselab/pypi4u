from random_auxiliary import uniformrand
from math import log
from scipy import optimize, random, stats


class UniformPrior():
    """ Class for dimensions with uniform prior. """
    def set_bounds(self, lower_bound, upper_bound):
        """ Set bounds """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self):
        """ Sample uniformly from domain [lower_bound, upper_bound]  """
        return uniformrand(self.lower_bound, self.upper_bound)

    def logpriorpdf(self, x=None):
        return -log(self.upper_bound-self.lower_bound)


class NormalPrior():
    """ Class for dimensions with normal prior. """
    def set_distribution(self, mu, sigma):
        self.mu, self.sigma = mu, sigma
        self.lower_bound = self.mu - 10 * self.sigma
        self.upper_bound = self.mu + 10 * self.sigma

    def sample(self):
        """ Sample with mean mu and variance sigma  """
        return random.normal(self.mu, self.sigma, 1)

    def logpriorpdf(self, x):
        return stats.lognorm.pdf(x, s=self.sigma, scale=exp(self.mu))


class TruncatedNormalPrior():
    def set_distribution(self, mu, sigma, lower_bound, upper_bound):
        self.mu = mu
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.sample_gen = stats.truncnorm(
            (self.lower_bound - self.mu) / self.sigma, (
             self.upper_bound - self.mu) / self.sigma, loc=self.mu,
             scale=self.sigma)

    def sample(self):
        """sample from truncated normal - very slow currently"""
        return self.sample_gen.rvs(1)

    def logpriorpdf(self, x):
        return log(self.sample_gen.pdf(x))


class LogNormalPrior(NormalPrior):

    def sample(self):
        return random.lognormal(self.mu, self.sigma, 1)
