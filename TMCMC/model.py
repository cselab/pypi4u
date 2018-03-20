import math


def model_function(theta, time):
   """Evaluates the model function for a given theta and time."""
   return time*theta[2]*math.cos(theta[0]*time) + theta[1]*math.sin(time)
