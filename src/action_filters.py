import numpy as np

from scipy.special import expit # inverse logit function

# 
# take actions as input, generate function which maps pops to total harvests
# 

def default_filter(act):
	"""linear -1,1 to 0,1"""
	return (act+1)/2

def inverse_logit_filter(act):
	return expit(act)

def quadratic_filter(act):
	return ( (act+1)/2 ) ** 2

def sqrt_filter(act):
	from math import sqrt
	return sqrt( (act+1)/2 )

def esc_filter(act, *, population):
	""" * makes population a keyword arg. """
	#
	esc = (act+1)/2
	#
	if (population == 0) or (population <= esc):
		return 0
	else: #population > esc and population != 0
		return (population - esc) / population # effort, not total harvest!

def absolute_filter(act, *, population):
	""" returns (act+1) / (2 * population), so that act is actually 
	a linear transform of the total harvest: 
	
	i.e. the env collects a harvest of 

		(return val of filter) * population = (act+1)/2.

	I'll impose some arbitrary threshold on the population size below 
	which it returns 0, so that we don't get ridiculously large numbers
	and possibly get noise due to multiplying/diving by large numbers.
	"""

	if population < 10 ** (-4):
		return 0
	else:
		return (act+1) / (2 * population)
