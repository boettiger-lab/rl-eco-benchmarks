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


def total_harvest(actions, bound=2, filter_shape = None):
	shape = filter_shape or _default_shape

	return lambda harvested_pops: np.clip(
		np.float32([shape(act) for act in actions]),
		np.zeros(len(actions)),
		np.ones(len(actions)),
	)

