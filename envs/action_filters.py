import numpy as np

from scipy.special import expit # inverse logit function

# 
# take actions as input, generate function which maps pops to total harvests
# 

def _default_shape(act):
	"""linear -1,1 to 0,1"""
	return (act+1)/2

def inverse_logit_filter(actions):
	efforts = np.float32([expit(act) for act in actions])
	return lambda harvested_pops: np.multiply(efforts, harvested_pops)

def escapement_filter(actions, filter_shape = None):

	def base_esc(esc, X):
		""" single species """
		if X<esc:
			return 0
		else:
			return X-esc

	shape = filter_shape or _default_shape

	escapements = np.float32([shape(act) for act in actions])

	return lambda harvested_pops: np.float32([base_esc(escapements[i], pop) for i, pop in enumerate(harvested_pops)])

def total_harvest(actions, bound=2, filter_shape = None):
	shape = filter_shape or _default_shape

	return lambda harvested_pops: np.clip(
		np.float32([shape(act) for act in actions]),
		np.zeros(len(actions)),
		np.ones(len(actions)),
	)

