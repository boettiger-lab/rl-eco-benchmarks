import numpy as np
from dynamical_system import base_params_obj
from base_env import general_env

##############################################################
####################### Example 1 ############################
##############################################################
print("\n\n" + "env 1 test:" + "\n\n")

def _unparametrized_dyn(X: np.float32):
	return np.float32([X * (1 - X)])

def _penalty_fn(t: int):
	return -1 / t

_metadata = {
	#
	# which env class
	'n_sp':  1,
	'n_act': 1,
	'_harvested_sp': [0],
	#
	# about episodes
	'init_pop': np.float32([0.5]),
	'reset_sigma': 0.01,
	'tmax': 1000,
	'penalty_fn': _penalty_fn,
	'extinct_thresh': 0.05,
	#
	# about dynamics / control
	'var_bound': 2,
	'_costs': np.zeros(1, dtype=np.float32),
	'_prices': np.ones(1, dtype=np.float32),
}

env_1 = general_env(
	metadata = _metadata,
	dyn_fn = _unparametrized_dyn, 
	dyn_params = {}, 
	non_stationary = False, 
	non_stationarities = {},
	)

env_1.reset()
for _ in range(10):
	obs, rew, term, _, info = env_1.step(action = [-0.9])
	print(info)


##############################################################
####################### Example 3 ############################
##############################################################
print("\n\n" + "env 3 test:" + "\n\n")

_params = {'r': 2, 'K': 1}

def _parametrized_dyn(X: np.float32, params: dict):
	P = params
	return np.float32([P["r"] * X * (1 - X / P["K"])])

env_3 = general_env(
	metadata = _metadata,
	dyn_fn = _parametrized_dyn, 
	dyn_params = _params, 
	non_stationary = False, 
	non_stationarities = {},
	)

env_3.reset()
for _ in range(10):
	obs, rew, term, _, info = env_3.step(action = [-0.9])
	print(info)


##############################################################
####################### Example 4 ############################
##############################################################
print("\n\n" + "env 4 test:" + "\n\n")

def _r(t):
	return 1 + t/1000


env_4 = general_env(
	metadata = _metadata,
	dyn_fn = _parametrized_dyn, 
	dyn_params = _params, 
	non_stationary = True, 
	non_stationarities = {"r": _r},
	)

env_4.reset()
for _ in range(10):
	obs, rew, term, _, info = env_4.step(action = [-0.9])
	print(info)












