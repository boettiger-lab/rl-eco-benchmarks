#
# factory for standard choices of env based on # species and actions
#

import numpy as np

from base_env import eco_env
from dyn_fns import (
	twoSp_1, twoSp_2, threeSp_1, threeSp_2, fourSp_1
)

def _default_penalty_fn(t: int):
	return - 1000 / (t+1)

#####################################################################
########################### 2 SP 1 ##################################
#####################################################################

def twoSp_1_factory(n_act: int = 2, non_stationarities: dict = {}):
	metadata = {
		#
		# which env class
		'n_sp':  2,
		'n_act': n_act,
		'_harvested_sp': [ i for i in range(n_act) ],
		#
		# about episodes
		'init_pop': np.float32([0.5, 0.5]),
		'reset_sigma': 0.01,
		'tmax': 1000,
		'penalty_fn': _default_penalty_fn,
		'extinct_thresh': 0.05,
		#
		# about dynamics / control
		'var_bound': 2,
		'_costs': np.zeros(n_act, dtype=np.float32),
		'_prices': np.ones(n_act, dtype=np.float32),
	}
	params = {
		'K_x': np.random.choice([0.9, 1, 1.1]),
		'K_y': np.random.choice([0.9, 1, 1.1]),
		'LV_xy': np.random.choice([0.05, 0.1, 0.15]),
		'r_x': np.random.choice([0.9, 1, 1.1]),
		'r_y': np.random.choice([0.9, 1, 1.1]),
		#
		'sigma_x': 0.1,
		'sigma_y': 0.1,
	}
	info = {'metadata': metadata, 'params': params}

	env = eco_env(
		metadata = metadata,
		dyn_fn = twoSp_1, 
		dyn_params = params, 
		non_stationary = (len(non_stationarities) > 0), 
		non_stationarities = non_stationarities,
		)

	return env, info

#####################################################################
########################### 2 SP 2 ##################################
#####################################################################

def twoSp_2_factory(n_act: int = 2, non_stationarities: dict = {}):
	metadata = {
		#
		# which env class
		'n_sp':  2,
		'n_act': n_act,
		'_harvested_sp': [ i for i in range(n_act) ],
		#
		# about episodes
		'init_pop': np.float32([0.5, 0.5]),
		'reset_sigma': 0.01,
		'tmax': 1000,
		'penalty_fn': _default_penalty_fn,
		'extinct_thresh': 0.05,
		#
		# about dynamics / control
		'var_bound': 2,
		'_costs': np.zeros(n_act, dtype=np.float32),
		'_prices': np.ones(n_act, dtype=np.float32),
	}
	params = {
		'c': np.random.choice([0.2, 0.25, 0.3]),
		'd_y': np.random.choice([0.2, 0.3, 0.4]),
		'K_x': np.random.choice([0.9, 1, 1.1]),
		'r_x': np.random.choice([0.9, 1, 1.1]),
		'r_y': np.random.choice([0.9, 1, 1.1]),
		#
		'sigma_x': 0.1,
		'sigma_y': 0.1,
	}
	info = {'metadata': metadata, 'params': params}

	env = eco_env(
		metadata = metadata,
		dyn_fn = twoSp_2, 
		dyn_params = params, 
		non_stationary = (len(non_stationarities) > 0), 
		non_stationarities = non_stationarities,
		)

	return env, info

#####################################################################
########################### 3 SP 1 ##################################
#####################################################################

def threeSp_1_factory(n_act: int = 2, non_stationarities: dict = {}):
	metadata = {
		#
		# which env class
		'n_sp':  3,
		'n_act': n_act,
		'_harvested_sp': [ i for i in range(n_act) ],
		#
		# about episodes
		'init_pop': np.float32([0.5, 0.5, 0.5]),
		'reset_sigma': 0.01,
		'tmax': 1000,
		'penalty_fn': _default_penalty_fn,
		'extinct_thresh': 0.05,
		#
		# about dynamics / control
		'var_bound': 2,
		'_costs': np.zeros(n_act, dtype=np.float32),
		'_prices': np.ones(n_act, dtype=np.float32),
	}
	params = {
		'c': np.random.choice([0.2, 0.25, 0.3]),
		'D': np.random.choice([0.05, 0.1, 0.15]),
		'd_z': np.random.choice([0.2, 0.3, 0.4]),
		'K': np.random.choice([0.9, 1, 1.1]),
		'LV_xy': np.random.choice([0.05, 0.1, 0.15]),
		'r_x': np.random.choice([0.9, 1, 1.1]),
		'r_y': np.random.choice([0.9, 1, 1.1]),
		'r_z': np.random.choice([0.9, 1, 1.1]),
		#
		'sigma_x': 0.1,
		'sigma_y': 0.1,
		'sigma_z': 0.1,
	}
	info = {'metadata': metadata, 'params': params}

	env = eco_env(
		metadata = metadata,
		dyn_fn = threeSp_1, 
		dyn_params = params, 
		non_stationary = (len(non_stationarities) > 0), 
		non_stationarities = non_stationarities,
		)

	return env, info

#####################################################################
########################### 3 SP 2 ##################################
#####################################################################

def threeSp_2_factory(n_act: int = 1, non_stationarities: dict = {}):
	metadata = {
		#
		# which env class
		'n_sp':  3,
		'n_act': n_act,
		'_harvested_sp': [ i for i in range(n_act) ],
		#
		# about episodes
		'init_pop': np.float32([0.5, 0.5, 0.5]),
		'reset_sigma': 0.01,
		'tmax': 1000,
		'penalty_fn': _default_penalty_fn,
		'extinct_thresh': 0.05,
		#
		# about dynamics / control
		'var_bound': 2,
		'_costs': np.zeros(n_act, dtype=np.float32),
		'_prices': np.ones(n_act, dtype=np.float32),
	}
	params = {
		'c_x': np.random.choice([0.2, 0.25, 0.3]),
		'c_y': np.random.choice([0.2, 0.25, 0.3]),
		'd_z': np.random.choice([0.9, 1, 1.1]),
		'K_x': np.random.choice([0.9, 1, 1.1]),
		'r_x': np.random.choice([0.9, 1, 1.1]),
		'r_y': np.random.choice([0.9, 1, 1.1]),
		'r_z': np.random.choice([0.9, 1, 1.1]),
		#
		'sigma_x': 0.1,
		'sigma_y': 0.1,
		'sigma_z': 0.1,
	}
	info = {'metadata': metadata, 'params': params}

	env = eco_env(
		metadata = metadata,
		dyn_fn = threeSp_1, 
		dyn_params = params, 
		non_stationary = (len(non_stationarities) > 0), 
		non_stationarities = non_stationarities,
		)

	return env, info

#####################################################################
##################### FULL ENV FACTORY ##############################
#####################################################################

NAME_TO_ENV_FACTORY = {
	'twoSp_1': twoSp_1_factory,
	'twoSp_2': twoSp_2_factory,
	'threeSp_1': threeSp_1_factory,
	'threeSp_2': threeSp_2_factory,
}

def env_factory(env_name, n_act, non_stationarities = {}):
	return NAME_TO_ENV_FACTORY[env_name](n_act=n_act, non_stationarities=non_stationarities)