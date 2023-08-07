#
# factory for standard choices of env based on # species and actions
#

import numpy as np

from base_env import general_env
from dyn_fns import (
	twoSp_1, twoSp_2, threeSp_1, threeSp_2, fourSp_1
)

def _default_penalty_fn(t: int):
	return - 1000 / (t+1)

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
	}
	info = {'metadata': metadata, 'params': params}

	env = general_env(
		metadata = metadata,
		dyn_fn = twoSp_1, 
		dyn_params = params, 
		non_stationary = (len(non_stationarities) > 0), 
		non_stationarities = non_stationarities,
		)

	return env, info
