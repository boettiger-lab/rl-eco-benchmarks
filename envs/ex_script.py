import numpy as np
from base_env import eco_env, ray_eco_env
from env_factories import env_factory, threeSp_1_factory
from dyn_fns import threeSp_1
from util import dict_pretty_print

##############################################################
####################### Example 1 ############################
##############################################################
# print("\n\n" + "env 1 test:" + "\n\n")

def _unparametrized_dyn(X: np.float32):
	return np.float32([X * (1 - X)])

def _penalty_fn(t: int):
	return -1 / (t+1)

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

env_1 = eco_env(
	metadata = _metadata,
	dyn_fn = _unparametrized_dyn, 
	dyn_params = {}, 
	non_stationary = False, 
	non_stationarities = {},
	)

env_1.reset()
for _ in range(10):
	obs, rew, term, _, info = env_1.step(action = [-0.9])
	# dict_pretty_print({**info, 'state': obs})


##############################################################
####################### Example 2 ############################
##############################################################
# print("\n\n" + "env 2 test:" + "\n\n")

_params = {'r': 2, 'K': 1}

def _parametrized_dyn(X: np.float32, params: dict):
	P = params
	return np.float32([P["r"] * X * (1 - X / P["K"])])

env_2 = eco_env(
	metadata = _metadata,
	dyn_fn = _parametrized_dyn, 
	dyn_params = _params, 
	non_stationary = False, 
	non_stationarities = {},
	)

env_2.reset()
for _ in range(10):
	obs, rew, term, _, info = env_2.step(action = [-0.9])
	# dict_pretty_print({**info, 'state': obs})


##############################################################
####################### Example 3 ############################
##############################################################
# print("\n\n" + "env 3 test:" + "\n\n")

def _r(t):
	return 1 + t/1000


env_3 = eco_env(
	metadata = _metadata,
	dyn_fn = _parametrized_dyn, 
	dyn_params = _params, 
	non_stationary = True, 
	non_stationarities = {"r": _r},
	)

env_3.reset()
for _ in range(10):
	obs, rew, term, _, info = env_3.step(action = [-0.9])
	# dict_pretty_print({**info, 'state': obs})


##############################################################
####################### Example 4 ############################
##############################################################
# print("\n\n" + "env 4 test:" + "\n\n")

def _dyn_4(X, Y, Z, params):
	P = params
	return np.float32([P['rx'] * X * (1 - X / (1 - Z)), P['ry'] *Y * (1 - Y / (1 - Z)), P['rz'] *(X + Y) * Z * (1 - Z)])

def _penalty_fn_4(t: int):
	return - 1000 / (t+1)

_metadata_4 = {
	#
	# which env class
	'n_sp':  3,
	'n_act': 2,
	'_harvested_sp': [0,1],
	#
	# about episodes
	'init_pop': np.float32([0.5, 0.5, 0.1]),
	'reset_sigma': 0.01,
	'tmax': 1000,
	'penalty_fn': _penalty_fn_4,
	'extinct_thresh': 0.05,
	#
	# about dynamics / control
	'var_bound': 2,
	'_costs': np.zeros(2, dtype=np.float32),
	'_prices': np.ones(2, dtype=np.float32),
}

def _rx(t):
	return 1 + t/1000

_params_4 = {'rx': 1, 'ry': 2, 'rz': 0.1}

_non_stationarities_4 = {'rx': _rx}

env_4 = eco_env(
	metadata = _metadata_4,
	dyn_fn = _dyn_4, 
	dyn_params = _params_4, 
	non_stationary = True, 
	non_stationarities = _non_stationarities_4,
	)

env_4.reset()
for _ in range(10):
	obs, rew, term, _, info = env_4.step(action = [-1])
	# dict_pretty_print({**info, 'state': obs}, dict_name = "step info")


##############################################################
####################### Example 5 ############################
##############################################################
# print("\n\n" + "env 5 test:" + "\n\n")

_config_5 = {
	'metadata': _metadata_4,
	'dyn_fn': _dyn_4,
	'dyn_params': _params_4,
	'non_stationary': True,
	'non_stationarities': _non_stationarities_4,
}

env_5 = ray_eco_env(config=_config_5)

env_5.reset()
for _ in range(10):
	obs, rew, term, _, info = env_5.step(action = [-1])
	# dict_pretty_print({**info, 'state': obs}, dict_name = "step info")


#####################################################################
#################### RAY TRAINER EXAMPLE ############################
#####################################################################
print("\n\n" + "ray_trainer test:" + "\n\n")

from ray_trainer_api import ray_trainer

_metadata = {
	#
	# which env class
	'name': 'threeSp_1',
	'n_sp':  3,
	'n_act': 2,
	'_harvested_sp': [0,1],
	#
	# about episodes
	'init_pop': np.float32([0.5, 0.5, 0.1]),
	'reset_sigma': 0.01,
	'tmax': 1000,
	'penalty_fn': _penalty_fn_4,
	'extinct_thresh': 0.05,
	#
	# about dynamics / control
	'var_bound': 4,
	'_costs': np.zeros(2, dtype=np.float32),
	'_prices': np.ones(2, dtype=np.float32),
}

_params = {
	'c': np.random.choice([0.2, 0.25, 0.3]),
	'D': np.random.choice([0.05, 0.1, 0.15]),
	'd_z': np.random.choice([0.2, 0.3, 0.4]),
	'K_x': np.random.choice([0.9, 1, 1.1]),
	'LV_xy': np.random.choice([0.05, 0.1, 0.15]),
	'r_x': np.random.choice([0.9, 1, 1.1]),
	'r_y': np.random.choice([0.9, 1, 1.1]),
	'r_z': np.random.choice([0.9, 1, 1.1]),
	#
	'sigma_x': 0.1,
	'sigma_y': 0.1,
	'sigma_z': 0.1,
}

def _dyn_fn(x,y,z):
	global _params
	return threeSp_1(x,y,z,_params)

_config_ray = {
	'metadata': _metadata,
	'dyn_fn': _dyn_fn,
	'dyn_params': {},
	'non_stationary': False,
	'non_stationarities': {},
}

_algo_set = {
	'a2c',
	'a3c',
	'appo',
	'ddppo',
	'ppo',
	'maml',
	# 'apex',
	# 'dqn',
	'ddpg',
	'td3',
	'ars',
}

_algo = 'ppo'
_failed_init = []
_failed_train = []
_init_exceptions = []
_train_exceptions = []
for _algo in _algo_set:
	print("\n\n" + f"working on {_algo}" + "\n\n")
	try:
		RT = ray_trainer(
			algo_name=_algo, 
			config=_config_ray,
		)
	except Exception as e:
		_failed_init.append({'algo': _algo, 'exception': str(e)})
		print("\n\n"+f"failed to initialize {_algo}. Exception thrown: " + "\n" + str(e))

	try:
		agent = RT.train(iterations=2)
	except Exception as e:
		_failed_train.append({'algo': _algo, 'exception': str(e)})
		print("\n\n"+f"failed to train {_algo}. Exception thrown: " + "\n" + str(e))

print("\n\n"+"init exceptions:"+"\n\n")
for fi in _failed_init:
	dict_pretty_print(fi)

print("\n\n"+"train exceptions:"+"\n\n")
for ft in _failed_train:
	dict_pretty_print(ft)


# for idx, fi in enumerate(_failed_init):
# 	print(f"{idx}: failed to initialize {fi}")

# for idx, ft in enumerate(_failed_train):
# 	print(f"{idx}: failed to train {ft}")

# for idx, ie in enumerate(_init_exceptions):
# 	print(f"{idx}: ")


