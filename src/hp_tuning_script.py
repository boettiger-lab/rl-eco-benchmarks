import numpy as np
import os
import pandas as pd

from ray_trainer_api import ray_trainer

from ray.rllib import _register_all
_register_all()

DATA_DIR = os.path.join("..", "data", "hp_tuning")
os.makedirs(DATA_DIR, exist_ok=True)

#
# Ecology / control:

def utility_fn(effort, pop, cull_cost=0.001):
	return 0.5 * pop[0] - cull_cost * sum(effort)

def penalty_fn(t):
	return - 5 * 800 / (t+1)

metadata = {
	#
	# structure of ctrl problem
	'name': 'minicourse_challenge', 
	'n_sp':  3,
	'n_act': 2,
	'controlled_species': [1,2],
	#
	# about episodes
	'init_pop': np.float32([0.5, 0.5, 0.2]),
	'reset_sigma': 0.01,
	'tmax': 800,
	#
	# about dynamics / control
	'extinct_thresh': 0.03,
	'penalty_fn': lambda t: - 5 * 800 / (t+1),
	'var_bound': 4,
}

def dyn_fn(X, Y, Z, params):
	p = params
	#
	return np.float32([
		X + (p["r_x"] * X * (1 - X / p["K"])
            - (1 - p["D"]) * p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
            + p["sigma_x"] * X * np.random.normal()
            ),
		Y + (p["r_y"] * Y * (1 - Y / p["K"] )
				- (1 + p["D"]) * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
				+ p["sigma_y"] * Y * np.random.normal()
				),
		Z + p["alpha"] * p["beta"] * Z * (
				(1-p["D"]) * (X**2) / (p["v0"]**2 + X**2)
				+ (1 + p["D"])  * (Y**2) / (p["v0"]**2 + Y**2)
				) - p["dH"] * Z +  p["sigma_z"] * Z  * np.random.normal()
	])

params = {
	"r_x": np.float32(0.12),
	"r_y": np.float32(0.2),
	"K": np.float32(1),
	"beta": np.float32(0.1),
	"v0":  np.float32(0.1),
	"D": np.float32(0.),
	"tau_yx": np.float32(0),
	"tau_xy": np.float32(0),
	"alpha": np.float32(1), 
	"dH": np.float32(0.1),
	"sigma_x": np.float32(0.05),
	"sigma_y": np.float32(0.05),
	"sigma_z": np.float32(0.05),
}


#
# do some RL

env_config = {
				'metadata': metadata,
				'dyn_fn': dyn_fn,
				'dyn_params': params,
				'utility_fn': utility_fn,
			}

RT = ray_trainer(
	algo_name="ppo", 
	config=env_config,
)

hp_dicts_list = [
	{
	'name': 'gamma',
	'val_type_str': 'float',
	'low_bound': 0.8,
	'high_bound': 0.9997,
	},
	{
	'name': 'lambda',
	'val_type_str': 'float',
	'low_bound': 0.9,
	'high_bound': 1,
	},
	{
	'name': 'kl_target',
	'val_type_str': 'float',
	'low_bound': 0.0003,
	'high_bound': 0.003,
	},
	{
	'name': 'clip_param',
	'val_type_str': 'categorical',
	'value_list': [0.1, 0.2, 0.3],
	}
]

tuning_df = RT.tune_hyper_params(
	hp_dicts_list=hp_dicts_list,
	num_workers=10,
	num_samples=12,
	criteria_max=10_000,
	)

print(tuning_df)


tuning_df.to_csv(os.path.join(DATA_DIR, 'ppo_tuning_results.csv'))

# [
# 	{
# 	'name':'lr', 
# 	'val_type_str':'float',
# 	'low_bound': 0.0001,
# 	'high_bound': 0.005,
# 	}
# ]

