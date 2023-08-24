from escapement import escapement_policy
from base_env import ray_eco_env

import numpy as np
import ray

#
# setting up the env

TMAX=800

def utility_fn(effort, pop, cull_cost=0.001):
	return 0.5 * pop[0] - cull_cost * sum(effort)

def penalty_fn(t):
	""" penalty for ending episode at t<TMAX steps. """
	global TMAX
	return - 5 * TMAX / (t+1)

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
	'tmax': TMAX,
	#
	# about dynamics / control
	'extinct_thresh': 0.03,
	'penalty_fn': lambda t: - 5 * TMAX / (t+1),
	'var_bound': 4,
	# '_costs': np.zeros(2, dtype=np.float32),
	# '_prices': np.ones(2, dtype=np.float32),
}

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

def dyn_fn(X, Y, Z):
	global params
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

env_config = {
				'metadata': metadata,
				'dyn_fn': dyn_fn,
				'utility_fn': utility_fn,
			}

env = ray_eco_env(config=env_config)

EscObj_handle = escapement_policy.remote(
	n_sp=env.metadata.n_sp,
	n_act=env.metadata.n_act,
	controlled_sp=env.metadata.controlled_species,
	max_esc=env.metadata.var_bound,
	)

best_esc_fn_ref = EscObj_handle.optimize.remote(env, verbose=True)

best_esc_fn = ray.get(object_ref)

