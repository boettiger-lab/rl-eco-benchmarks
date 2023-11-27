import json
import numpy as np
import os
import pandas as pd
from plotnine import ggplot, aes, geom_line # later for plotting evaluation

from ray.rllib.examples.models.trajectory_view_utilizing_models import \
	FrameStackingCartPoleModel, TorchFrameStackingCartPoleModel
from ray.rllib.models.catalog import ModelCatalog

from base_env import ray_eco_env
from util import dict_pretty_print
from ray_trainer_api import ray_trainer

# ###################################
# ############# GLOBALS #############
# ###################################

TMAX = 800
DATA_DIR = os.path.join("..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ###################################
# ###### PROBLEM SPECIFICATION ######
# ###################################

def utility_fn(effort, pop, cull_cost=0.001):
	""" reward in each time step """
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

params = { # dynamic parameters used by dyn_fn
	"r_x": np.float32(0.12),
	"r_y": np.float32(0.2),
	"K": np.float32(1),
	"beta": np.float32(0.1),
	"v0":  np.float32(0.1),
	"D": np.float32(-0.1),
	"tau_yx": np.float32(0),
	"tau_xy": np.float32(0),
	"alpha": np.float32(1), 
	"dH": np.float32(0.1),
	"sigma_x": np.float32(0.05),
	"sigma_y": np.float32(0.05),
	"sigma_z": np.float32(0.05),
}

def dyn_fn(X, Y, Z):
	""" the dynamics of the system """
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

# summarize problem into a dict (the syntax that our interface uses):
#
problem_summary = {
				'metadata': metadata,
				'dyn_fn': dyn_fn,
				'utility_fn': utility_fn,
			}

import base_env

train_config = {
	"env": base_env.eco_env(**problem_summary),
	"rl_module": {
		"_enable_rl_module_api": False,
	}
	"training": {
		"_enable_learner_api": False,
	}
	"model": {
		"vf_share_layers": True,
		"custom_model": "frame_stack_model",
		"custom_model_config": {
			"num_frames": 10,
		},
		# "use_lstm": True,
		# "lstm_cell_size": 64,
		# "max_seq_len": 10,
		# "lstm_use_prev_action": True,
	},
	"num_sgd_iter": 5,
	"vf_loss_coeff": 0.0001,
	"framework": "torch",
}

# ###################################
# ######## INIT & TRAIN ALGO ########
# ###################################

from ray import tune
import ray

ray.init(num_cpus=3)

ModelCatalog.register_custom_model(
	"frame_stack_model", TorchFrameStackingCartPoleModel
)

stop = {
	"training_iteration": 200,
}

results = tune.run(
	"PPO", config=train_config, stop=stop, verbose=2, checkpoint_at_end=True
)

# ITERATIONS = 100
# ALGO = "ppo"

# RT = ray_trainer( # wrapper class around ray RLLib algorithms.
# 	algo_name=ALGO, 
# 	config=problem_summary,
# 	train_cfg=train_config,
# )
# agent = RT.train(iterations=ITERATIONS)

