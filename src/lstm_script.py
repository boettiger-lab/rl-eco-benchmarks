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




import numpy as np

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, _ = try_import_torch()

# __sphinx_doc_begin__


# The custom model that will be wrapped by an LSTM.
class MyCustomModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.num_outputs = int(np.product(self.obs_space.shape))
        self._last_batch_size = None

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        # Return 2x the obs (and empty states).
        # This will further be sent through an automatically provided
        # LSTM head (b/c we are setting use_lstm=True below).
        return obs * 2.0, []

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))


if __name__ == "__main__":
    ray.init()

    # Register the above custom model.
    ModelCatalog.register_custom_model("my_torch_model", MyCustomModel)

    # Create the Algorithm from a config object.
    config = (
        ppo.PPOConfig()
        .environment(base_env.ray_eco_env)
        .framework("torch")
        .training(
            model={
                # Auto-wrap the custom(!) model with an LSTM.
                "use_lstm": True,
                # To further customize the LSTM auto-wrapper.
                "lstm_cell_size": 64,
                # Specify our custom model from above.
                # "custom_model": "my_torch_model",
                # Extra kwargs to be passed to your model's c'tor.
                # "custom_model_config": {},
            }
        )
    )
    config.env_config = problem_summary
    algo = config.build()
    algo.train()
    algo.stop()

# __sphinx_doc_end__
