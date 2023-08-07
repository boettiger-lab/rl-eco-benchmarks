import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, List

from metadata import ParamObj_3s, env_metadata

def make_env(n_species, n_actions, config):

	
	if (n_species == 3) & (n_actions == 3):
		return 3s3a(config)
	else:
		raise ValueError(f"No env choice found for {n_species} species, {n_action} actions.")

class 3s3a(gym.Env):
	""" top class in this file's hierarchy: maximal complexity class of the file. """

	def __init__(self, config: dict):

		try:
			self.dyn_fn = config["dyn_fn"]
			self.dyn_params = ParamObj_3s(
				config["dyn_params"]
			)
			self.env_md = env_metadata(
				config["env_metadata"]
			)

		except:
			raise Warning(
				"config dict must contain the following keys: "
				"'dyn_fn': Callable, 'dyn_params': dict {str: float}, "
				"'env_metadata': dict {str: float or int}"
				)


		self.action_space = spaces.Box(
			np.float32(3 * [-1]),
			np.float32(3 * [1]),
			dtype = np.float32,
		)

		self.observation_space = spaces.Box(
			np.float32(3 * [-1]),
			np.float32(3 * [1]),
			dtype = np.float32,
		)

		self.reset()

		# check compatibility of parameters, init state and dynamics
		try:
			_test_init_pop = [0.5, 0.5, 0.5]
			dyn_closure = lambda params: self.dyn_fn(*_test_init_pop, params)
			dyn_closure(self.dyn_params)

		except:
			""" fixing init pop to be a value I know should work, the problem is in params. """
			raise Warning(
				"self.dyn_params not compatible with self.dyn_fn \n"
				f"dyn_params = {self.dyn_params} \n"
				f"dyn_fn = {self.dyn_fn.__name__}."
				)

		try:
			self.dyn_fn(*env_md.init_pop, self.dyn_params)

		except:
			""" If a problem happens here, it is due to the initial pop. """
			raise Warning(
				"self.env_md.init_pop not compatible with self.dyn_fn \n"
				f"init_pop = {self.env_md.init_pop} \n"
				f"dyn_fn = {self.dyn_fn.__name__}."
				)



	def reset(self):
		self.timestep = 0
		reset_pert = np.random.normal(scale = reset_sigma)
		self.pop = np.float32(self.env_md.init_pop + reset_pert)
		self.state = self.pop_to_state(self.pop)
		info = {
			"init pop mean": self.env_md.init_pop, 
			"init pop perturbation": reset_pert,
			"init state realized": self.state,
		}
		return self.state, info

	def step(self, action):
		old_pop = self.pop.copy() # copy for safety on user-defined dyn_fn
		
		self.pop = np.float32(self.dyn_fn(old_pop))


