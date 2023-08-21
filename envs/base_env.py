import gymnasium as gym
import numpy as np
import warnings

from gymnasium import spaces
from typing import Callable, Dict, Union

# in-house imports
from dynamical_system import dynamical_system
from metadata import envMetadata

class eco_env(gym.Env):
	""" what all our envs will have in common """

	#
	# gym.Env API

	def __init__(
		self, 
		metadata: dict,
		dyn_fn: Callable, 
		dyn_params: Dict[str, Union[int, float, np.float32, np.float64]] = {}, 
		non_stationary: bool = False, 
		non_stationarities: Dict[str, Callable] = {},
		):

		self.metadata = envMetadata(
			**metadata, 
			)

		self.env_dyn_obj = dynamical_system(
			n=self.metadata.n_sp,
			dyn_fn=dyn_fn, 
			dyn_params=dyn_params, 
			non_stationary=non_stationary, 
			non_stationarities=non_stationarities,
			)

		self.action_space = spaces.Box(
			np.float32([-1] * self.metadata.n_act),
			np.float32([1] * self.metadata.n_act),
			dtype = np.float32,
			)

		self.observation_space = spaces.Box(
			np.float32([-1] * self.metadata.n_sp),
			np.float32([1] * self.metadata.n_sp),
			dtype = np.float32,
			)

		# easier access for objects I reference explicitly
		self.init_pop = self.metadata.init_pop
		self.reset_sigma = self.metadata.reset_sigma
		self.var_bound = self.metadata.var_bound
		self.n_sp = self.metadata.n_sp
		self.n_act = self.metadata.n_act
		
		# reset
		self.reset()

	def reset(self, *, seed=42, options=None):
		self.timestep = 0
		reset_pert = self.reset_sigma * np.random.normal(size=3)
		self.pop = self.init_pop + reset_pert
		self.state = self.pop_to_state(self.pop)
		info = {
			"init pop mean": self.init_pop, 
			"init pop perturbation": reset_pert,
			"init state realized": self.state,
		}
		return self.state, info

	def step(self, action):
		#
		# regularize actions
		action = np.clip(action, self.n_act * [-1], self.n_act * [1])
		effort = self.action_to_effort(action)
		#
		# implement harvests
		harvest_arr, self.pop = self.extract_harvest(self.pop, effort)
		reward = self.compute_profit(effort, harvest_arr)
		#
		# natural dynamics
		self.pop = self.env_dyn_obj.dyn_fn(
			* self.pop, 
			# ok to pass empty parameters object, backend deals with that.
			params=self.env_dyn_obj.dyn_params,
			t=self.timestep,
		)
		#
		# check for early end
		terminated = False
		if any(self.pop < self.metadata.extinct_thresh):
			reward += self.metadata.penalty_fn(self.timestep)
			terminated = True
		#
		# update
		self.state = self.pop_to_state(self.pop)
		self.timestep += 1
		#
		# cap off state:
		if any(x>1 for x in self.state):
			NORM_WARN = ( 
				f"State {self.state} was capped off at 1 for being out of bounds. "
				"Check dynamics or increase env.var_bound (supplied through the "
				"metadata dictionary to the env). In this run, values will be clipped "
				"at 1."
				)
			warnings.warn(NORM_WARN)
		self.state = np.clip(self.state, self.n_sp * [-1], self.n_sp * [1])
		self.pop = self.state_to_pop(self.state)
		#
		# info
		info = {"actions": action, "harvests": harvest_arr, "reward": reward, }
		return self.state, reward, terminated, False, info

	# 
	# extra methods

	def extract_harvest(self, pop, effort):
		effort_dict = dict(zip(self.metadata.harvested_sp, effort)) # {sp_index: harvest_effort}
		harvest_arr = np.float32([pop[i] * effort_dict[i] for i in self.metadata.harvested_sp])
		new_pop = np.clip(
			np.float32(
				[
					pop[i] - harvest_arr[i] if i in self.metadata.harvested_sp else pop[i]
					for i in range(self.n_sp)
				]
			),
			self.n_sp * [0],
			self.n_sp * [self.var_bound]
		)
		return harvest_arr, new_pop

	def compute_profit(self, effort_arr, harvest_arr):
		return sum(harvest_arr * self.metadata.prices - effort_arr * self.metadata.costs)

	def pop_to_state(self, pop):
		""" from pop-space [0, self.var_bound] to  state-space [-1, 1]. """
		return 2 * pop / self.var_bound - 1

	def state_to_pop(self, state):
		""" inverse of pop_to_state. """
		return (state + 1) * (self.var_bound / 2)

	def action_to_effort(self, action):
		""" from action-space [-1,1] to effort-space [0,1]. """
		return (action + 1) / 2

class ray_eco_env(gym.Env):
	""" formatted to fit ray RLLib's most natural syntax for training. """

	def __init__(self, config):
		super(ray_eco_env, self).__init__()
		self.config = config
		self.env = eco_env(
			metadata=self.config.get('metadata', self.needed_cfg('metadata')),
			dyn_fn=self.config.get('dyn_fn', self.needed_cfg('dyn_fn')),
			dyn_params=self.config.get('dyn_params', {}),
			non_stationary=self.config.get('non_stationary', None),
			non_stationarities=self.config.get('non_stationarities', {}),
		)
		# super().__init__(self.env)

		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		# print(f""" env observation space =

		# 	{self.observation_space}

		# 	""")
		# self.action_space = self.env.action_space

	def reset(self):
		return self.env.reset()

	def step(self, action):
		return self.env.step(action)

	# helpers

	def needed_cfg(self, dict_entry: str):
		raise Warning(f"ray_eco_env.config dict requires a '{dict_entry}' entry.")