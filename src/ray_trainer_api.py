import os
import torch

# from ray.tune import register_env
from ray import tune
from gymnasium.envs.registration import register
from dataclasses import dataclass
from typing import Dict, List

from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.maml import MAMLConfig
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.ars import ARSConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ddppo import DDPPOConfig
from ray.rllib.algorithms.ppo import PPOConfig

from env_factories import env_factory, ray_env_factory
from tuners import sb2_tuning

class ray_trainer:
	""" an RL agent training on one of ray's algorithms. """

	def __init__(
		self, 
		algo_name, 
		config, 
		env_registration_name: str = 'my_env',
	):
		#
		# env
		tune.register_env(
			env_registration_name, 
			lambda env_config: ray_env_factory(env_config=env_config)
		)
		#
		# algo
		self.algo_name = algo_name
		self.algo_config = self._make_config()
		#
		# boiler plate algo settings
		if algo_name == "ppo":
			self.algo_config.training(vf_clip_param = 50.0)
		self.algo_config.disable_env_checking = True # otherwise it complains about the env
		self.algo_config.framework_str="torch"
		self.algo_config.create_env_on_local_worker = True
		#
		# computational resources
		self.algo_config.num_envs_per_worker=50
		self.cpus_to_use = os.cpu_count() // 2
		self.algo_config = self.algo_config.resources(
			num_gpus=torch.cuda.device_count(), 
			num_gpus_per_worker=0.5,
			num_cpus_per_worker=30,
		)
		# 
		# config.env
		self.algo_config.env=env_registration_name
		self.algo_config.env_config = config		
		#
		# agent
		self.agent = self.algo_config.build()
		#
		# shorthands
		self.env_name = self.algo_config.env 

	def _make_config(self):
		config_dict = {
			'a2c': A2CConfig,
			'a3c': A3CConfig,
			'appo': APPOConfig,
			'ddppo': DDPPOConfig,
			'ppo': PPOConfig,
			'maml': MAMLConfig,
			'apex': ApexDQNConfig,
			'dqn': DQNConfig,
			'ddpg': DDPGConfig,
			'td3': TD3Config,
			'ars': ARSConfig,
		}
		return config_dict[self.algo_name]()

	def train(
		self, 
		iterations: int,
		save_checkpoint: bool = False,
		path_to_checkpoint: str = "../cache", 
		verbose: bool = True
	):
		if save_checkpoint and path_to_checkpoint == "":
			warning.warn("Empty checkpoint path received, setting it to '../cache'.")
			path_to_checkpoint = "../cache"
		#
		for i in range(iterations):
			if verbose:
				print(f"iteration nr. {i}", end="\r")
			self.agent.train()
		if save_checkpoint:
			checkpoint = self.agent.save(os.path.join(path_to_checkpoint, f"PPO{iterations}_checkpoint"))
		return self.agent

	def tune_hyper_params(self, hp_dicts_list, **kwargs):
		""" eventually allow for different schedulers for optimizaton """
		"""
		args:
			hp_dicts_list: see sb2_tuning implementation for required structure
			**kwargs: see sb2_tuning optional kwargs
		"""
		return sb2_tuning(
			algo_name=self.algo_name,
			env_name=self.env_name,
			hp_dicts_list=hp_dicts_list,
			**kwargs,
			)
		
