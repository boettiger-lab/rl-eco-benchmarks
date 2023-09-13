import numpy as np
import pandas as pd
import torch

from datetime import datetime
from typing import List, Union

from ray.tune import run, sample_from
from ray.tune.schedulers.pb2 import PB2

from hyperparams import make_hyperparams
from util import dict_pretty_print

# docs: 
# https://docs.ray.io/en/latest/tune/api/doc/ (cont.)
# ray.tune.schedulers.pb2.PB2.html#:~:text= (cont.)
# Implements%20the%20Population%20Based%20Bandit, (cont.)
# selected%20using%20GP%2Dbandit%20optimization. (end)


def sb2_tuning(
	algo_name, 
	env_name,
	env_config,
	hp_dicts_list, # see 'make_hyperparams' for required structure
	num_workers=20,
	num_samples=50,
	perturbation_interval=50_000,
	seed=42,
	horizon=1000,
	quantile_fraction=0.25,
	criteria="timesteps_total",
	criteria_max=10_000,
	ff_net_sizes: List[int] = [32, 32],
	):

	# only consider pb2 here
	# 	
	# pbt = PopulationBasedTraining(
	# 	time_attr=criteria,
	# 	metric="episode_reward_mean",
	# 	mode="max",
	# 	perturbation_interval=perturbation_interval,
	# 	resample_probability=quantile_fraction,
	# 	quantile_fraction=quantile_fraction,
	# 	hyperparam_mutations = {
	# 		name: (lambda: random.uniform(value[0], value[1])) 
	# 		for name, value in hyperparam_bounds_dict
	# 		}
	# 	)

	hyperparams_list = make_hyperparams(hp_dicts_list)

	pb2 = PB2(
		time_attr=criteria,
		metric="episode_reward_mean",
		mode="max",
		perturbation_interval=perturbation_interval,
		quantile_fraction=quantile_fraction,
		hyperparam_bounds={
			hp.name: [hp.low_bound, hp.high_bound] 
			for hp in hyperparams_list
			if hp.val_type_str in ['float', 'int']
			},
		)

	print("PB2 defined\n")

	hyperparam_mutations = {
		hp.name: hp.sample_fn() for hp in hyperparams_list
		}

	hp_mutations_ = {
		name: sample_from(sample_fn) for name, sample_fn in hyperparam_mutations.items()
		}

	print("hp mutations sampled\n")

	analysis = run(
		algo_name.upper(),
		name="{}_{}_seed{}".format(
			algo_name, env_name, str(seed)
		),
		scheduler=pb2,
		verbose=1,
		num_samples=num_samples,
		stop={criteria: criteria_max},
		log_to_file=True,
		config={
			"env": env_name,
			"env_config": env_config,
			"log_level": "DEBUG",
			"seed": seed,
			"kl_coeff": 1.0,
			"num_gpus": torch.cuda.device_count(),
			"horizon": horizon,
			"observation_filter": "MeanStdFilter",
			"model": {
				"fcnet_hiddens": ff_net_sizes,
				"free_log_std": True,
			},
			"num_sgd_iter": 10,
			"sgd_minibatch_size": 128,
			**hp_mutations_,
		},
	)

	print("ran simulations\n")

	# 
	# gather results
	#

	#
	# factor out utility function
	def process_df(df, i):
		"""
		- 'thins out' df
		- adds 'Agent' column 
		"""
		df = df[
						[
							"timesteps_total",
							"episodes_total",
							"episode_reward_mean",
							# "info/learner/default_policy/cur_kl_coeff",
						]
					]
		df.loc["Agent"] = i
		return df

	all_dfs = analysis.trial_dataframes
	"""
	source:
	@property
	def trial_dataframes(self) -> Dict[str, DataFrame]:
	"""
	names = list(all_dfs.keys())[:num_samples] # just take num_samples

	df = (
		pd.concat(
			[process_df(all_dfs[name], i) for i, name in enumerate(names)]
		)
		.reset_index(drop=True)
	)

	return df
	
