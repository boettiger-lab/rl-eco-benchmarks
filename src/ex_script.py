import numpy as np
import os
import pandas as pd
from plotnine import ggplot, aes, geom_line

from base_env import eco_env, ray_eco_env
from env_factories import env_factory, threeSp_1_factory
from dyn_fns import threeSp_1
from util import dict_pretty_print

#####################################################################
#################### RAY TRAINER EXAMPLE ############################
#####################################################################
print("\n\n" + "ray_trainer test:" + "\n\n")

import json

from ray_trainer_api import ray_trainer

TMAX = 800
DATA_DIR = os.path.join("..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

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

iterations = 5_000

with open(os.path.join(DATA_DIR, "params.json"), 'w') as params_file:
	json.dump(
		{
		'params': {key: str(value) for key, value in params.items()}, 
		'metadata': {key: str(value) for key, value in metadata.items()}, 
		'iterations': iterations,
		}, 
		params_file)

#### Algo testing:

ALGO_SET = [
	# 'a2c',
	'ddpg',
	'ddppo',
	'td3',
	'ars',
	'a3c',
	'appo',
	'ppo',
	# 'maml',
	# 'apex',
	# 'dqn',
]

#
# uncontrolled
env = ray_eco_env(config=env_config)
unctrl_data = []
episode_reward = 0
observation, _ = env.reset()
for t in range(TMAX):
	pop = env.env.state_to_pop(observation)
	# remember that actions are [-1,1] valued!
	observation, reward, terminated, done, info = env.step([-1] * metadata['n_act'])
	#
	# notice that for some utility-functions reward can be non-zero even if action is zero
	episode_reward += reward
	unctrl_data.append([t, *pop, episode_reward])

	if done or terminated:
		break

unctrl_df = pd.DataFrame(unctrl_data, columns = ["t", "X", "Y", "Z", "reward"])

unctrl_plt = ggplot(
		unctrl_df[["t", "X", "Y", "Z"]].melt(["t"]),
		aes("t", "value", color="variable"),
		) + geom_line()

unctrl_plt.save(
			os.path.join(DATA_DIR, f"unctrl.png")
			)

print(f"uncontrolled reward = {episode_reward}")



def workflow(algo: str):

	print(f"Working on {algo} now...\n\n")

	global env_config
	global iterations
	global DATA_DIR

	####################################################################
	########################### TRAINING ###############################
	####################################################################

	try:
		RT = ray_trainer(
			algo_name=algo, 
			config=env_config,
		)
		agent = RT.train(iterations=iterations)

	except Exception as e:
		print("\n\n"+f"#################### failed for {algo}! #################### "+"\n\n")
		print(str(e))
		return {"algo": [algo], "mean_rew": "failed", "std_rew": "failed"}

	print(f"Done training {algo}.")

	####################################################################
	################### EVALUATION AND PLOTTING ########################
	####################################################################

	#
	# helper functions

	def generate_episode(agent, env):
		df = []
		episode_reward = 0
		observation, _ = env.reset()
		for t in range(env.env.metadata.tmax):
			action = agent.compute_single_action(observation, deterministic=True)
			pop = env.env.state_to_pop(observation) # natural units
			effort = (action + 1)/2
			df.append([
				t, 
				episode_reward, 
				*effort, 
				*pop
				])
			#
			observation, reward, terminated, done, info = env.step(action)
			episode_reward += reward
			if terminated or done:
				break
		df = pd.DataFrame(df, columns=['t', 'reward', 'y_cull', 'z_cull', 'x', 'y', 'z'])
		#
		return df, episode_reward

	def plot_episode(df):
	    """ plots an episode df. df generated using generate_episode(). """
	    return ggplot(
	        df.melt(["t"]),
	        aes("t", "value", color="variable"),
	    ) + geom_line()


	#
	# generate data

	print("Generating data...")

	env = ray_eco_env(config=env_config)

	rewards = []
	episodes = []
	for i in range(20):
		print(i, end="\r")
		ep_df, ep_rew = generate_episode(agent, env)
		rewards.append(ep_rew)
		ep_df["rep"] = i
		episodes.append(ep_df)

	episode_data = pd.concat(episodes)

	#
	# save data

	print("Saving data...")

	algo_dir = os.path.join(DATA_DIR, algo)
	os.makedirs(algo_dir, exist_ok=True)
	episode_data.to_csv(
		os.path.join(algo_dir, "episodes.csv")
		)

	#
	# plot data

	print("Generating plots...")

	for r in range(5):
		plot = plot_episode(
			episode_data.loc[episode_data.rep == r][
				['t', 'y_cull', 'z_cull', 'x', 'y', 'z']
			]
			)
		plot.save(
			os.path.join(algo_dir, f"ep_{r}.png")
			)

	mean_rew_str = f"{np.mean(rewards):.3f}"
	stdev_rew_str = f"{np.std(rewards):.3f}"
	algo_eval = {"algo": [algo], "mean_rew": mean_rew_str, "std_rew": stdev_rew_str}

	print(
		"\n\n"+
		f"{algo} evaluation reward = {mean_rew_str} +/- {stdev_rew_str}"+
		"\n\n"
		)

	return algo_eval

evals_ = []
algos_tested = []
for algo in ALGO_SET:
	print(f"Have already tested: {algos_tested}. {len(ALGO_SET) - len(algos_tested)} algos to go.")
	algos_tested.append(algo)
	algo_eval = workflow(algo)
	evals_.append(pd.DataFrame(algo_eval))

evals_df = pd.concat(evals_)
print(evals_df.head(7))
evals_df.to_csv(os.path.join("..", "data", "summary.csv"))
