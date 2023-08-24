import numpy as np
import os
import pandas as pd
from plotnine import ggplot, aes, geom_line

from base_env import eco_env, ray_eco_env
from env_factories import env_factory, threeSp_1_factory
from dyn_fns import threeSp_1
from util import dict_pretty_print


# ##############################################################
# ####################### Example 1 ############################
# ##############################################################
# # print("\n\n" + "env 1 test:" + "\n\n")

# def _unparametrized_dyn(X: np.float32):
# 	return np.float32([X * (1 - X)])

# def _penalty_fn(t: int):
# 	return -1 / (t+1)

# _metadata = {
# 	#
# 	# which env class
# 	'n_sp':  1,
# 	'n_act': 1,
# 	'_harvested_sp': [0],
# 	#
# 	# about episodes
# 	'init_pop': np.float32([0.5]),
# 	'reset_sigma': 0.01,
# 	'tmax': 1000,
# 	'penalty_fn': _penalty_fn,
# 	'extinct_thresh': 0.05,
# 	#
# 	# about dynamics / control
# 	'var_bound': 2,
# 	'_costs': np.zeros(1, dtype=np.float32),
# 	'_prices': np.ones(1, dtype=np.float32),
# }

# env_1 = eco_env(
# 	metadata = _metadata,
# 	dyn_fn = _unparametrized_dyn, 
# 	dyn_params = {}, 
# 	non_stationary = False, 
# 	non_stationarities = {},
# 	)

# env_1.reset()
# for _ in range(10):
# 	obs, rew, term, _, info = env_1.step(action = [-0.9])
# 	# dict_pretty_print({**info, 'state': obs})


# ##############################################################
# ####################### Example 2 ############################
# ##############################################################
# # print("\n\n" + "env 2 test:" + "\n\n")

# _params = {'r': 2, 'K': 1}

# def _parametrized_dyn(X: np.float32, params: dict):
# 	P = params
# 	return np.float32([P["r"] * X * (1 - X / P["K"])])

# env_2 = eco_env(
# 	metadata = _metadata,
# 	dyn_fn = _parametrized_dyn, 
# 	dyn_params = _params, 
# 	non_stationary = False, 
# 	non_stationarities = {},
# 	)

# env_2.reset()
# for _ in range(10):
# 	obs, rew, term, _, info = env_2.step(action = [-0.9])
# 	# dict_pretty_print({**info, 'state': obs})


# ##############################################################
# ####################### Example 3 ############################
# ##############################################################
# # print("\n\n" + "env 3 test:" + "\n\n")

# def _r(t):
# 	return 1 + t/1000


# env_3 = eco_env(
# 	metadata = _metadata,
# 	dyn_fn = _parametrized_dyn, 
# 	dyn_params = _params, 
# 	non_stationary = True, 
# 	non_stationarities = {"r": _r},
# 	)

# env_3.reset()
# for _ in range(10):
# 	obs, rew, term, _, info = env_3.step(action = [-0.9])
# 	# dict_pretty_print({**info, 'state': obs})


# ##############################################################
# ####################### Example 4 ############################
# ##############################################################
# # print("\n\n" + "env 4 test:" + "\n\n")

# def _dyn_4(X, Y, Z, params):
# 	P = params
# 	return np.float32([P['rx'] * X * (1 - X / (1 - Z)), P['ry'] *Y * (1 - Y / (1 - Z)), P['rz'] *(X + Y) * Z * (1 - Z)])

# def _penalty_fn_4(t: int):
# 	return - 1000 / (t+1)

# _metadata_4 = {
# 	#
# 	# which env class
# 	'n_sp':  3,
# 	'n_act': 2,
# 	'_harvested_sp': [0,1],
# 	#
# 	# about episodes
# 	'init_pop': np.float32([0.5, 0.5, 0.1]),
# 	'reset_sigma': 0.01,
# 	'tmax': 1000,
# 	'penalty_fn': _penalty_fn_4,
# 	'extinct_thresh': 0.05,
# 	#
# 	# about dynamics / control
# 	'var_bound': 2,
# 	'_costs': np.zeros(2, dtype=np.float32),
# 	'_prices': np.ones(2, dtype=np.float32),
# }

# def _rx(t):
# 	return 1 + t/1000

# _params_4 = {'rx': 1, 'ry': 2, 'rz': 0.1}

# _non_stationarities_4 = {'rx': _rx}

# env_4 = eco_env(
# 	metadata = _metadata_4,
# 	dyn_fn = _dyn_4, 
# 	dyn_params = _params_4, 
# 	non_stationary = True, 
# 	non_stationarities = _non_stationarities_4,
# 	)

# env_4.reset()
# for _ in range(10):
# 	obs, rew, term, _, info = env_4.step(action = [-1])
# 	# dict_pretty_print({**info, 'state': obs}, dict_name = "step info")


# ##############################################################
# ####################### Example 5 ############################
# ##############################################################
# # print("\n\n" + "env 5 test:" + "\n\n")

# _config_5 = {
# 	'metadata': _metadata_4,
# 	'dyn_fn': _dyn_4,
# 	'dyn_params': _params_4,
# 	'non_stationary': True,
# 	'non_stationarities': _non_stationarities_4,
# }

# env_5 = ray_eco_env(config=_config_5)

# env_5.reset()
# for _ in range(10):
# 	obs, rew, term, _, info = env_5.step(action = [-1])
# 	# dict_pretty_print({**info, 'state': obs}, dict_name = "step info")


#####################################################################
#################### RAY TRAINER EXAMPLE ############################
#####################################################################
print("\n\n" + "ray_trainer test:" + "\n\n")

from ray_trainer_api import ray_trainer

TMAX = 800

def utility_fn(effort, pop, cull_cost=0.001):
	return 0.5 * pop[0] - cull_cost * sum(effort)

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
	'extinct_thresh': 0.005,
	'penalty_fn': lambda t: - 5 * TMAX / (t+1),
	'var_bound': 4,
	# '_costs': np.zeros(2, dtype=np.float32),
	# '_prices': np.ones(2, dtype=np.float32),
}

params = {
	"r_x": np.float32(0.13),
	"r_y": np.float32(0.2),
	"K": np.float32(1),
	"beta": np.float32(0.5),
	"v0":  np.float32(0.1),
	"D": np.float32(0.),
	"tau_yx": np.float32(0),
	"tau_xy": np.float32(0),
	"alpha": np.float32(1), 
	"dH": np.float32(0.03),
	"sigma_x": np.float32(0.05),
	"sigma_y": np.float32(0.05),
	"sigma_z": np.float32(0.05),
}

def dyn_fn(X, Y, Z):
	global params
	p = params
	#
	return np.float32([
		X + (p["r_x"] * X * (1 - (X + p["tau_xy"] * Y) / p["K"])
            - (1 - p["D"]) * p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
            + p["sigma_x"] * X * np.random.normal()
            ),
		Y + (p["r_y"] * Y * (1 - (Y + p["tau_yx"]* X ) / p["K"] )
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

#### Algo testing:

ALGO_SET = {
	# 'a2c',
	'a3c',
	'appo',
	'ddppo',
	'ppo',
	# 'maml',
	# 'apex',
	# 'dqn',
	'ddpg',
	'td3',
	'ars',
}

#
# uncontrolled
env = ray_eco_env(config=env_config)
unctrl_data = []
episode_reward = 0
observation, _ = env.reset()
for t in range(TMAX):
	pop = env.env.state_to_pop(observation)
	observation, reward, terminated, done, info = env.step([0] * metadata['n_act'])
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
			os.path.join("..", "data", f"unctrl.png")
			)

print(f"uncontrolled reward = {episode_reward}")



def workflow(algo: str):

	print(f"Working on {algo} now...\n\n")

	global env_config

	# env_config = {
	# 			'metadata': metadata,
	# 			'dyn_fn': dyn_fn,
	# 			'utility_fn': utility_fn,
	# 		}

	####################################################################
	########################### TRAINING ###############################
	####################################################################

	try:
		RT = ray_trainer(
			algo_name=algo, 
			config=env_config,
		)
		agent = RT.train(iterations=50)

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

	data_dir = os.path.join("..", "data", algo)
	os.makedirs(data_dir, exist_ok=True)
	episode_data.to_csv(
		os.path.join(data_dir, "episodes.csv")
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
			os.path.join(data_dir, f"ep_{r}.png")
			)

	algo_eval = {"algo": [algo], "mean_rew": np.mean(rewards), "std_rew": np.std(rewards)}

	print(
		"\n\n"+
		f"{algo} evaluation reward = {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}"+
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

#### Algo testing:

# _algo_set = {
# 	'a2c',
# 	'a3c',
# 	'appo',
# 	'ddppo',
# 	'ppo',
# 	'maml',
# 	# 'apex',
# 	# 'dqn',
# 	'ddpg',
# 	'td3',
# 	'ars',
# }

# _failed_init = []
# _failed_train = []
# _init_exceptions = []
# _train_exceptions = []
# for _algo in _algo_set:
# 	print("\n\n" + f"working on {_algo}" + "\n\n")
# 	try:
# 		RT = ray_trainer(
# 			algo_name=_algo, 
# 			config=_config_ray,
# 		)
# 	except Exception as e:
# 		_failed_init.append({'algo': _algo, 'exception': str(e)})
# 		print("\n\n"+f"failed to initialize {_algo}. Exception thrown: " + "\n" + str(e))

# 	try:
# 		agent = RT.train(iterations=2)
# 	except Exception as e:
# 		_failed_train.append({'algo': _algo, 'exception': str(e)})
# 		print("\n\n"+f"failed to train {_algo}. Exception thrown: " + "\n" + str(e))

# print("\n\n"+"init exceptions:"+"\n\n")
# for fi in _failed_init:
# 	dict_pretty_print(fi)

# print("\n\n"+"train exceptions:"+"\n\n")
# for ft in _failed_train:
# 	dict_pretty_print(ft)


# for idx, fi in enumerate(_failed_init):
# 	print(f"{idx}: failed to initialize {fi}")

# for idx, ft in enumerate(_failed_train):
# 	print(f"{idx}: failed to train {ft}")

# for idx, ie in enumerate(_init_exceptions):
# 	print(f"{idx}: ")


