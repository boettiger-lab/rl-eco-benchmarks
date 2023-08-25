# 
# A simple example of the interface
# 

import numpy as np
import os

from ray_trainer_api import ray_trainer
from data_generation import generate_multiple_episodes

#
# make dir

DATA_DIR = os.path.join("..", "data", "example")
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
				'params': params,
				'utility_fn': utility_fn,
			}

RT = ray_trainer(
	algo_name="ddppo", 
	config=env_config,
)
agent = RT.train(iterations=250)


#
# evaluate

env = ray_eco_env(config=env_config)
episodes, rewards = generate_multiple_episodes(agent, env, N=50)

print("\n\n"+f"evaluation rewards = {np.mean(rewards)} +/- {np.std(rewards)}"+"\n\n")

first_ep = episodes.loc[episodes.rep==0]

dynamics_plt = ggplot(
	first_ep[['t', 'reward', 'x', 'y', 'z']].melt(["t"]),
	aes("t", "value", color="variable"),
	) + geom_line()

dynamics_plt.save("../data/example/dynamics.png")

ctrl_plt = ggplot(
	first_ep[['t', 'effort1', 'effort2']].melt(["t"]),
	aes("t", "value", color="variable"),
	) + geom_line()

ctrl_plt.save("../data/example/ctrl.png")



