#
# functions to generate episode data to plot

import pandas as pd

from typing import List, Tuple

################################################################################
########################### GENERATING EPISODES ################################
################################################################################

def generate_episode(agent, env):
	df = []
	episode_reward = 0
	observation, _ = env.reset()
	for t in range(env.env.metadata.tmax):
		action = agent.compute_single_action(observation, deterministic=True)
		pop = env.env.state_to_pop(observation) # natural units
		effort = (action + 1)/2
		df.append([t, episode_reward, 
			*effort, 
			*pop
			])

		observation, reward, terminated, done, info = env.step(action)
		episode_reward += reward
		if terminated or done:
			break
	df = pd.DataFrame(df, columns=['t', 'reward', 'effort1', 'effort2', 'x', 'y', 'z']), episode_reward

	return df

def generate_multiple_episodes(agent, env, N):
	episodes = []
	rewards = []
	for i in range(N):
		ep, rew = generate_episode(agent, env)
		ep["rep"] = i
		episodes.append(ep)
		rewards.append(rew)
	return pd.concat(episodes), rewards


################################################################################
######################### FINDING POPULAR WINDOWS ##############################
################################################################################

def popular_ranges_fixed_size(df, var: str, rel_size = 0.3):
  """ 
  Determines most populated range for var in df. 
  """
  min_val = df[var].min()
  max_val = df[var].max()
  size = rel_size * (max_val - min_val)
  
  histogram =  {}
  for window_start in np.linspace(min_val, max_val - size, 100):
    histogram[window_start] = (
      sum(
        (
          (df[var] >= window_start) &
          (df[var] <= window_start+size)
        )
      )
    )
  
  opt_window_start = max(histogram, key=histogram.get)
  return {opt_window_start: histogram[opt_window_start]}
  
def popular_ranges(df, var: str, min_fraction = 0.7):
  """ vary rel_size parameter in popular_ranges_fixed_size"""
  rel_size_list = [0.8 ** i for i in range(11)]
  N = len(df.index)
  popular_ranges_dict = {}
  for rel_size in rel_size_list:
    fixed_size_opt_range = popular_ranges_fixed_size(df, var, rel_size=rel_size)
    if list(fixed_size_opt_range.values())[0] > min_fraction * N:
      # only add the ones that satisfy the min_fraction condition
      popular_ranges_dict[rel_size] = (
        fixed_size_opt_range
      )
  
  min_popular_range_size = min(popular_ranges_dict.keys())
  return min_popular_range_size, popular_ranges_dict[min_popular_range_size]


def evaluate_policy(
	agent, 
	env,
	*,
	ranges: List[Tuple],
	var_names: List[str],
	):
	""" samples random points from cube defined by ranges, 
	evaluates the agent's policy on those points. 

	I will set things up as follows

	action = agent.compute_single_action(env.state)
	state, ... = env.step(action)

	for the interaction between agent and env
	"""
	pass


