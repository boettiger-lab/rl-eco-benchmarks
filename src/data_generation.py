#
# functions to generate episode data to plot

import pandas as pd

def generate_episode(agent, env, only_df=False):
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
	df = pd.DataFrame(df, columns=['t', 'reward', 'x_cull', 'y_cull', 'x', 'y', 'z'])

	return {'df': df, 'reward': episode_reward}