from data_generation import generate_multiple_episodes
from ray_trainer_api import ray_trainer

from typing import List, Dict, Any

ALGO_LIST = [
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

def ray_algos_iter_test(
	ray_config: Dict[str, Any], 
	iterations: int, 
	algo_list: List[str] = ALGO_LIST,
):
	benchmarks = {}
	for algo in algo_list:
		try:
			RT = ray_trainer(
				algo_name=algo, 
				config=ray_config,
			)
			agent = RT.train(iterations=iterations)
			benchmarks[algo] = agent.evaluate()
			# benchmarks[algo] = generate_multiple_episodes(agent)

		except Exception as e:
			print("\n\n"+f"#################### failed for {algo}! #################### "+"\n\n")
			print(str(e))
			benchmarks[algo] = "failed run."

	return benchmarks
