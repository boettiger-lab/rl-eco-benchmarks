#
# factory for standard choices of env based on # species and actions
#
from base_env import eco_env, ray_eco_env
from util import dict_pretty_print

def env_factory(**env_config):
	return eco_env(**env_config)

def ray_env_factory(env_config):
	print("\n ############ \nENV CONFIG:\n\n")
	dict_pretty_print(env_config)
	print("############")
	return ray_eco_env(env_config)