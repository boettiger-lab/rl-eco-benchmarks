#
# factory for standard choices of env based on # species and actions
#
from base_env import eco_env, ray_eco_env

def env_factory(**env_config):
	return eco_env(**env_config)

def ray_env_factory(env_config):
	print("\n ############ \nENV CONFIG:\n\n", env_config, "\n############")
	assert False, "was here."
	return eco_env(**env_config)