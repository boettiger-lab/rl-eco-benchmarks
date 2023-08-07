from base_env import eco_env, ray_eco_env

RAY_ALGOS_CONFIG = {
  'a2c': A2CConfig,
  'a3c': A3CConfig,
  'maml': MAMLConfig,
  'apex': ApexDQNConfig,
  'dqn': DQNConfig,
  'ddpg': DDPGConfig,
  'td3': TD3Config,
  'sac': SACConfig,
  'ars': ARSConfig,
}

def make_ray_trainer(
  algo_name,
  config):
  env_name = "ray_eco_env-v0"
  register_env("ray_eco_env-v0", ray_eco_env)
  config = RAY_ALGOS_CONFIG[algo_name]()
  config.training(vf_clip_param = 50.0)
  config.num_envs_per_worker=30
  config = config.resources(num_gpus=torch.cuda.device_count())
  config.framework_str="torch"
  config.create_env_on_local_worker = True
  config.env=env_name
  #
  config.env_config = config
  agent = config.build()
  # agent = PPOTrainer(config=config)
  return agent