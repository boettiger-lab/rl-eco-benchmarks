import argparse
import numpy as np

import ray
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.examples.models.trajectory_view_utilizing_models import \
    FrameStackingCartPoleModel, TorchFrameStackingCartPoleModel
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray import tune

tf1, tf, tfv = try_import_tf()


###
# ECOLOGY

# ###################################
# ############# GLOBALS #############
# ###################################

TMAX = 800
# DATA_DIR = os.path.join("..", "data")
# os.makedirs(DATA_DIR, exist_ok=True)

# ###################################
# ###### PROBLEM SPECIFICATION ######
# ###################################

def utility_fn(effort, pop, cull_cost=0.001):
    """ reward in each time step """
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

params = { # dynamic parameters used by dyn_fn
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
    """ the dynamics of the system """
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

# summarize problem into a dict (the syntax that our interface uses):
#
problem_summary = {
                'metadata': metadata,
                'dyn_fn': dyn_fn,
                'utility_fn': utility_fn,
            }

import base_env
env = base_env.eco_env(**problem_summary)

###

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=50,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=200000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=150.0,
    help="Reward at which we stop training.")

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=3)

    num_frames = 16

    ModelCatalog.register_custom_model(
        "frame_stack_model", FrameStackingCartPoleModel
        if args.framework != "torch" else TorchFrameStackingCartPoleModel)

    config = {
        "env": env,
        "model": {
            "vf_share_layers": True,
            "custom_model": "frame_stack_model",
            "custom_model_config": {
                "num_frames": num_frames,
            },

            # To compare against a simple LSTM:
            # "use_lstm": True,
            # "lstm_use_prev_action": True,
            # "lstm_use_prev_reward": True,

            # To compare against a simple attention net:
            # "use_attention": True,
            # "attention_use_n_prev_actions": 1,
            # "attention_use_n_prev_rewards": 1,
        },
        "num_sgd_iter": 5,
        "vf_loss_coeff": 0.0001,
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    results = tune.run(
        args.run, config=config, stop=stop, verbose=2, checkpoint_at_end=True)

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    # checkpoints = results.get_trial_checkpoints_paths(
    #     trial=results.get_best_trial("episode_reward_mean", mode="max"),
    #     metric="episode_reward_mean")

    # checkpoint_path = checkpoints[0][0]
    # trainer = PPOTrainer(config)
    # trainer.restore(checkpoint_path)

    # # Inference loop.
    # env = StatelessCartPole()

    # # Run manual inference loop for n episodes.
    # for _ in range(10):
    #     episode_reward = 0.0
    #     reward = 0.0
    #     action = 0
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         # Create a dummy action using the same observation n times,
    #         # as well as dummy prev-n-actions and prev-n-rewards.
    #         action, state, logits = trainer.compute_single_action(
    #             input_dict={
    #                 "obs": obs,
    #                 "prev_n_obs": np.stack([obs for _ in range(num_frames)]),
    #                 "prev_n_actions": np.stack([0 for _ in range(num_frames)]),
    #                 "prev_n_rewards": np.stack(
    #                     [1.0 for _ in range(num_frames)]),
    #             },
    #             full_fetch=True)
    #         obs, reward, done, info = env.step(action)
    #         episode_reward += reward

    #     print(f"Episode reward={episode_reward}")

    ray.shutdown()
