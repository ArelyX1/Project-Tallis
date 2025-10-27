import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl

if __name__ == "__main__":
    # Suppress GPU warning
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    
    ray.init()

    env_name = "4x4grid"
    
    # Rutas absolutas: sube un nivel desde experiments/ a sumo-rl/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    net_file = os.path.join(parent_dir, "sumo_rl/nets/4x4-Lucas/4x4.net.xml")
    route_file = os.path.join(parent_dir, "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml")

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=net_file,
                route_file=route_file,
                out_csv_name="outputs/4x4grid/ppo",
                use_gui=False,
                num_seconds=80000,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .env_runners(num_env_runners=2, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            minibatch_size=64,
            num_epochs=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        storage_path=os.path.expanduser("~/ray_results/" + env_name),
        config=config.to_dict(),
    )
