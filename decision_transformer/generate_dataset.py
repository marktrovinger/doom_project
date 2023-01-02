import numpy as np


from stable_baselines3 import DQN 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import WarpFrame

import wandb
from wandb.integration.sb3 import WandbCallback

import gym
import vizdoomgym.gym_wrapper

config = {
        "policy_type": 'CnnPolicy',
        "env_name": 'VizdoomBasic-v0',
        "total_timesteps": 10000,
        "num_cpu": 6
        }


run = wandb.init(
    project="doom",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
# arg parse
def parse_args():
    pass


def make_env():
    env = make_vec_env(config['env_name'], n_envs=config['num_cpu'], seed=0, vec_env_cls=SubprocVecEnv)
    env = Monitor(env)
    env = WarpFrame(env)
    return env

def main():
    
    env = make_env()
    #env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    env = VecFrameStack(env, 4, "last")


    #model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model = DQN(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    model.save_replay_buffer(f"./dataset/dqn_replay/{config['env_name']}")

    run.finish()

if __name__ == '__main__':
    main()
