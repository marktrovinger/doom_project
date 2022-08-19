# imports, don't forget wandb
import numpy as np


from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import WarpFrame

import wandb
from wandb.integration.sb3 import WandbCallback

import gym
import vizdoomgym

config = {
        "policy_type": 'CnnPolicy',
        "env_name": 'VizdoomBasic-v0',
        "total_timesteps": 500000
        }
# main comparison, investigate how many timesteps are needed, probably close
# to 2M, but have to read some papers first


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
    env = gym.make(config['env_name'])
    env = Monitor(env)
    env = WarpFrame(env)
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
env = VecFrameStack(env, 4, "last")

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
# model = DQN(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
# write results to wand for that run

# save model if performance is better
