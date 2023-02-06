import gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1)
env = VecFrameStack(env, n_stack=4)


model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
#model.save("dqn_cartpole")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_cartpole")
model.save_replay_buffer("breakout_dqn_replay")
