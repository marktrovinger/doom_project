import optuna
import gymnasium
import numpy as np

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import torch.nn as nn

def tune_PPO():
    # this might need to be changed for gymnasium
    env_id = "Pendulum-v1"
    
    eval_envs = make_vec_env()


def main():
    pass

    

if __name__ == "__main__":
    main()

