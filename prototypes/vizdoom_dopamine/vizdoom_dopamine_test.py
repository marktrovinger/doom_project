import numpy as np
import os
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
from gym.envs.registration import register

BASE_PATH = 'colab_dopamine_run'  # @param

# @title Load the configuration for DQN.

DQN_PATH = os.path.join(BASE_PATH, 'dqn')
# Modified from dopamine/agents/dqn/config/dqn_cartpole.gin
dqn_config = """
# Copied from Dopamine's dqn_nature gin.
# Hyperparameters used in Mnih et al. (2015).
import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables

DQNAgent.gamma = 0.99
DQNAgent.update_horizon = 1
DQNAgent.min_replay_history = 20000  # agent steps
DQNAgent.update_period = 4
DQNAgent.target_update_period = 10000  # agent steps
DQNAgent.epsilon_train = 0.1
DQNAgent.epsilon_eval = 0.05
DQNAgent.epsilon_decay_period = 10000  # agent steps
DQNAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
DQNAgent.optimizer = @tf.train.RMSPropOptimizer()

tf.train.RMSPropOptimizer.learning_rate = 0.00025
tf.train.RMSPropOptimizer.decay = 0.95
tf.train.RMSPropOptimizer.momentum = 0.0
tf.train.RMSPropOptimizer.epsilon = 0.00001
tf.train.RMSPropOptimizer.centered = True

atari_lib.create_atari_environment.game_name = 'VizdoomHealthGatheringSupreme'
#atari.create_atari_environment.sticky_actions = False
create_agent.agent_name = 'dqn'
Runner.num_iterations = 1
Runner.training_steps = 50000  # agent steps
Runner.evaluation_steps = 12500  # agent steps
Runner.max_steps_per_episode = 7000  # agent steps

AtariPreprocessing.terminal_on_life_loss = True

WrappedReplayBuffer.replay_capacity = 100000
WrappedReplayBuffer.batch_size = 32
"""
gin.parse_config(dqn_config, skip_unknown=False)

# try a manual registration for gym...
try:
    register(
        id="VizdoomBasic-v0",
        entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_file": "basic.cfg"}
    )

    register(
        id="VizdoomHealthGatheringSupreme-v0",
        entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_file": "health_gathering_supreme.cfg"}
    )
except:
    print(f"Registration failed for Vizdoom envs")
    pass

# @title Train DQN on Cartpole
dqn_runner = run_experiment.create_runner(DQN_PATH, schedule='continuous_train')
print('Will train DQN agent, please be patient, may be a while...')
dqn_runner.run_experiment()
print('Done training!')