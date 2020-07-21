from deepqnetwork import DeepQNetwork
import os
import torch as T
from replay_memory import ReplayBuffer
import numpy as np


class DQNAgent():
    def __init__(self, input_dims, n_actions, lr, mem_size, batch_size, replace=1000, algo=None, env_name=None, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01, chkpoint_dir=os.getcwd()):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.batch_size = batch_size
        self.replace_target_count = replace
        self.env_name = env_name
        self.chkpoint_dir = chkpoint_dir
        self.algo = algo
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DeepQNetwork(lr = self.lr, n_actions = self.n_actions, chkpt_dir=self.chkpoint_dir, name=self.env_name+'_'+self.algo+'_q_eval')

        self.q_next = DeepQNetwork(lr = self.lr, n_actions = self.n_actions, chkpt_dir=self.chkpoint_dir, name=self.env_name+'_'+self.algo+'_q_next')

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transistion(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, rewards, dones, actions, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad() 

        self.replace_target_network()

        states, rewards, dones, actions, states_ = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()