from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

class ReplayBufferCallback(BaseCallback):
    def __init__(self, training_timesteps, buffer_size=int(1e6), sampling_rate = 0.02, verbose: int = 0):
        super().__init__(verbose)
        self.rb = None
        self.training_timesteps = training_timesteps
        self.timesteps = self.training_timesteps * sampling_rate #how often are we sampling?
        self.buffer_size = buffer_size

    def _on_training_start(self) -> None:
        self.rb = ReplayBuffer(self.buffer_size, self.training_env.observation_space, self.training_env.action_space)
        return super()._on_training_start()

    def _on_step(self) -> bool:
        # check what we can see
        if self.num_timesteps > 1 and self.num_timesteps == self.timesteps:
            obs = self.model._last_obs
            next_obs = np.zeros(shape=self.model.env.observation_space.shape)
            act = self.training_env.unwrapped.actions
            rew = self.training_env.unwrapped.buf_rews
            done = self.training_env.unwrapped.buf_dones
            infos= self.training_env.unwrapped.buf_infos
            self.rb.add(obs, next_obs, act, rew, done, infos)
        if self.num_timesteps == self.training_timesteps:
            obs = self.rb.observations
            act = self.rb.actions
            rew = self.rb.rewards
            done = self.rb.dones
            np.savez_compressed("testing.npz", obs=obs, act=act, rew=rew, done=done)
        #    self.rb.add()
        return super()._on_step()