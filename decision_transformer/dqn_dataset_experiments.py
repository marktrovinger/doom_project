import gym

from stable_baselines3 import DQN

def main(env_name="CartPole-v0"):
    env = gym.make(env_name)

    model = DQN(policy="MlpPolicy", env=env, verbose=1, tensorboard_log=f"runs/DQN/{env_name}")
    model.learn(total_timesteps=50000)
    model.save_replay_buffer(f"dataset/DQN/{env_name}")



if __name__ == "__main__":
    main()