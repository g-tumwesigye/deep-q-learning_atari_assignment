import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# creating the environment
env_id = "Breakout-v4"
env = Monitor(gym.make(env_id, render_mode='rgb_array'))

# Training a DQN agent using MlpPolicy for comparison
model_mlp = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    exploration_fraction=0.1,
    verbose=1
)

# Training for 50,000 steps
model_mlp.learn(total_timesteps=50000)

# Saving the MLP model separately
import os
os.makedirs("models", exist_ok=True)
model_mlp.save("models/dqn_model_mlp.zip")

env.close()

