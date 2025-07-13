from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os

# Creating the environment
env_id = "Breakout-v4"
env = Monitor(gym.make(env_id, render_mode='rgb_array'))

# Defining the model using a Convolutional Policy (CNN)
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=0.0001,
    gamma=0.99,
    batch_size=32,
    buffer_size=100_000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    exploration_fraction=0.1,
    verbose=1,
    tensorboard_log="./dqn_tensorboard/"
)

# Training the agent for 50,000 steps
model.learn(total_timesteps=50000)

# Saving the model
os.makedirs("models", exist_ok=True)
model.save("models/dqn_model.zip")

env.close()

