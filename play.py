from stable_baselines3 import DQN
import gymnasium as gym
import time
import numpy as np
import sys
import subprocess

def install_atari():
    """Install required Atari packages"""
    try:
        import ale_py
    except ImportError:
        print("Installing Atari dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "gymnasium[atari]", "gymnasium[accept-rom-license]", "ale-py"
        ])
        print("Please restart the script after installation.")
        exit()

def play_model(model_path="models/dqn_model_cnn.zip"):
    install_atari()  # Ensure packages are installed
    
    try:
        # Try modern ALE first
        env = gym.make("ALE/Breakout-v5", render_mode="human")
    except:
        # Fallback to classic version
        env = gym.make("Breakout-v4", render_mode="human")
    
    model = DQN.load(model_path, env=env)
    
    episode_rewards = []
    
    for episode in range(3):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        # Launch ball
        obs, _, _, _, _ = env.step(1)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Auto-relaunch if life lost
            if info.get('lives', 5) < 5:
                obs, _, _, _, _ = env.step(1)
            
            env.render()
            time.sleep(0.02)
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1} - Score: {total_reward}")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    rewards = play_model()
    print("\nPerformance Summary:")
    print(f"Average Score: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")