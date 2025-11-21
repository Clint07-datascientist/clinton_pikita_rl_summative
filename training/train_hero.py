import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Ensure we can import environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DroneScoutingEnv

def train_hero_model():
    # We will save this in a special folder
    save_dir = "models/hero_ppo"
    os.makedirs(save_dir, exist_ok=True)
    
    print("--- Starting HERO Training (Long Run) ---")
    print("This will train for 500,000 timesteps. Please be patient!")
    
    # 1. Setup Environment
    env = Monitor(DroneScoutingEnv(seed=123)) # Fixed seed for reproducibility
    
    # 2. Setup Model with "Run 2" settings (Best performer)
    # Run 2 params were: lr=5e-4, n_steps=1024, ent_coef=0.0
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        seed=123,
        learning_rate=5e-4,
        n_steps=1024,
        ent_coef=0.0,
        clip_range=0.2,
        gae_lambda=0.95,
    )
    
    # 3. Train for much longer (30k -> 500k)
    # This ensures it finds the sparse rewards (waypoints)
    model.learn(total_timesteps=500000)
    
    # 4. Save
    model.save(os.path.join(save_dir, "ppo_hero"))
    print("Hero Model Saved! You can now run the visualization.")

if __name__ == "__main__":
    train_hero_model()