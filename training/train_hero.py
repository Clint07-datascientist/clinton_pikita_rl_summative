import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DroneScoutingEnv

def train_complex_hero():
    save_dir = "models/hero_complex"
    os.makedirs(save_dir, exist_ok=True)
    
    print("--- Starting HERO Training (12x12 Complex Map with Compass) ---")
    print("Training for 400,000 steps. This may take ~20 mins.")
    
    env = Monitor(DroneScoutingEnv(seed=777)) 
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        seed=777,
        learning_rate=3e-4, # Standard rate
        n_steps=2048, 
        ent_coef=0.01,
        clip_range=0.2,
    )
    
    # 400k steps for the complex map
    model.learn(total_timesteps=400000)
    
    model.save(os.path.join(save_dir, "ppo_complex_hero"))
    print("Complex Hero Model Saved!")

if __name__ == "__main__":
    train_complex_hero()