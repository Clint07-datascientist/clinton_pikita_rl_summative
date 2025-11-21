import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DroneScoutingEnv

def train_ppo():
    save_dir = "models/ppo"
    os.makedirs(save_dir, exist_ok=True)
    
    hyperparams = [
        {"lr": 3e-4, "n_steps": 2048, "ent_coef": 0.0},
        {"lr": 1e-4, "n_steps": 4096, "ent_coef": 0.01},
        {"lr": 5e-4, "n_steps": 1024, "ent_coef": 0.0},
        {"lr": 3e-4, "n_steps": 2048, "ent_coef": 0.05},
        {"lr": 1e-3, "n_steps": 2048, "ent_coef": 0.0},
        {"lr": 3e-4, "n_steps": 512, "ent_coef": 0.0},
        {"lr": 2e-4, "n_steps": 2048, "clip_range": 0.1},
        {"lr": 3e-4, "n_steps": 2048, "clip_range": 0.3},
        {"lr": 3e-4, "n_steps": 2048, "gae_lambda": 0.9},
        {"lr": 3e-4, "n_steps": 2048, "gae_lambda": 0.99},
    ]

    results = []

    print("--- Starting PPO Training (10 Runs) ---")
    for i, params in enumerate(hyperparams):
        run_name = f"ppo_run_{i}"
        env = Monitor(DroneScoutingEnv(seed=i))
        
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=i,
            learning_rate=params["lr"],
            n_steps=params["n_steps"],
            ent_coef=params.get("ent_coef", 0.0),
            clip_range=params.get("clip_range", 0.2),
            gae_lambda=params.get("gae_lambda", 0.95),
        )
        
        model.learn(total_timesteps=30000)
        
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Run {i}: Reward {mean_r:.2f} +/- {std_r:.2f}")
        
        model.save(os.path.join(save_dir, run_name))
        
        res = params.copy()
        res["run_id"] = i
        res["mean_reward"] = mean_r
        results.append(res)

    pd.DataFrame(results).to_csv(os.path.join(save_dir, "ppo_results.csv"), index=False)
    print("PPO Results Saved.")

if __name__ == "__main__":
    train_ppo()