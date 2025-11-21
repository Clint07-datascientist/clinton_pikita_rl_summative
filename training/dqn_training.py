import os
import sys
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Ensure we can import environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DroneScoutingEnv

def train_dqn():
    save_dir = "models/dqn"
    os.makedirs(save_dir, exist_ok=True)
    
    # Hyperparameters to tune
    hyperparams = [
        {"lr": 1e-3, "batch_size": 64, "gamma": 0.99, "expl": 0.2},
        {"lr": 5e-4, "batch_size": 128, "gamma": 0.99, "expl": 0.3},
        {"lr": 1e-4, "batch_size": 32, "gamma": 0.95, "expl": 0.1},
        {"lr": 1e-3, "batch_size": 64, "gamma": 0.90, "expl": 0.5}, 
        {"lr": 2e-3, "batch_size": 256, "gamma": 0.99, "expl": 0.2},
        {"lr": 1e-3, "batch_size": 64, "gamma": 0.99, "expl": 0.1},
        {"lr": 5e-4, "batch_size": 64, "gamma": 0.995, "expl": 0.2},
        {"lr": 1e-3, "batch_size": 64, "gamma": 0.99, "expl": 0.2, "tau": 0.5},
        {"lr": 1e-4, "batch_size": 128, "gamma": 0.99, "expl": 0.2},
        {"lr": 1e-3, "batch_size": 64, "gamma": 0.8, "expl": 0.3},
    ]

    results = []

    print("--- Starting DQN Training (10 Runs) ---")
    for i, params in enumerate(hyperparams):
        run_name = f"dqn_run_{i}"
        
        # Wrap env for logging
        env = Monitor(DroneScoutingEnv(seed=i))
        
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=0,
            learning_rate=params["lr"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            exploration_fraction=params["expl"],
            tau=params.get("tau", 1.0),
            seed=i
        )
        
        # Train
        model.learn(total_timesteps=30000)
        
        # Evaluate
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Run {i}: Reward {mean_r:.2f} +/- {std_r:.2f}")
        
        # Save
        model.save(os.path.join(save_dir, run_name))
        
        # Log
        res = params.copy()
        res["run_id"] = i
        res["mean_reward"] = mean_r
        res["std_reward"] = std_r
        results.append(res)

    # Save CSV
    pd.DataFrame(results).to_csv(os.path.join(save_dir, "dqn_results.csv"), index=False)
    print("DQN Results Saved.")

if __name__ == "__main__":
    train_dqn()