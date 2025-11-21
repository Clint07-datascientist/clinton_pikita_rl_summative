import os
import sys
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DroneScoutingEnv

def train_a2c():
    save_dir = "models/a2c"
    os.makedirs(save_dir, exist_ok=True)
    
    hyperparams = [
        {"lr": 7e-4, "n_steps": 5, "ent_coef": 0.0},
        {"lr": 1e-3, "n_steps": 5, "ent_coef": 0.01},
        {"lr": 5e-4, "n_steps": 10, "ent_coef": 0.0},
        {"lr": 7e-4, "n_steps": 5, "ent_coef": 0.05},
        {"lr": 1e-4, "n_steps": 20, "ent_coef": 0.0},
        {"lr": 7e-4, "n_steps": 5, "gamma": 0.90},
        {"lr": 7e-4, "n_steps": 5, "gamma": 0.995},
        {"lr": 3e-4, "n_steps": 5, "ent_coef": 0.0},
        {"lr": 7e-4, "n_steps": 30, "ent_coef": 0.0},
        {"lr": 1e-3, "n_steps": 5, "rms_prop_eps": 1e-4},
    ]

    results = []

    print("--- Starting A2C Training (10 Runs) ---")
    for i, params in enumerate(hyperparams):
        run_name = f"a2c_run_{i}"
        env = Monitor(DroneScoutingEnv(seed=i))
        
        model = A2C(
            "MlpPolicy", env, verbose=0, seed=i,
            learning_rate=params["lr"],
            n_steps=params["n_steps"],
            ent_coef=params.get("ent_coef", 0.0),
            gamma=params.get("gamma", 0.99),
            rms_prop_eps=params.get("rms_prop_eps", 1e-5)
        )
        
        model.learn(total_timesteps=30000)
        
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Run {i}: Reward {mean_r:.2f} +/- {std_r:.2f}")
        
        model.save(os.path.join(save_dir, run_name))
        
        res = params.copy()
        res["run_id"] = i
        res["mean_reward"] = mean_r
        results.append(res)

    pd.DataFrame(results).to_csv(os.path.join(save_dir, "a2c_results.csv"), index=False)
    print("A2C Results Saved.")

if __name__ == "__main__":
    train_a2c()