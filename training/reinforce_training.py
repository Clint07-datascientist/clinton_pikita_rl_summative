import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DroneScoutingEnv

# Define Policy Network (PyTorch)
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def train_reinforce():
    save_dir = "models/reinforce"
    os.makedirs(save_dir, exist_ok=True)

    hyperparams = [
        {"lr": 1e-3, "gamma": 0.99, "hidden": 128},
        {"lr": 5e-4, "gamma": 0.99, "hidden": 256},
        {"lr": 1e-2, "gamma": 0.99, "hidden": 64},
        {"lr": 1e-3, "gamma": 0.95, "hidden": 128},
        {"lr": 1e-3, "gamma": 0.90, "hidden": 128},
        {"lr": 5e-4, "gamma": 0.995, "hidden": 128},
        {"lr": 3e-4, "gamma": 0.99, "hidden": 128},
        {"lr": 1e-3, "gamma": 0.99, "hidden": 64},
        {"lr": 2e-3, "gamma": 0.99, "hidden": 128},
        {"lr": 1e-3, "gamma": 0.99, "hidden": 128},
    ]
    
    results = []

    print("--- Starting REINFORCE Training (10 Runs) ---")
    for run_id, params in enumerate(hyperparams):
        env = DroneScoutingEnv(seed=run_id)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        policy = PolicyNet(obs_dim, act_dim, hidden=params["hidden"])
        optimizer = optim.Adam(policy.parameters(), lr=params["lr"])
        
        episodes = 1000 # Number of trajectory collections
        all_rewards = []
        
        for ep in range(episodes):
            obs, _ = env.reset(seed=run_id+ep)
            log_probs = []
            rewards = []
            terminated = False
            truncated = False
            
            # 1. Collect Trajectory
            while not (terminated or truncated):
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                probs = policy(obs_t)
                
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                
                obs, r, terminated, truncated, _ = env.step(action.item())
                
                log_probs.append(m.log_prob(action))
                rewards.append(r)
            
            total_r = sum(rewards)
            all_rewards.append(total_r)
            
            # 2. Calculate Discounted Returns (Monte Carlo)
            R = 0
            returns = []
            for r in rewards[::-1]:
                R = r + params["gamma"] * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            
            # Normalize Returns (Baseline stability trick)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # 3. Policy Gradient Update
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
            
        # Mean of last 50 episodes as result
        mean_final = np.mean(all_rewards[-50:])
        print(f"Run {run_id}: Reward {mean_final:.2f}")
        
        # Save PyTorch Model
        torch.save(policy.state_dict(), os.path.join(save_dir, f"reinforce_run_{run_id}.pth"))
        
        res = params.copy()
        res["run_id"] = run_id
        res["mean_reward"] = mean_final
        results.append(res)

    pd.DataFrame(results).to_csv(os.path.join(save_dir, "reinforce_results.csv"), index=False)
    print("REINFORCE Results Saved.")

if __name__ == "__main__":
    train_reinforce()