import argparse
import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from environment.custom_env import DroneScoutingEnv

# Re-import REINFORCE net structure for loading weights
class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super(PolicyNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, act_dim),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

def run_random(episodes=3):
    print("Running Random Agent Mode...")
    env = DroneScoutingEnv(render_mode="human", seed=42)
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            time.sleep(0.1)
            if term or trunc:
                done = True
                time.sleep(1)
    env.close()

def run_model(model_path, algo, episodes=3):
    print(f"Loading {algo} model from {model_path}...")
    env = DroneScoutingEnv(render_mode="human", seed=100)
    
    model = None
    policy_net = None
    
    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "dqn":
        model = DQN.load(model_path)
    elif algo == "a2c":
        model = A2C.load(model_path)
    elif algo == "reinforce":
        # Initialize net structure (assuming default hidden=128 used in training)
        policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n)
        policy_net.load_state_dict(torch.load(model_path))
        policy_net.eval()

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep+500) # New seed for testing
        done = False
        total_reward = 0
        
        print(f"Episode {ep+1} Start")
        
        while not done:
            if algo == "reinforce":
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    probs = policy_net(obs_t)
                    action = torch.argmax(probs).item()
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            env.render()
            
            # Throttle speed for viewing
            time.sleep(0.1)
            
            if term or trunc:
                outcome = info.get('outcome', 'timeout')
                print(f"Episode End. Reason: {outcome}. Reward: {total_reward:.2f}")
                done = True
                time.sleep(1)
                
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "model"], default="random")
    parser.add_argument("--algo", choices=["ppo", "dqn", "a2c", "reinforce"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to model file (e.g., models/ppo/ppo_run_0.zip)")
    args = parser.parse_args()

    if args.mode == "random":
        run_random()
    else:
        if not args.path:
            print("Error: Please provide --path to model file for model mode.")
        else:
            run_model(args.path, args.algo)