# AgroInsightX: Autonomous Drone Scouting with Reinforcement Learning
---

## Project Overview

AgroInsightX is a precision-agriculture Reinforcement Learning (RL) project designed to solve a real-world food security challenge in Africa. In this project the goal is to train an autonomous drone agent to scout crop fields, detect disease hotspots, and return to base safely while navigating complex environmental constraints. To train a reinforcement learning agent by comparing RL Methods, that is, Value-Based (Deep Q Networks) and Policy Methods (REINFORCE, PPO, A2C), to optimize for a simulated mission-based environment.
---

## The Goal

The drone must navigate a 12x12 km grid to:

1. Scan all designated Crop Waypoints (Blue squares).
2. Avoid static Obstacles (Trees/Buildings) and No-Fly Zones (Red cells).
3. Manage limited Battery life.
4. Compensate for stochastic Wind patterns.
5. Return to Base (Green square) for data upload.
--- 

## Project Structure

`
clinton_pikita_rl_summative/
├── environment/                  # Custom Gymnasium Environment
│   ├── custom_env.py             # Logic: Physics, Rewards, Observations
│   └── rendering.py              # Visualization: Pygame rendering engine
├── training/                     # Training Scripts
│   ├── train_hero.py             # The "Hero" PPO trainer (Complex Map)
│   ├── dqn_training.py           # Baseline DQN trainer
│   ├── ppo_training.py           # Logic: Physics, Rewards, Observations
│   ├── a2c_training.py           # Logic: Physics, Rewards, Observations
│   ├── reinforce_training.py     # Logic: Physics, Rewards, Observations
│   └── ...                       # Other baselines (A2C, REINFORCE)
├── models/                       # Saved Models & Logs
│   ├── hero_complex/             # The best performing agent
│   └── ...                       # Baseline results
├── main.py                       # Main execution script (Visualization)
├── plot_results.py               # Graph generation script
└── requirements.txt              # Dependencies
`

---

## Installation & Setup
1. Clone the repositosry
2. Install dependencies

