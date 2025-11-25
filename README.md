# AgroInsightX: Autonomous Drone Scouting with Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green)
![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.3.0-orange)

## 🌍 Project Overview
**AgroInsightX** is a mission-based Reinforcement Learning (RL) project designed to solve a real-world precision agriculture challenge. The goal is to train an autonomous drone agent to scout crop fields, detect disease hotspots, and return to base safely while navigating complex environmental constraints.

### 🎯 The Mission
The drone must navigate a **12x12 km grid** to:
1.  **Scan** all designated Crop Waypoints (Blue squares).
2.  **Avoid** static Obstacles (Trees/Buildings) and No-Fly Zones (Red cells).
3.  **Manage** limited Battery life.
4.  **Compensate** for stochastic Wind patterns.
5.  **Return** to Base (Green square) for data upload.

## 🛠️ Project Structure
```
student_name_rl_summative/
├── environment/           # Custom Gymnasium Environment
│   ├── custom_env.py      # Logic: Physics, Rewards, Compass Navigation
│   └── rendering.py       # Visualization: Pygame rendering with Flight Trails
├── training/              # Training Scripts
│   ├── train_hero.py      # The "Hero" PPO trainer (Complex Map)
│   ├── dqn_training.py    # Baseline DQN trainer
│   ├── ppo_training.py    # Baseline PPO trainer
│   ├── a2c_training.py    # Baseline A2C trainer
│   └── reinforce_training.py # Baseline REINFORCE trainer
├── models/                # Saved Models & Logs
│   ├── hero_complex/      # The best performing agent
│   ├── dqn/               # Baseline results
│   ├── ppo/               # Baseline results
│   └── ...
├── main.py                # Main execution script (Visualization)
├── plot_results.py        # Graph generation script
└── requirements.txt       # Dependencies
````

## 🚀 Installation & Setup

1.  **Clone the repository** (or extract the folder).
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage Guide

### 1\. View the "Hero" Agent (Best Performance)

To see the fully trained PPO agent attempting the complex mission (12x12 Grid with Wind + Compass):

```bash
python main.py --mode model --algo ppo --path models/hero_complex/ppo_complex_hero.zip
```

*Watch as the drone (White) follows the compass vectors to scan targets (Blue -\> Cyan) and fights the wind to return to base (Green).*

### 2\. Train the Models

To train the "Hero" model from scratch (takes \~20-30 mins):

```bash
python training/train_hero.py
```

To train the baseline algorithms for comparison:

```bash
python training/dqn_training.py
python training/a2c_training.py
python training/reinforce_training.py
python training/ppo_training.py
```

### 3\. Generate Analysis Graphs

To create the comparison chart (`algorithm_comparison.png`) used in the report:

```bash
python plot_results.py
```

## 📊 Results & Analysis

We compared four algorithms: **PPO**, **DQN**, **A2C**, and **REINFORCE**.

  * **Winner:** **PPO (Proximal Policy Optimization)** proved to be the most robust.
  * **Key Finding:** Simple RL agents struggled with the "blind" search on a large map. By adding a **Compass Vector** to the observation space and shaping the reward for navigation, the PPO agent achieved a **Mission Success rate of \>90%** in final testing.

| Agent | Average Reward | Status |
| :--- | :--- | :--- |
| **PPO (Hero)** | **\~83.41** | **Solved** |
| PPO (Baseline) | -2.10 | Failed |
| A2C | -3.50 | Failed |
| DQN | -4.50 | Failed |
| REINFORCE | -35.0 | Unstable |

## 🎥 Visualization Features

The custom `DroneRenderer` includes:

  * **Flight Trails:** Blue lines showing the agent's path history.
  * **HUD:** Real-time display of Battery, Altitude, Wind Vector, and Waypoint status.
  * **Dynamic Elements:** Visual indicators for scanning (color change) and altitude (ring size).

-----

*Submitted as part of the Reinforcement Learning Summative Assignment.*

```
```
