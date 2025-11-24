import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # The baselines we want to load from CSVs
    algorithms = ['dqn', 'a2c', 'reinforce', 'ppo']
    data = {}
    
    print("Loading baseline results...")
    
    # 1. Load Baselines from the models folder
    for algo in algorithms:
        path = f"models/{algo}/{algo}_results.csv"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Average of the 10 random runs
                avg_reward = df['mean_reward'].mean()
                data[algo.upper()] = avg_reward
            except Exception as e:
                print(f"Error reading {algo}: {e}")
        else:
            print(f"Warning: CSV not found for {algo}")

    # 2. Add the HERO (Manually added based on your successful observation)
    # This represents the PPO agent trained for 400k steps with the Compass
    data['PPO (HERO)'] = 83.41 

    if not data:
        print("No data found! Make sure you ran the training scripts.")
        return

    # --- Plotting ---
    names = list(data.keys())
    values = list(data.values())
    
    # Define colors: Baselines are Grey, Standard PPO is Blue, Hero is Gold
    colors = []
    for name in names:
        if "HERO" in name:
            colors.append('#FFD700') # Gold
        elif "PPO" in name:
            colors.append('#2196F3') # Blue
        else:
            colors.append('#B0B0B0') # Grey
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(names, values, color=colors)
    
    plt.title('AgroInsightX: Baseline vs. Tuned Hero Performance', fontsize=16)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.xlabel('Agent Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        # Handle negative bars (put label below) vs positive bars (put label above)
        y_pos = height if height > 0 else 0
        va = 'bottom' if height > 0 else 'bottom'
        
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}',
                ha='center', va=va, fontweight='bold')

    # Save to file
    plt.savefig('algorithm_comparison.png')
    print("Graph saved as 'algorithm_comparison.png'.")
    plt.show()

if __name__ == "__main__":
    plot_comparison()