import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    algorithms = ['ppo', 'dqn', 'a2c', 'reinforce']
    data = {}
    
    for algo in algorithms:
        path = f"models/{algo}/{algo}_results.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            avg_reward = df['mean_reward'].mean()
            data[algo.upper()] = avg_reward

    if not data:
        print("No results to plot!")
        return

    names = list(data.keys())
    values = list(data.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    
    plt.title('AgroInsightX: Algorithm Performance Comparison', fontsize=16)
    plt.ylabel('Average Reward (Mean of 10 Runs)', fontsize=12)
    plt.xlabel('RL Algorithm', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.savefig('algorithm_comparison.png')
    print("Graph saved as algorithm_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_comparison()