# results/visualize_comparison.py
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")


def parse_log_file(log_path):

    pattern = re.compile(
        r"(?P<timestamp>[\d\-:\s,]+) - Episode: (?P<episode>\d+), Reward: (?P<reward>-?[\d\.]+), Epsilon: (?P<epsilon>[\d\.]+)"
    )
    data = []
    with open(log_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                entry = match.groupdict()
                data.append({
                    'episode': int(entry['episode']),
                    'reward': float(entry['reward']),
                    'epsilon': float(entry['epsilon'])
                })
    return pd.DataFrame(data)


def aggregate_data(weights_dir, env_name):
    all_data = []
    for algo in os.listdir(weights_dir):
        algo_path = os.path.join(weights_dir, algo)
        if os.path.isdir(algo_path):
            env_dir = os.path.join(algo_path, env_name)
            if os.path.isdir(env_dir):
                log_path = os.path.join(env_dir, 'training.log')
                if os.path.exists(log_path):
                    df = parse_log_file(log_path)
                    print("Hello Parser === ", df)
                    df['algorithm'] = algo
                    df['environment'] = env_name
                    all_data.append(df)
                else:
                    print(f"Warning: No training.log found at {log_path}")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No data found to aggregate.")
        return pd.DataFrame()


def compute_metrics(df, window=50):
    df_sorted = df.sort_values(['algorithm', 'episode'])
    df_sorted['reward_ma'] = df_sorted.groupby('algorithm')['reward'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    df_sorted['reward_std'] = df_sorted.groupby('algorithm')['reward'].transform(
        lambda x: x.rolling(window, min_periods=1).std())
    return df_sorted

# env_name = 'MountainCar-v0' # manually need to change for each environment
# env_name = 'LunarLander-v3' # manually need to change for each environment
env_name = 'CartPole-v1' # manually need to change for each environment


def plot_rewards(df, save_path=None):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='episode', y='reward_ma', hue='algorithm', ci='sd')
    plt.title('Comparison of Reward Moving Average Across Algorithms in Env: ' + env_name)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Moving Average)')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Reward comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_epsilon(df, save_path=None):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='episode', y='epsilon', hue='algorithm', ci=None)
    plt.title('Epsilon Decay Across Algorithms in Env: ' + env_name)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Epsilon decay plot saved to {save_path}")
    else:
        plt.show()
    plt.close()



def plot_convergence(df, threshold=200, save_path=None):
    # Automatically determine appropriate threshold based on environment and data
    if env_name == 'MountainCar-v0':
        # For MountainCar-v0, use 90th percentile of rewards as threshold
        threshold = df['reward'].quantile(0.90)
    elif env_name == 'CartPole-v1':
        threshold = 200  # Standard threshold for CartPole
    elif env_name == 'LunarLander-v2':
        threshold = 200  # Standard threshold for LunarLander
    else:
        # For unknown environments, use 75th percentile of rewards
        threshold = df['reward'].quantile(0.75)
    
    # Add debug information
    print(f"\nConvergence Analysis for {env_name}")
    print(f"Automatically determined threshold: {threshold:.1f}")
    print(f"Reward range: {df['reward'].min():.1f} to {df['reward'].max():.1f}")
    
    # Find episodes where each algorithm first reaches threshold
    convergence = df[df['reward'] >= threshold].groupby('algorithm')['episode'].min().reset_index()
    
    # Print convergence information
    print("\nConvergence summary:")
    for _, row in convergence.iterrows():
        print(f"{row['algorithm']}: First reached threshold at episode {row['episode']}")
    
    plt.figure(figsize=(10, 6))
    
    if not convergence.empty:
        # Create main convergence plot
        sns.barplot(data=convergence, x='algorithm', y='episode')
        plt.title(f'Episodes to Reach Reward ≥ {threshold:.1f}\nin {env_name}')
        
        # Add value labels on top of each bar
        for i, v in enumerate(convergence['episode']):
            plt.text(i, v, f'{v}', ha='center', va='bottom')
    else:
        # Alternative visualization if no algorithm reached threshold
        best_rewards = df.groupby('algorithm')['reward'].max().reset_index()
        sns.barplot(data=best_rewards, x='algorithm', y='reward')
        plt.title(f'Best Achieved Rewards\n(Threshold: {threshold:.1f}) in {env_name}')
        
        # Add value labels on top of each bar
        for i, v in enumerate(best_rewards['reward']):
            plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Episodes' if not convergence.empty else 'Best Reward')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()    


def plot_stability(df, window=50, save_path=None):
    plt.figure(figsize=(12, 8))
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('episode')
        algo_df = algo_df.copy()
        algo_df['reward_ma'] = algo_df['reward'].rolling(
            window, min_periods=1).mean()
        algo_df['reward_std'] = algo_df['reward'].rolling(
            window, min_periods=1).std()
        plt.plot(algo_df['episode'], algo_df['reward_std'], label=algo)
    plt.title(f'Stability of Rewards Over Episodes (Window Size = {window}) in Env: ' + env_name)
    plt.xlabel('Episode')
    plt.ylabel('Reward Standard Deviation')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Stability plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_combined_metrics(df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)

    # Plot Reward Moving Average
    sns.lineplot(data=df, x='episode', y='reward_ma',
                 hue='algorithm', ax=axes[0, 0], ci='sd')
    axes[0, 0].set_title(f'Reward Moving Average - {env_name}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (Moving Average)')

    # Plot Epsilon Decay
    sns.lineplot(data=df, x='episode', y='epsilon',
                 hue='algorithm', ax=axes[0, 1], ci=None)
    axes[0, 1].set_title(f'Epsilon Decay - {env_name}')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')

    # Plot Convergence Speed with environment-specific threshold
    if env_name == 'MountainCar-v0':
        threshold = -110  # Adjusted threshold for MountainCar
    else:
        threshold = 200   # Default threshold for other environments

    convergence = df[df['reward'] >= threshold].groupby(
        'algorithm')['episode'].min().reset_index()
    
    if not convergence.empty:
        # If algorithms reached threshold, show convergence speed
        sns.barplot(data=convergence, x='algorithm', y='episode', ax=axes[1, 0])
        axes[1, 0].set_title(
            f'Convergence Speed (Episode to Reach Reward ≥ {threshold}) - {env_name}')
        
        # Add value labels on top of each bar
        for i, v in enumerate(convergence['episode']):
            axes[1, 0].text(i, v, f'{v}', ha='center', va='bottom')
    else:
        # If no algorithm reached threshold, show best achieved rewards
        best_rewards = df.groupby('algorithm')['reward'].max().reset_index()
        sns.barplot(data=best_rewards, x='algorithm', y='reward', ax=axes[1, 0])
        axes[1, 0].set_title(
            f'Best Achieved Rewards (Threshold: {threshold}) - {env_name}')
        
        # Add value labels on top of each bar
        for i, v in enumerate(best_rewards['reward']):
            axes[1, 0].text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    axes[1, 0].set_xlabel('Algorithm')
    axes[1, 0].set_ylabel('Episode' if not convergence.empty else 'Best Reward')
    
    # Rotate x-axis labels if needed
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot Stability with environment-specific adjustments
    window_size = 50
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('episode')
        algo_df = algo_df.copy()
        algo_df['reward_ma'] = algo_df['reward'].rolling(
            window_size, min_periods=1).mean()
        algo_df['reward_std'] = algo_df['reward'].rolling(
            window_size, min_periods=1).std()
        axes[1, 1].plot(algo_df['episode'], algo_df['reward_std'], label=algo)
    
    axes[1, 1].set_title(
        f'Stability of Rewards (Window Size = {window_size}) - {env_name}')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward Standard Deviation')
    axes[1, 1].legend(title='Algorithm')

    # Add overall title
    plt.suptitle(f'Performance Metrics for {env_name}', 
                 fontsize=16, y=1.02)

    # Add debug information
    print(f"\nPlotting combined metrics for {env_name}")
    print(f"Reward range: {df['reward'].min():.1f} to {df['reward'].max():.1f}")
    print(f"Using threshold: {threshold}")
    if not convergence.empty:
        print("\nConvergence episodes by algorithm:")
        for _, row in convergence.iterrows():
            print(f"{row['algorithm']}: Episode {row['episode']}")
    else:
        print("\nBest rewards by algorithm:")
        for _, row in best_rewards.iterrows():
            print(f"{row['algorithm']}: {row['reward']:.1f}")

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Combined metrics plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    # Path to the weights directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, 'weights')

    # Step 1: Aggregate data from all training logs
    df = aggregate_data(weights_dir, env_name)

    if df.empty:
        print("No data available for plotting.")
        return

    # Step 2: Compute moving averages and standard deviations
    df = compute_metrics(df, window=50)

    # Step 3: Generate and save plots
    plot_rewards(df, save_path=os.path.join(current_dir,"plots_images",
                 env_name+"_ENV", 'comparison_rewards.png'))
    plot_epsilon(df, save_path=os.path.join(current_dir,"plots_images",
                 env_name+"_ENV", 'comparison_epsilon_decay.png'))
    plot_convergence(df, threshold=200, save_path=os.path.join(current_dir,
        "plots_images", env_name+"_ENV", 'convergence_speed.png'))
    plot_stability(df, window=50, save_path=os.path.join(current_dir,
        "plots_images", env_name+"_ENV", 'stability_rewards.png'))
    plot_combined_metrics(df, save_path=os.path.join(current_dir,
        "plots_images", env_name+"_ENV", 'combined_metrics.png'))

    print("All plots have been generated and saved successfully.")


if __name__ == "__main__":
    main()
