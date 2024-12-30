# results/compare_checkpoints.py
import os
import sys
import gymnasium as gym 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set(style="darkgrid")

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import your models
from models import DQNNetwork, DoubleDQNNetwork, DuelingDQNNetwork

# Manually select environment
# env_name = 'MountainCar-v0'
# env_name = 'CartPole-v1'
env_name = 'LunarLander-v3'  # Use v3 instead of v2

# At the top of your file, after imports
ENV_MAPPING = {
    'LunarLander-v3': 'LunarLander-v3',  # Use v3 instead of v2
    'MountainCar-v0': 'MountainCar-v0',
    'CartPole-v1': 'CartPole-v1'
}

folder_name = ENV_MAPPING[env_name]

def print_versions():
    """Print version information for relevant packages."""
    print("\nPackage versions:")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Gymnasium: {gym.__version__}")
    print("\n")

# Add this right after imports
print_versions()


def evaluate_checkpoint(model_path, env_name, algorithm, num_episodes=100):
    """Evaluate a single checkpoint."""
    try:
        # Create environment using gymnasium with v3
        env = gym.make(env_name, render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create and load model
        if algorithm == 'DQN':
            model = DQNNetwork(state_dim, action_dim)
        elif algorithm == 'DoubleDQN':
            model = DoubleDQNNetwork(state_dim, action_dim)
        elif algorithm == 'DuelingDQN':
            model = DuelingDQNNetwork(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Load model with weights_only=True
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model.eval()

        # Evaluate
        rewards = []
        for episode in range(num_episodes):
            try:
                state, _ = env.reset()
                total_reward = 0
                terminated = truncated = False

                while not (terminated or truncated):
                    if state is None or (isinstance(state, np.ndarray) and state.size == 0):
                        break

                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action = model(state_tensor).max(1)[1].item()

                    try:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward
                        state = next_state
                    except Exception as e:
                        print(f"Error in step: {e}")
                        break

                rewards.append(total_reward)
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue

        env.close()

        if rewards:
            return np.mean(rewards), np.std(rewards)
        else:
            return None, None

    except Exception as e:
        print(f"Error evaluating {model_path}: {str(e)}")
        return None, None

def collect_checkpoint_data(weights_dir):
    """Collect performance data for all checkpoints."""
    data = []
    checkpoints = ['1000.pth', '2000.pth', 'best.pth']
    
    # Use folder_name instead of env_name for directory navigation
    print(f"Looking for checkpoints in folders named: {folder_name}")
    print(f"Using gym environment: {env_name}")

    for algo in os.listdir(weights_dir):
        algo_path = os.path.join(weights_dir, algo)
        if os.path.isdir(algo_path):
            # Use folder_name here instead of env_name
            env_path = os.path.join(algo_path, folder_name)
            if os.path.isdir(env_path):
                print(f"\nEvaluating {algo} checkpoints for {folder_name}")

                for checkpoint in checkpoints:
                    model_path = os.path.join(env_path, checkpoint)
                    if os.path.exists(model_path):
                        print(f"Evaluating {checkpoint}...")
                        # Pass env_name here for gym environment creation
                        mean_reward, std_reward = evaluate_checkpoint(model_path, env_name, algo)

                        if mean_reward is not None and std_reward is not None:
                            data.append({
                                'algorithm': algo,
                                'checkpoint': checkpoint,
                                'mean_reward': mean_reward,
                                'std_reward': std_reward
                            })
                        else:
                            print(f"Warning: Could not evaluate {algo} {checkpoint}")
                    else:
                        print(f"Checkpoint {checkpoint} not found in {env_path}")

    if not data:
        print("\nNo valid checkpoint data collected!")
        print(f"Checked directory: {weights_dir}")
        print(f"Looking for folders named: {folder_name}")
        print(f"Expected structure: {weights_dir}/<algorithm>/{folder_name}/<checkpoint>.pth")
        return pd.DataFrame()

    return pd.DataFrame(data)

def plot_checkpoint_comparison(df, save_path=None):
    """Create bar plot comparing checkpoint performances."""
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar plot
    sns.barplot(data=df, x='algorithm', y='mean_reward', hue='checkpoint', ci=None)
    
    # Add error bars manually
    for i, row in df.iterrows():
        plt.errorbar(
            x=i % len(df['algorithm'].unique()), 
            y=row['mean_reward'], 
            yerr=row['std_reward'], 
            fmt='none', 
            c='black', 
            capsize=5
        )
    
    plt.title(f'Checkpoint Performance Comparison - {env_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Mean Reward')
    plt.legend(title='Checkpoint')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Checkpoint comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_learning_progress(df, save_path=None):
    """Create line plot showing learning progress across checkpoints."""
    plt.figure(figsize=(12, 6))
    
    # Convert checkpoint names to numeric values for plotting
    checkpoint_map = {'1000.pth': 1000, '2000.pth': 2000, 'best.pth': 2500}  # Adjust 'best.pth' as needed
    df['checkpoint_num'] = df['checkpoint'].map(checkpoint_map)
    
    # Plot learning curves
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        plt.errorbar(
            algo_data['checkpoint_num'], 
            algo_data['mean_reward'],
            yerr=algo_data['std_reward'], 
            label=algo, 
            marker='o'
        )
    
    plt.title(f'Learning Progress Across Checkpoints - {env_name}')
    plt.xlabel('Training Episodes')
    plt.ylabel('Mean Reward')
    plt.legend(title='Algorithm')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Learning progress plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def create_summary_table(df, save_path=None):
    """Create and save summary table."""
    # Format the results nicely
    summary = df.copy()
    summary['performance'] = summary.apply(
        lambda x: f"{x['mean_reward']:.2f} ± {x['std_reward']:.2f}", axis=1)
    
    # Pivot table for better visualization
    summary_table = summary.pivot(index='algorithm', 
                                  columns='checkpoint', 
                                  values='performance')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_table.to_csv(save_path)
        print(f"Summary table saved to {save_path}")
    
    return summary_table

def main():
    # Set up paths
    weights_dir = os.path.join(current_dir, 'weights')
    plots_dir = os.path.join(current_dir, 'plots_images', f"{folder_name}_ENV")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Collecting checkpoint data...")
    print(f"Gym environment: {env_name}")
    print(f"Folder name: {folder_name}")
    print(f"Looking in directory: {weights_dir}")

    # Verify directory exists
    if not os.path.exists(weights_dir):
        print(f"\nError: Weights directory not found: {weights_dir}")
        return

    # Print available directories for debugging
    print("\nAvailable directories in weights folder:")
    if os.path.exists(weights_dir):
        for algo in os.listdir(weights_dir):
            algo_path = os.path.join(weights_dir, algo)
            if os.path.isdir(algo_path):
                print(f"\n{algo}:")
                for env in os.listdir(algo_path):
                    print(f"  - {env}")

    df = collect_checkpoint_data(weights_dir)
    if df.empty:
        print("\nNo data to plot!")
        return

    # Generate plots
    plot_checkpoint_comparison(
        df, 
        save_path=os.path.join(plots_dir, 'checkpoint_comparison.png')
    )
    plot_learning_progress(
        df, 
        save_path=os.path.join(plots_dir, 'learning_progress.png')
    )

    # Generate summary table
    summary_table = create_summary_table(
        df, 
        save_path=os.path.join(plots_dir, 'checkpoint_summary.csv')
    )

    print("\nCheckpoint Performance Summary:")
    print(summary_table)
    print("\nAll visualizations have been saved to:", plots_dir)

if __name__ == "__main__":
    main()
