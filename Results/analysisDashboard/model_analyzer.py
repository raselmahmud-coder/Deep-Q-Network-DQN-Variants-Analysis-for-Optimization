# model_analyzer.py

import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
import os

# Determine the path to the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))  # Adjust based on your structure

# Add the root directory to sys.path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models import DQNNetwork, DuelingDQNNetwork, DuelingDQNNetwork  # Replace with actual import paths

def load_model(algorithm, environment, checkpoint_path, num_states, num_actions):
    """
    Loads a model based on the algorithm and environment.
    
    Args:
        algorithm (str): Algorithm name (DQN, DoubleDQN, DuelingDQN).
        environment (str): Environment name (CartPole-v1, etc.).
        checkpoint_path (str): Path to the .pth checkpoint file.
        num_states (int): Number of state features.
        num_actions (int): Number of possible actions.
    
    Returns:
        torch.nn.Module: Loaded model.
    """
    if algorithm == 'DQN':
        model = DQNNetwork(num_states, num_actions)
    elif algorithm == 'DoubleDQN':
        model = DoubleDQNNetwork(num_states, num_actions)
    elif algorithm == 'DuelingDQN':
        model = DuelingDQNNetwork(num_states, num_actions)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def extract_weights(model, layer_name):
    """
    Extracts weights from a specified layer of the model.
    
    Args:
        model (torch.nn.Module): The loaded model.
        layer_name (str): Name of the layer to extract weights from.
    
    Returns:
        np.ndarray: Weights as a NumPy array.
    """
    layer = dict(model.named_parameters()).get(layer_name, None)
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")
    return layer.data.cpu().numpy()

def plot_weight_distribution(weights, title, save_path=None):
    """
    Plots the distribution of weights.
    
    Args:
        weights (np.ndarray): Weights array.
        title (str): Plot title.
        save_path (str, optional): Path to save the plot image. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(weights.flatten(), bins=50, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_heatmap(weights, title, save_path=None):
    """
    Plots a heatmap of weights.
    
    Args:
        weights (np.ndarray): Weights array.
        title (str): Plot title.
        save_path (str, optional): Path to save the plot image. Defaults to None.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights, cmap='viridis')
    plt.title(title)
    plt.xlabel('Neuron')
    plt.ylabel('Input Features')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    base_dir = '../weights'
    algorithms = ['DQN', 'DoubleDQN', 'DuelingDQN']
    environments = ['CartPole-v1', 'LunarLander-v3', 'MountainCar-v0']
    checkpoints = ['1000.pth', '2000.pth', 'best.pth']
    
    for algo in algorithms:
        for env in environments:
            env_path = os.path.join(base_dir, algo, env)
            num_states = 4 if env == 'CartPole-v1' else 8 if env == 'LunarLander-v3' else 6  # Example state sizes
            num_actions = 2 if env in ['CartPole-v1', 'MountainCar-v0'] else 4  # Example action sizes
            
            for chk in checkpoints:
                chk_path = os.path.join(env_path, chk)
                if os.path.exists(chk_path):
                    model = load_model(algo, env, chk_path, num_states, num_actions)
                    # Example: Extract and plot weights from the first layer (e.g., 'fc1.weight')
                    try:
                        weights = extract_weights(model, 'fc1.weight')  # Replace with actual layer names
                        plot_title = f"{algo} - {env} - {chk} - fc1.weight Distribution"
                        save_fig_path = os.path.join(env_path, f"{chk}_fc1_weight_distribution.png")
                        plot_weight_distribution(weights, plot_title, save_path=save_fig_path)
                        
                        # Plot heatmap (optional, may be large)
                        # plot_heatmap(weights, f"{algo} - {env} - {chk} - fc1.weight Heatmap", save_path=os.path.join(env_path, f"{chk}_fc1_weight_heatmap.png"))
                    except ValueError as e:
                        print(e)
                else:
                    print(f"Checkpoint {chk} not found in {env_path}")
