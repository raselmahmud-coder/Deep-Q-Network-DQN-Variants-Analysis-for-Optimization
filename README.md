# Project outline

Each video saved inside `"results/video"` directory which contain one video for each environment and each algorithm only best strategy using gymnasium package.

In the `weights` folder have trained file `1000.pth`, `2000.pth`, `best.pth` and `training.log` for each algorithm and each environment.

`plots_images` folder contain all visualization `7 images` and a `checkpoint_summary.csv` file for each environments and algorithms

## Installation

Please follow these steps to install the necessary dependencies and set up the project locally. To see required packages please open the `requirements.txt` file.

### 1. Clone the repository:

```bash
git clone https://github.com/raselmahmud-coder/RL_Experiment_2.git
cd RL_Experiment_2
pip install -r requirements.txt
```

### For Training the Project:
In this project have 3 algorithms `"DQN", "DoubleDQN", "DuelingDQN"` and 3 environments `"CartPole-v1", "MountainCar-v0", "LunarLander-v3"`

You need to change the algorithm and environment argument for sequential training.

```bash
python main.py --algorithm DQN --environment CartPole-v1     
```

### For Visualization the Project:
We have 3 environment here:
- "CartPole-v1"
- "MountainCar-v0"
- "LunarLander-v3"


For visual and compare you need to set i.e., `env_name = 'LunarLander-v3'` manually need to change for each environment name and then run below command it will save specific folder each algorithms plot.

```bash
python .\results\visualize_comparison.py    
python .\results\compare_checkpoints.py    
```


## File Hierarchy
```bash
📦RL_Experiment_2
 ┣ 📂results
 ┃ ┣ 📂plots_images
 ┃ ┃ ┣ 📂CartPole-v1_ENV
 ┃ ┃ ┃ ┣ 📜checkpoint_comparison.png
 ┃ ┃ ┃ ┣ 📜checkpoint_summary.csv
 ┃ ┃ ┃ ┣ 📜combined_metrics.png
 ┃ ┃ ┃ ┣ 📜comparison_epsilon_decay.png
 ┃ ┃ ┃ ┣ 📜comparison_rewards.png
 ┃ ┃ ┃ ┣ 📜convergence_speed.png
 ┃ ┃ ┃ ┣ 📜learning_progress.png
 ┃ ┃ ┃ ┗ 📜stability_rewards.png
 ┃ ┃ ┣ 📂LunarLander-v3_ENV
 ┃ ┃ ┃ ┣ 📜checkpoint_comparison.png
 ┃ ┃ ┃ ┣ 📜checkpoint_summary.csv
 ┃ ┃ ┃ ┣ 📜combined_metrics.png
 ┃ ┃ ┃ ┣ 📜comparison_epsilon_decay.png
 ┃ ┃ ┃ ┣ 📜comparison_rewards.png
 ┃ ┃ ┃ ┣ 📜convergence_speed.png
 ┃ ┃ ┃ ┣ 📜learning_progress.png
 ┃ ┃ ┃ ┗ 📜stability_rewards.png
 ┃ ┃ ┣ 📂MountainCar-v0_ENV
 ┃ ┃ ┃ ┣ 📜checkpoint_comparison.png
 ┃ ┃ ┃ ┣ 📜checkpoint_summary.csv
 ┃ ┃ ┃ ┣ 📜combined_metrics.png
 ┃ ┃ ┃ ┣ 📜comparison_epsilon_decay.png
 ┃ ┃ ┃ ┣ 📜comparison_rewards.png
 ┃ ┃ ┃ ┣ 📜convergence_speed.png
 ┃ ┃ ┃ ┣ 📜learning_progress.png
 ┃ ┃ ┃ ┗ 📜stability_rewards.png
 ┃ ┃ ┣ 📜DDQN_Alog.png
 ┃ ┃ ┣ 📜dqn_algo.png
 ┃ ┃ ┣ 📜Dueling_dqn.png
 ┃ ┃ ┗ 📜model_code_snippet.jpeg
 ┃ ┣ 📂videos
 ┃ ┃ ┣ 📂DoubleDQN
 ┃ ┃ ┃ ┣ 📂CartPole-v1
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┃ ┣ 📂LunarLander-v3
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┃ ┗ 📂MountainCar-v0
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┣ 📂DQN
 ┃ ┃ ┃ ┣ 📂CartPole-v1
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┃ ┣ 📂LunarLander-v3
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┃ ┗ 📂MountainCar-v0
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┗ 📂DuelingDQN
 ┃ ┃ ┃ ┣ 📂CartPole-v1
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┃ ┣ 📂LunarLander-v3
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┃ ┃ ┗ 📂MountainCar-v0
 ┃ ┃ ┃ ┃ ┗ 📜best_strategy.mp4
 ┃ ┣ 📂weights
 ┃ ┃ ┣ 📂DoubleDQN
 ┃ ┃ ┃ ┣ 📂CartPole-v1
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┃ ┣ 📂LunarLander-v3
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┃ ┗ 📂MountainCar-v0
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┣ 📂DQN
 ┃ ┃ ┃ ┣ 📂CartPole-v1
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┃ ┣ 📂LunarLander-v3
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┃ ┗ 📂MountainCar-v0
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┗ 📂DuelingDQN
 ┃ ┃ ┃ ┣ 📂CartPole-v1
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┃ ┣ 📂LunarLander-v3
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┃ ┃ ┗ 📂MountainCar-v0
 ┃ ┃ ┃ ┃ ┣ 📜1000.pth
 ┃ ┃ ┃ ┃ ┣ 📜2000.pth
 ┃ ┃ ┃ ┃ ┣ 📜best.pth
 ┃ ┃ ┃ ┃ ┗ 📜training.log
 ┃ ┣ 📜compare_checkpoints.py
 ┃ ┗ 📜visualize_comparison.py
 ┣ 📂utils
 ┃ ┣ 📜logger.py
 ┃ ┗ 📜video_recorder.py
 ┣ 📜base_dqn.py
 ┣ 📜data.py
 ┣ 📜double_dqn.py
 ┣ 📜dqn.py
 ┣ 📜dueling_dqn.py
 ┣ 📜main.py
 ┣ 📜memory.py
 ┣ 📜models.py