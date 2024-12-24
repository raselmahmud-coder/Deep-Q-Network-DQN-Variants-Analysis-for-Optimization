# utils/video_recorder.py

import gymnasium as gym
import imageio

def record_video(env, agent, path, max_steps=1000):
    """Record a video of the agent's performance."""
    env = gym.make(env.spec.id, render_mode="rgb_array")
    frames = []
    state, info = env.reset(seed=42)
    ep_reward = 0
    done = False
    step = 0
    while not done and step < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        frame = env.render()
        frames.append(frame)
        state = next_state
        step += 1
    env.close()
    
    # Save frames as video
    imageio.mimsave(path, frames, fps=30)
    print(f"Video saved to {path} with reward {ep_reward}")
