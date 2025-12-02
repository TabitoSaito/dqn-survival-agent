import torch
from itertools import count
import numpy as np
from collections import deque
from utils.constants import DEVICE


def train_loop(agent, env, episodes=0):
    scores_on_100_episodes = deque(maxlen=100)
    steps_on_100_episodes = deque(maxlen=100)

    for cur_episode in count():
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        score = 0
        for t in count():
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env.step(action.item())

            score += reward
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    obs, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

            agent.step(state, action, next_state, reward)

            if done:
                scores_on_100_episodes.append(score)
                steps_on_100_episodes.append(t)
                break

        print(
            f"\rEpisode {cur_episode}\t\tAverage Score: {np.mean(scores_on_100_episodes):.2f}\t\tAverage steps: {np.mean(steps_on_100_episodes):.2f}",
            end="",
        )

        if cur_episode % 100 == 0:
            print(
                f"\rEpisode {cur_episode}\t\tAverage Score: {np.mean(scores_on_100_episodes):.2f}\t\tAverage steps: {np.mean(steps_on_100_episodes):.2f}",
            )

        if episodes == 0:
            continue
        if cur_episode >= episodes:
            break
