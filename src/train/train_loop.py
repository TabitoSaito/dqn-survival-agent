import torch
from itertools import count
import numpy as np
from collections import deque
from utils.constants import DEVICE


def train_loop(agent, env, episodes=0, seed=None):
    scores_on_100_episodes = deque(maxlen=100)
    steps_on_100_episodes = deque(maxlen=100)
    try:
        for cur_episode in count(start=1):
            state, info = env.reset(seed=seed)
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            score = 0
            for t in count():
                action = agent.act(state)
                obs, reward, terminated, truncated, info = env.step(action.item())

                score += reward
                reward = torch.tensor([reward], device=DEVICE)
                done = terminated or truncated
                done = torch.tensor([done], device=DEVICE, dtype=torch.float32)

                next_state = torch.tensor(
                    obs, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

                agent.step(state, action, next_state, reward, done)

                state = next_state

                if terminated or truncated:
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
    except KeyboardInterrupt:
        pass

    if cur_episode % 100 != 0:
        print("")
