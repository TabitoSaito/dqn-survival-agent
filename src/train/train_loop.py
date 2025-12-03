import torch
from itertools import count
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from utils.constants import DEVICE
from train.plots import plot_training
from multiprocessing import Process, Queue


def train_loop(agent, env, episodes=0, seed=None):
    scores = []
    steps = []
    best_avg_reward = -float("inf")

    queue = Queue(maxsize=1000)

    p = Process(target=plot_training, args=(queue,), daemon=False)
    p.start()

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
                    scores.append(score)
                    steps.append(t)
                    break

            if np.mean(scores[-100:]) > best_avg_reward and cur_episode > 100:
                best_avg_reward = np.mean(scores[-100:])

            print(
                f"\rEpisode {cur_episode}\t\tAverage Score: {np.mean(scores[-100:]):.2f}\t\tAverage steps: {np.mean(steps[-100:]):.2f}",
                end="",
            )

            if cur_episode % 100 == 0:
                print(
                    f"\rEpisode {cur_episode}\t\tAverage Score: {np.mean(scores[-100:]):.2f}\t\tAverage steps: {np.mean(steps[-100:]):.2f}",
                )
            
            queue.put((scores, best_avg_reward))

            if episodes == 0:
                continue
            if cur_episode >= episodes:
                break
    except KeyboardInterrupt:
        pass

    if cur_episode % 100 != 0:
        print("")
