import torch
from itertools import count, cycle
import numpy as np
from utils.constants import DEVICE
from train.plots import plot_training
from multiprocessing import Process, Queue
from typing import Iterable, Optional


class TrainLoop:
    def __init__(self, agent, env, seeds: Optional[Iterable[int]] = None, dyn_print=True, plot=True, save_agent: Optional[str] = None) -> None:
        self.agent = agent
        self.env = env

        self.dyn_print = dyn_print
        self.plot = plot

        self.scores = []
        self.steps = []
        self.best_avg_reward = -float("inf")

        self.queue1 = Queue(maxsize=1000)
        self.queue2 = Queue(maxsize=1000)

        self.seeds = cycle(seeds) if seeds is Iterable else None
        self.save_agent = save_agent

        if plot:
            self.p = Process(target=plot_training, args=(self.queue1, self.queue2,), daemon=False)
            self.p.start()

        self.cur_episode = 0
        self.best_episode = 0

    def episode_step(self):
        self.cur_episode += 1

        seed = next(self.seeds) if self.seeds is Iterable else None

        state, info = self.env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        self.agent.update_epsilon()

        score = 0
        q_values_buffer = []
        q_values_mean = []
        for t in count():
            action, q_values = self.agent.act(state)
            obs, reward, terminated, truncated, info = self.env.step(action.item())

            q_values_buffer.append(q_values)

            score += reward
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated
            done = torch.tensor([done], device=DEVICE, dtype=torch.float32)

            next_state = torch.tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            loss = self.agent.step(state, action, next_state, reward, done)

            state = next_state

            if self.plot:
                self.queue2.put((loss))

            if terminated or truncated:
                self.scores.append(score)
                self.steps.append(t)
                q_values_buffer = torch.cat(q_values_buffer)
                q_values_mean = q_values_buffer.mean(dim=0).tolist()
                break

        if np.mean(self.scores[-100:]) > self.best_avg_reward and self.cur_episode > 100:
            self.best_avg_reward = np.mean(self.scores[-100:])
            self.best_episode = self.cur_episode
            if self.save_agent is not None:
                self.agent.save(self.save_agent)

        if self.dyn_print:
            print(
                f"\rEpisode {self.cur_episode}\t\tAverage Score: {np.mean(self.scores[-100:]):.2f}\t\tAverage steps: {np.mean(self.steps[-100:]):.2f}\t\tEpsilon: {self.agent.epsilon:.4f}",
                end="",
            )

            if self.cur_episode % 100 == 0:
                print(
                    f"\rEpisode {self.cur_episode}\t\tAverage Score: {np.mean(self.scores[-100:]):.2f}\t\tAverage steps: {np.mean(self.steps[-100:]):.2f}\t\tEpsilon: {self.agent.epsilon:.4f}",
                )
        
        if self.plot:
            self.queue1.put((self.scores, self.best_avg_reward, q_values_mean, self.agent.epsilon))

    def end_training(self):
        if self.cur_episode % 100 != 0 and self.dyn_print:
            print("")


def prebuilt_train_loop(agent, env, episodes=0, seeds: Optional[Iterable[int]] = None, dyn_print=True, plot=True, save_agent: Optional[str] = None):
    loop = TrainLoop(agent=agent, env=env, seeds=seeds, dyn_print=dyn_print, plot=plot, save_agent=save_agent)
    try:
        for _ in count(start=1):
            loop.episode_step()
            if episodes == 0:
                continue
            if loop.cur_episode >= episodes:
                break
    except KeyboardInterrupt:
        pass

    if loop.cur_episode % 100 != 0 and dyn_print:
        print("")

