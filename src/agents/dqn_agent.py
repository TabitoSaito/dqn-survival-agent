import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from typing import Callable

from buffers.replay_memory import ReplayMemory
from utils.constants import DEVICE, Experiences


class DQNAgent:
    def __init__(self, action_size, observation_size, config, network: Callable[[int, int], nn.Module]) -> None:
        self.action_size = action_size
        self.config = config

        self.policy_net = network(observation_size, action_size).to(DEVICE)
        self.target_net = network(observation_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=config["LR"], amsgrad=True
        )
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(config["CAPACITY"])

        self.steps_done = 0

    def act(self, state, train_mode=True):
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()

        sample = random.random()

        epsilon = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.steps_done / self.config["EPS_DECAY"])

        if train_mode:
            self.steps_done += 1

        if sample > epsilon or not train_mode:
            return q_values.argmax(keepdim=True)
        else:
            return torch.tensor(
                [random.sample([i for i in range(self.action_size)], 1)],
                device=DEVICE,
                dtype=torch.long,
            )

    def step(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences = self.memory.sample(self.config["MINI_BATCH_SIZE"])
            batch = Experiences(*zip(*experiences))
            self.learn(batch)

    def learn(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        try:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        except AttributeError:
            pass

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_action_values = self.target_net(next_state_batch).max(1).values

            expected_state_action_values = (
                next_state_action_values * self.config["GAMMA"]
            ) * (1 - done_batch) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.update_net()

    def update_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config[
                "TAU"
            ] + target_net_state_dict[key] * (1 - self.config["TAU"])
        self.target_net.load_state_dict(target_net_state_dict)



class DoubleDQNAgent:
    def __init__(self, action_size, observation_size, config, network: Callable[[int, int], nn.Module]) -> None:
        self.action_size = action_size
        self.config = config

        self.policy_net = network(observation_size, action_size).to(DEVICE)
        self.target_net = network(observation_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=config["LR"], amsgrad=True
        )
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(config["CAPACITY"])

        self.steps_done = 0

    def act(self, state, train_mode=True):
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()

        sample = random.random()

        epsilon = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.steps_done / self.config["EPS_DECAY"])

        if train_mode:
            self.steps_done += 1

        if sample > epsilon or not train_mode:
            return q_values.argmax(keepdim=True)
        else:
            return torch.tensor(
                [random.sample([i for i in range(self.action_size)], 1)],
                device=DEVICE,
                dtype=torch.long,
            )

    def step(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences = self.memory.sample(self.config["MINI_BATCH_SIZE"])
            batch = Experiences(*zip(*experiences))
            self.learn(batch)

    def learn(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        try:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        except AttributeError:
            pass

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_state_action_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)

            expected_state_action_values = (
                next_state_action_values * self.config["GAMMA"]
            ) * (1 - done_batch) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.update_net()

    def update_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config[
                "TAU"
            ] + target_net_state_dict[key] * (1 - self.config["TAU"])
        self.target_net.load_state_dict(target_net_state_dict)
