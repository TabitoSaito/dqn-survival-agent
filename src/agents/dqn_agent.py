import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from buffers.replay_memory import ReplayMemory
from utils.constants import DEVICE, Experiences


class DQNAgent:
    def __init__(self, action_size, config, network: nn.Module) -> None:
        self.action_size = action_size
        self.config = config

        self.policy_net = network.to(DEVICE)
        self.target_net = network.to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=config["LR"], amsgrad=True
        )
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(config["CAPACITY"])

        self.steps_done = 0

    def act(self, state, train_mode=True):
        sample = random.random()

        epsilon = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.steps_done / self.config["EPS_DECAY"])

        if train_mode:
            self.steps_done += 1

        if sample > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = self.target_net(next_state_batch).max(1).values

        expected_state_action_values = (
            next_state_values * self.config["GAMMA"]
        ) + reward_batch * (1 - done_batch)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
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
