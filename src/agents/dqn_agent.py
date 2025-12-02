import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from buffers.replay_memory import ReplayMemory
from utils.constants import DEVICE, Experiences


class Agent:
    def __init__(self, action_size, config, network: nn.Module) -> None:
        self.action_size = action_size
        self.config = config

        self.policy_net = network.to(DEVICE)
        self.target_net = network.to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=config["LR"], amsgrad=True
        )
        self.memory = ReplayMemory(config["CAPACITY"])

        self.steps_done = 0

    def act(self, state):
        sample = random.random()

        epsilon = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.steps_done / self.config["EPS_DECAY"])

        if sample > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [random.sample([i for i in range(self.action_size)], 1)],
                device=DEVICE,
                dtype=torch.long,
            )

    def step(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences = self.memory.sample(self.config["MINI_BATCH_SIZE"])
            batch = Experiences(*zip(*experiences))
            self.learn(batch)

    def learn(self, batch):
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=DEVICE,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.config["MINI_BATCH_SIZE"], device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (
            next_state_values * self.config["GAMMA"]
        ) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config[
                "TAU"
            ] + target_net_state_dict[key] * (1 - self.config["TAU"])
        self.target_net.load_state_dict(target_net_state_dict)
