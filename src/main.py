import yaml
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import numpy as np
import torch

from envs.gridworld import GridWorldEnv
from networks.dqn_networks import DQN, DuelingDQN, NoisyDQN
from agents.dqn_agent import DQNAgent, DoubleDQNAgent, DoubleDQNAgentPER
from train.train_loop import TrainLoop, prebuilt_train_loop
from train.evaluation import render_run, eval_agent

from train.hyperparameter_tuning import optimize_agent

with open("configs/envs/default.yaml") as stream:
    env_config = yaml.safe_load(stream)

with open("configs/agent/test_dqn_cartpole.yaml") as stream:
    agent_config = yaml.safe_load(stream)

env = gym.make("CartPole-v1", render_mode="rgb_array")

# env = GridWorldEnv(config=env_config, size=5, render_mode="rgb_array")
env = FlattenObservation(env)
state, info = env.reset()

num_actions = env.action_space.n
num_obs = len(state)

agent = DoubleDQNAgent(num_actions, num_obs, config=agent_config, network=DQN)

prebuilt_train_loop(agent, env, save_agent="test", episodes=600)

agent.load("test")
agent.policy_net.eval()

render_run(agent, env, "test", runs=10)

# eval_agent(agent, env)

# with open("configs/hyperparameter_tuning/default.yaml") as stream:
#     agent_config = yaml.safe_load(stream)

# optimize_agent(100, agent_config, DQN, DQNAgent, env, max_episodes=600, name="DQN")

# optimize_agent(100, agent_config, DQN, DoubleDQNAgent, env, max_episodes=600, name="DoubleDQN")
