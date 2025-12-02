import yaml
from gymnasium.wrappers import FlattenObservation

from envs.gridworld import GridWorldEnv
from networks.dqn_networks import Network
from agents.dqn_agent import Agent
from train.train_loop import train_loop
from train.evaluation import render_run

with open("configs/envs/default.yaml") as stream:
    env_config = yaml.safe_load(stream)

with open("configs/agent/default.yaml") as stream:
    agent_config = yaml.safe_load(stream)

env = GridWorldEnv(config=env_config, size=10, render_mode="rgb_array")
env = FlattenObservation(env)
state, info = env.reset()

num_actions = env.action_space.n
num_obs = len(state)

network = Network(num_obs, num_actions)

agent = Agent(num_actions, config=agent_config, network=network)

train_loop(agent, env, seed=0)

render_run(agent, env, "test", seed=0)
