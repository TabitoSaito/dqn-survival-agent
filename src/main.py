import yaml
from gymnasium.wrappers import FlattenObservation

from envs.gridworld import GridWorldEnv
from networks.dqn_networks import DQN, DuelingDQN, NoisyDQN
from agents.dqn_agent import DQNAgent, DoubleDQNAgent, DoubleDQNAgentPER
from train.train_loop import train_loop
from train.evaluation import render_run, eval_agent

with open("configs/envs/default.yaml") as stream:
    env_config = yaml.safe_load(stream)

with open("configs/agent/default.yaml") as stream:
    agent_config = yaml.safe_load(stream)

env = GridWorldEnv(config=env_config, size=5, render_mode="rgb_array")
env = FlattenObservation(env)
state, info = env.reset()

num_actions = env.action_space.n
num_obs = len(state)

agent = DoubleDQNAgentPER(num_actions, num_obs, config=agent_config, network=NoisyDQN)

train_loop(agent, env, episodes=5000, seed=1)

render_run(agent, env, "test", runs=1, seed=1)

# eval_agent(agent, env)
