import yaml
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym

from envs.gridworld import GridWorldEnv
from train.train_loop import TrainLoop, prebuilt_train_loop
from train.evaluation import render_run, eval_agent
from train.hyperparameter_tuning import optimize_agent
from utils.helper import build_agent

with open("configs/envs/default.yaml") as stream:
    env_config = yaml.safe_load(stream)

with open("configs/agent/test_dqn_cartpole.yaml") as stream:
    agent_config = yaml.safe_load(stream)

env = gym.make("CartPole-v1", render_mode="rgb_array")

# env = GridWorldEnv(config=env_config, size=5, render_mode="rgb_array")
env = FlattenObservation(env)

agent = build_agent(agent_config, env)

prebuilt_train_loop(agent, env, save_agent="test", episodes=600)

agent.load("test")
agent.policy_net.eval()

render_run(agent, env, "test", runs=10)

# eval_agent(agent, env)

# with open("configs/hyperparameter_tuning/default.yaml") as stream:
#     agent_config = yaml.safe_load(stream)

# optimize_agent(100, agent_config, DQN, DQNAgent, env, max_episodes=600, name="DQN")

# optimize_agent(100, agent_config, DQN, DoubleDQNAgent, env, max_episodes=600, name="DoubleDQN")
