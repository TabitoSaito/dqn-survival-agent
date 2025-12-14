import yaml
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym

from envs.gridworld import GridWorldEnv
from train.train_loop import TrainLoop, prebuilt_train_loop, train_best_agent
from train.evaluation import render_run, eval_agent
from train.hyperparameter_tuning import optimize_agent
from utils.helper import build_agent, load_agent

with open("configs/envs/default.yaml") as stream:
    env_config = yaml.safe_load(stream)

with open("configs/agent/test_survival.yaml") as stream:
    agent_config = yaml.safe_load(stream)

env = gym.make("CartPole-v1", render_mode="rgb_array")

env = GridWorldEnv(config=env_config, size=5, render_mode="rgb_array")
env = FlattenObservation(env)

train_best_agent(agent_config, env, "test", max_episodes=2000)

agent = load_agent("test")

# prebuilt_train_loop(agent, env, save_agent="test", episodes=600)

render_run(agent, env, "test", runs=10)

# with open("configs/hyperparameter_tuning/default.yaml") as stream:
#     agent_config = yaml.safe_load(stream)

# optimize_agent(100, agent_config, env, max_episodes=600, name="CartPole")

# env = GridWorldEnv(config=env_config, size=5, render_mode="rgb_array")
# env = FlattenObservation(env)

# optimize_agent(100, agent_config, env, max_episodes=1000, name="Survival")

