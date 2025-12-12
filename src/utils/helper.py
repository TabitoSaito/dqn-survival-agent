import random
import numpy as np
from agents.dqn_agent import BaseAgent, DQNAgent, DQNAgentPER, DoubleDQNAgent, DoubleDQNAgentPER
from networks.dqn_networks import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN

def get_unique_coordinates(shape: tuple[int, int], n: int):
    row, col = shape
    return [np.array([value // row, value % col], dtype=int) for value in random.sample(range(row * col), n)]

def build_agent(config, env) -> BaseAgent:
    if config["DOUBLE_DQN"] is True:
        if config["PER"] is True:
            agent_class = DoubleDQNAgentPER
        else:
            agent_class = DoubleDQNAgent
    else:
        if config["PER"] is True:
            agent_class = DQNAgentPER
        else:
            agent_class = DQNAgent

    noisy = False

    if config["DUELING"] is True and config["NOISY"] is True:
        network = NoisyDuelingDQN
        noisy = True
    elif config["DUELING"] is True:
        network = DuelingDQN
    elif config["NOISY"] is True:
        network = NoisyDQN
        noisy = True
    else:
        network = DQN

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    agent = agent_class(num_actions, num_obs, config, network, noisy=noisy)

    return agent

    
