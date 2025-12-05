from train.train_loop import train_loop
import optuna
from functools import partial
from typing import Optional, Iterable
import numpy as np


def objective(trial, config, network, agent, env, episodes: int = 200, loops: int = 10, seeds: Optional[Iterable[int]] = None):
    assert episodes > 0, "episodes argument has to be bigger than 0"
    if seeds is None:
        seeds = [i for i in range(10)]

    converted_config = {}
    for k, v in config.items():
        match str(v["type"]).lower():
            case "float":
                converted_config[k] = trial.suggest_float(str(k), v["value"]["low"], v["value"]["high"])
            case "log_float":
                converted_config[k] = trial.suggest_loguniform(str(k), v["value"]["low"], v["value"]["high"])
            case "int":
                converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"])
            case "list":
                converted_config[k] = trial.suggest_categorical(str(k), v["value"])
        

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    agent = agent(num_actions, num_obs, config=converted_config, network=network)

    scores = []
    for _ in range(loops):
        score = train_loop(agent, env, episodes=episodes, seeds=seeds, dyn_print=False, plot=False)
        scores.append(score)

    scores.sort()
    return np.median(scores)

def optimize_agent(n_trials: int, config, network, agent, env, episodes: int = 200, loops: int = 5, seeds: Optional[Iterable[int]] = None):
    study = optuna.create_study(direction="maximize", storage="sqlite:///instance/db.sqlite3")

    par_objective = partial(objective, config=config, network=network, agent=agent, env=env, episodes=episodes, loops=loops, seeds=seeds)

    study.optimize(par_objective, n_trials=n_trials, show_progress_bar=True)
    