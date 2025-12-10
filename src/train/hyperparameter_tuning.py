from train.train_loop import train_loop
import optuna
from functools import partial
from typing import Optional, Iterable
import numpy as np
import copy


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
                converted_config[k] = trial.suggest_float(str(k), v["value"]["low"], v["value"]["high"], log=True)
            case "int":
                converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"])
            case "list":
                converted_config[k] = trial.suggest_categorical(str(k), v["value"])

    scores = []
    aucs = []
    for loop_idx in range(loops):
        cur_env = copy.deepcopy(env)

        state, info = env.reset()

        num_actions = env.action_space.n
        num_obs = len(state)

        cur_agent = agent(num_actions, num_obs, config=converted_config, network=network)

        score, auc = train_loop(cur_agent, cur_env, episodes=episodes, seeds=seeds, dyn_print=False, plot=False)
        scores.append(score)
        aucs.append(auc)

    scores.sort()
    aucs.sort()
    return np.median(scores), np.median(aucs)

def optimize_agent(n_trials: int, config, network, agent, env, episodes: int = 200, loops: int = 5, seeds: Optional[Iterable[int]] = None):
    study = optuna.create_study(directions=["maximize", "maximize"], storage="sqlite:///instance/db.sqlite3")

    par_objective = partial(objective, config=config, network=network, agent=agent, env=env, episodes=episodes, loops=loops, seeds=seeds)

    study.optimize(par_objective, n_trials=n_trials, show_progress_bar=True)
    