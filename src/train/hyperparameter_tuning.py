from train.train_loop import TrainLoop
import optuna
from functools import partial
from typing import Optional, Iterable
import numpy as np
import copy
import torch


def objective(trial, config, network, agent, env, max_episodes: int = 2000, loops: int = 5, seeds: Optional[Iterable[int]] = None):
    assert max_episodes > 0, "episodes argument has to be bigger than 0"

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
            case "log_int":
                converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"], log=True)
            case "list":
                converted_config[k] = trial.suggest_categorical(str(k), v["value"])

    aucs = []
    train_len = []

    torch.manual_seed(0)

    for loop_idx in range(loops):
        cur_env = copy.deepcopy(env)

        state, info = env.reset()

        num_actions = env.action_space.n
        num_obs = len(state)

        cur_agent = agent(num_actions, num_obs, config=converted_config, network=network)

        loop = TrainLoop(cur_agent, cur_env, seeds=seeds, dyn_print=False, plot=False)

        prune_count = 0
        while loop.cur_episode < max_episodes:
            loop.episode_step()

            if loop.cur_episode < 200:
                continue
            if loop.cur_episode % 50 != 0:
                continue
            
            if (np.mean(loop.scores[-200: -100]) - np.mean(loop.scores[-100:])) / np.mean(loop.scores[-200: -100]) < 0.02:
                prune_count += 1
            else:
                prune_count = 0

            if prune_count >= 3:
                break

        auc = np.trapezoid(loop.scores) / loop.cur_episode

        aucs.append(auc)
        train_len.append(loop.cur_episode)
        trial.set_user_attr("Episodes", train_len)

    aucs.sort()

    trial.set_user_attr("Episodes", train_len)

    return np.median(aucs)

def optimize_agent(n_trials: int, config, network, agent, env, max_episodes: int = 2000, loops: int = 5, seeds: Optional[Iterable[int]] = None, name: Optional[str] = None):
    study = optuna.create_study(direction="maximize", storage="sqlite:///instance/db.sqlite3", study_name=name)

    par_objective = partial(objective, config=config, network=network, agent=agent, env=env, max_episodes=max_episodes, loops=loops, seeds=seeds)

    study.optimize(par_objective, n_trials=n_trials, show_progress_bar=True)
    