from train.train_loop import TrainLoop
import optuna
from functools import partial
from typing import Optional, Iterable
import numpy as np
import copy


def objective(trial, config, network, agent, env, max_episodes: int = 2000, loops: int = 10, seeds: Optional[Iterable[int]] = None):
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
            case "list":
                converted_config[k] = trial.suggest_categorical(str(k), v["value"])

    scores = []
    aucs = []
    train_len = []
    for loop_idx in range(loops):
        cur_env = copy.deepcopy(env)

        state, info = env.reset()

        num_actions = env.action_space.n
        num_obs = len(state)

        cur_agent = agent(num_actions, num_obs, config=converted_config, network=network)

        loop = TrainLoop(cur_agent, cur_env, seeds=seeds, dyn_print=False, plot=False)
        while loop.cur_episode < max_episodes:
            loop.episode_step()
            if loop.cur_episode - loop.best_episode > 200:
                break
        
        trial.set_user_attr("Episodes", loop.cur_episode)

        score = np.mean(loop.scores[-100:])
        auc = np.trapezoid(loop.scores) / loop.cur_episode

        scores.append(score)
        aucs.append(auc)
        train_len.append(loop.cur_episode)

    scores.sort()
    aucs.sort()

    trial.set_user_attr("Episodes", np.mean(train_len))

    return np.median(scores), np.median(aucs)

def optimize_agent(n_trials: int, config, network, agent, env, max_episodes: int = 2000, loops: int = 5, seeds: Optional[Iterable[int]] = None):
    study = optuna.create_study(directions=["maximize", "maximize"], storage="sqlite:///instance/db.sqlite3")

    par_objective = partial(objective, config=config, network=network, agent=agent, env=env, max_episodes=max_episodes, loops=loops, seeds=seeds)

    study.optimize(par_objective, n_trials=n_trials, show_progress_bar=True)
    