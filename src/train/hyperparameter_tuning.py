from train.train_loop import train_loop
import optuna
from functools import partial


def objective(trial, config, agent, network, env, episodes: int = 200, loops: int = 5):
    assert episodes > 0, "episodes argument has to be bigger than 0"
    converted_config = {}
    for k, v in config:
        assert type(v) is list, "config values have to be lists"
        match str(v["type"]).lower():
            case "float":
                converted_config[k] = trial.suggest_uniform(str(k), v["low"], v["high"])
            case "log_float":
                converted_config[k] = trial.suggest_loguniform(str(k), v["low"], v["high"])
            case "int":
                converted_config[k] = trial.suggest_int(str(k), v["low"], v["high"])
            case "list":
                converted_config[k] = trial.suggest_categorical(str(k), v["value"])
        

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    agent = agent(num_actions, num_obs, config=converted_config, network=network)

    score = 0
    for _ in range(loops):
        score += train_loop(agent, env, episodes=episodes, dyn_print=False, plot=False)

    return score / loops

def optimize_agent(n_trials: int, config, agent, network, env, episodes: int = 200, loops: int = 5):
    study = optuna.create_study(direction="maximize")

    par_objective = partial(objective, config=config, agent=agent, network=network, env=env, episodes=episodes, loops=loops)

    study.optimize(par_objective, n_trials=n_trials, show_progress_bar=True)

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_slice(study)
    
