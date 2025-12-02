from envs.gridworld import GridWorldEnv
import random
import yaml

with open("configs/envs/default.yaml") as stream:
    env_config = yaml.safe_load(stream)

env = GridWorldEnv(render_mode="human")

state, info = env.reset(env_config)

done = False
while not done:
    _, _, done, _, _ = env.step(random.randint(0, 3))
