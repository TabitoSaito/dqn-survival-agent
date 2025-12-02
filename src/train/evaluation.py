import torch
import cv2
from itertools import count
from utils.constants import DEVICE


def render_run(agent, env, run_name: str, runs: int = 10, seed=None):
    assert env.render_mode == "rgb_array"

    for i in range(runs):
        state, info = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        frames = []

        score = 0
        for t in count():
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env.step(action.item())

            score += reward
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    obs, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

            agent.step(state, action, next_state, reward)

            frame = env.render()
            frames.append(frame)

            if done:
                break
        
        print(f"Run {i + 1} Closed with total Reward: {score:.2f} and total Steps: {t}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"src/replays/{run_name}_{i + 1}.mp4",
            fourcc,
            env.metadata["render_fps"],
            (frame.shape[0], frame.shape[1]),
        )
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"saved video under src/replays/{run_name}_{i + 1}.mp4")
    env.close()