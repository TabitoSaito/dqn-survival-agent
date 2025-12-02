from typing import Optional, Any
import numpy as np
import gymnasium as gym
import pygame

from utils.constants import Actions
import utils.helper as helper

from gridworld_components import Human, Fruit


class GridWorldEnv(gym.Env):
    def __init__(self, render_mode=None, size: int = 5) -> None:
        self.size = size
        self.window_size = 512
        self._num_fruits = 4

        self._agent = Human(np.array([-1, -1], dtype=int))
        self._fruits = [
            Fruit(np.array([-1, -1], dtype=int)) for _ in range(self._num_fruits)
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "fruits": gym.spaces.Tuple(
                    [
                        gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
                        for _ in range(self._num_fruits)
                    ]
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self) -> dict:
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and fruits positions
        """
        return {
            "agent": self._agent.pos,
            "fruits": [fruit.pos for fruit in self._fruits],
        }

    def _get_info(self) -> dict:
        """Get full observation objects for debugging

        Returns:
            dict: agent and fruit objects
        """
        return {"agent": self._agent, "fruits": self._fruits}

    def reset(
        self, options: dict[str, Any], seed: Optional[int] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        # initialization logic
        self._agent = Human(
            self.np_random.integers(0, self.size, size=2, dtype=int),
            max_food=options["max_food"],
            max_age=options["max_age"],
            food_decay=options["food_decay"],
        )

        fruit_locations = helper.get_unique_coordinates(
            (self.size, self.size), self._num_fruits
        )
        self._fruits = [
            Fruit(position, amount=options["amount"], reg_time=options["reg_time"])
            for position in fruit_locations
        ]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # game logic
        direction = self._action_to_direction[action]
        self._agent.pos = np.clip(self._agent.pos + direction, 0, self.size - 1)

        reward = 0

        for fruit in self._fruits:
            if np.array_equal(self._agent.pos, fruit.pos):
                amount = fruit.harvest()
                if amount > 0:
                    self._agent.eat(amount)
                    reward += 1
            fruit.tick()
        self._agent.tick()

        terminated = self._agent.check_alive()

        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.typing.NDArray[Any] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.typing.NDArray[Any] | None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # draw fruits
        for fruit in self._fruits:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * fruit.pos,
                    (pix_square_size, pix_square_size),
                ),
            )

        # draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent.pos + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # draw grid
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
