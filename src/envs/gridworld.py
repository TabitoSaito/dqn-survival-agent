from typing import Optional, Any
import numpy as np
import gymnasium as gym
import pygame

from utils.constants import Actions
import utils.helper as helper


class GridWorldEnv(gym.Env):
    def __init__(self, render_mode=None, size: int = 5) -> None:
        self.size = size
        self.window_size = 512
        self._num_fruits = 4

        self._agent_location = np.array([-1, -1], dtype=int)
        self._fruit_locations = [np.array([-1, -1], dtype=int) for _ in range(self._num_fruits)]

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "fruits": gym.spaces.Tuple([gym.spaces.Box(0, size - 1, shape=(2,), dtype=int) for _ in range(self._num_fruits)])
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
        return {"agent": self._agent_location, "fruits": self._fruit_locations}
    
    def _get_info(self) -> dict:
        """Get relevant info if needed. Not Implemented.

        Returns:
            dict: Info content
        """
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        # initialization logic
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._fruit_locations = helper.get_unique_coordinates((self.size, self.size), self._num_fruits)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # game logic
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = False
        truncated = False
        reward = 0

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
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )

        # draw fruits
        for fruit_location in self._fruit_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * fruit_location,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
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

