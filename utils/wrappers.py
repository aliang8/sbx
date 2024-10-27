import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super(FrameStackWrapper, self).__init__(env)
        self.n_stack = n_stack
        self.frames = np.zeros(
            (n_stack,) + env.observation_space.shape, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.repeat(env.observation_space.low[np.newaxis, ...], n_stack, axis=0),
            high=np.repeat(
                env.observation_space.high[np.newaxis, ...], n_stack, axis=0
            ),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.frames.fill(0)
        self.frames[-1] = obs  # Only set the last frame with the initial observation
        return self.frames, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs
        return self.frames, reward, terminated, truncated, info


class FlipImageWrapper(gym.Wrapper):
    def render(self):
        img = self.env.render()
        return np.flip(img, axis=0)
