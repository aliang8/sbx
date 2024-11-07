import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_stack=4, save_imgs: bool = False, img_shape=(64, 64)):
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
        self.save_imgs = save_imgs
        if save_imgs:
            self.img_frames = np.zeros((n_stack,) + img_shape + (3,), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.frames.fill(0)
        self.frames[-1] = obs  # Only set the last frame with the initial observation
        return self.frames, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs

        # update the img_frames
        if self.save_imgs:
            img = self.env.render()
            # resize the image
            img = np.array(Image.fromarray(img).resize((64, 64)))
            self.img_frames = np.roll(self.img_frames, shift=-1, axis=0)
            self.img_frames[-1] = img
        return self.frames, reward, terminated, truncated, info


class FlipImageWrapper(gym.Wrapper):
    def render(self):
        img = self.env.render()
        return np.flip(img, axis=0)


# make wrapper to save the trajectory


class SaveTrajectoryWrapper(gym.Wrapper):
    def __init__(self, env, save_imgs: bool = False):
        super(SaveTrajectoryWrapper, self).__init__(env)
        self.save_imgs = save_imgs
        self.trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
        }
        if save_imgs:
            self.trajectory["images"] = []

        self.trajectories = []

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.trajectory["observations"].append(obs)
        self.trajectory["infos"].append(info)
        if self.save_imgs:
            self.trajectory["images"].append(self.env.img_frames)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.trajectory["observations"].append(obs)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["dones"].append(terminated or truncated)
        self.trajectory["infos"].append(info)

        if self.save_imgs:
            self.trajectory["images"].append(self.env.img_frames)

        if terminated or truncated:
            # convert to numpy
            for key in self.trajectory.keys():
                if key != "infos":
                    self.trajectory[key] = np.array(self.trajectory[key])

            self.trajectories.append(self.trajectory)

            # reset the trajectory data
            self.trajectory = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "infos": [],
            }

            if self.save_imgs:
                self.trajectory["images"] = []
        return obs, reward, terminated, truncated, info
