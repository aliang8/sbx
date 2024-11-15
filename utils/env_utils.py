import metaworld
import numpy as np
from gymnasium.wrappers import TimeLimit

from utils.wrappers import FlipImageWrapper, FrameStackWrapper, SaveTrajectoryWrapper


def env_fn(
    env_id,
    env_idx,
    n_frame_stack=4,
    max_episode_steps=500,
    save_trajectory: bool = False,
    save_imgs: bool = False,
):
    ml1 = metaworld.ML1(env_id)
    env_cls = ml1.train_classes[env_id]

    # this is required to ensure that each environment
    # has a different random seed
    st0 = np.random.get_state()
    np.random.seed(env_idx)
    env = env_cls(render_mode="rgb_array", camera_name="corner")
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    env._freeze_rand_vec = True
    env.seed(env_idx)
    np.random.set_state(st0)

    env = FlipImageWrapper(env)

    if n_frame_stack > 1:
        env = FrameStackWrapper(env, n_stack=4, save_imgs=save_imgs)

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if save_trajectory:
        env = SaveTrajectoryWrapper(env, save_imgs=save_imgs)
    return env
