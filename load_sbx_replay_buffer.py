from pathlib import Path

import cv2
import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig
from sbx import PPO, SAC

from utils.env_utils import env_fn
from utils.general_utils import omegaconf_to_dict, print_dict
from utils.logger import log


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    log("start")
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # load replay buffer
    exp_dir = Path(cfg.results_dir) / cfg.ckpt_file_name
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory {exp_dir} does not exist")

    policy_cls = PPO if cfg.algo_name == "ppo" else SAC

    ckpt_model = str(exp_dir / cfg.ckpt_dir / f"ckpt_{cfg.ckpt_step}_steps")
    model = policy_cls.load(ckpt_model)
    log(f"The model has {model.replay_buffer.size()} transitions in its buffer")

    replay_buffer_f = str(
        exp_dir / cfg.ckpt_dir / f"ckpt_replay_buffer_{cfg.ckpt_step}_steps"
    )
    model.load_replay_buffer(replay_buffer_f)
    log(f"Replay buffer loaded from {replay_buffer_f}")
    log(f"The model has {model.replay_buffer.size()} transitions in its buffer")

    # replay some of these transitions
    num_timesteps = model.replay_buffer.size()
    n_envs = model.replay_buffer.n_envs
    log(f"Replaying {num_timesteps} transitions from {n_envs} environments")

    observations = model.replay_buffer.observations[:num_timesteps]
    actions = model.replay_buffer.actions[:num_timesteps]
    infos = model.replay_buffer.infos[:num_timesteps]
    dones = model.replay_buffer.dones[:num_timesteps]

    # split transitions into chunks based on dones
    # (i.e. split into episodes)
    episode_starts = np.where(dones)[0] + 1
    log(f"Number of episodes: {len(episode_starts) + 1}")

    observations = np.split(observations, episode_starts)
    actions = np.split(actions, episode_starts)
    infos = np.split(infos, episode_starts)
    dones = np.split(dones, episode_starts)

    env = env_fn(cfg.env_id, 0, **cfg.env_kwargs)
    env.reset()

    for episode_indx, (obs, act, info, done) in tqdm.tqdm(
        enumerate(zip(observations, actions, infos, dones))
    ):
        if episode_indx >= cfg.num_episodes_replay:
            break

        log(f"Replaying episode {episode_indx}, length: {len(obs)}")
        frames = []

        for t in range(len(obs)):
            obs_t = obs[t]
            act_t = act[t]
            info_t = info[t]
            done_t = done[t]

            env.set_env_state(info_t[0]["env_state"])
            img = env.render()
            frames.append(img)
            if done_t:
                break

        # save frames as video
        video_f = (
            exp_dir
            / cfg.video_dir
            / "buffer_vids"
            / f"replay_episode_{episode_indx}.mp4"
        )
        video_f.parent.mkdir(parents=True, exist_ok=True)
        log(f"Saving video to {video_f}")
        writer = cv2.VideoWriter(
            str(video_f),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cfg.fps,
            (img.shape[1], img.shape[0]),
        )
        for img in frames:
            # flip the image vertically
            img = cv2.flip(img, 0)
            # convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img)
        writer.release()


if __name__ == "__main__":
    main()
