from pathlib import Path

import cv2
import einops
import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig
from prompt_dtla.scripts.data import save_dataset
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

    # [T, N, *obs_shape]
    # flatten the first two dimensions
    observations = model.replay_buffer.observations[:num_timesteps]
    actions = model.replay_buffer.actions[:num_timesteps]
    infos = model.replay_buffer.infos[:num_timesteps]
    dones = model.replay_buffer.dones[:num_timesteps]
    rewards = model.replay_buffer.rewards[:num_timesteps]

    observations = einops.rearrange(observations, "T N ... -> (N T) ...", N=n_envs)
    actions = einops.rearrange(actions, "T N ... -> (N T) ...", N=n_envs)
    dones = einops.rearrange(dones, "T N ... -> (N T) ...", N=n_envs)
    env_states = np.array([[_info["env_state"] for _info in info] for info in infos])
    env_states = einops.rearrange(env_states, "T N ... -> (N T) ...", N=n_envs)
    rewards = einops.rearrange(rewards, "T N ... -> (N T) ...", N=n_envs)

    log(f"Flattened observations shape: {observations.shape}")

    # split transitions into chunks based on dones
    # (i.e. split into episodes)
    episode_starts = np.where(dones)[0] + 1
    num_episodes = len(episode_starts) + 1
    log(f"Number of episodes: {num_episodes}")

    observations = np.split(observations, episode_starts)
    actions = np.split(actions, episode_starts)
    env_states = np.split(env_states, episode_starts)
    dones = np.split(dones, episode_starts)
    rewards = np.split(rewards, episode_starts)

    # figure out the episode returns
    episode_returns = np.array([np.sum(reward) for reward in rewards])
    log(
        f"Mean episode return: {np.mean(episode_returns)}, std: {np.std(episode_returns)}"
    )

    # threshold rewards by value
    min_ret = 2000
    traj_indices = np.where(episode_returns >= min_ret)[0]
    log(f"Number of trajectories with return >= {min_ret}: {len(traj_indices)}")

    # filter out trajectories with return < min_ret
    observations = [observations[i] for i in traj_indices]
    actions = [actions[i] for i in traj_indices]
    env_states = [env_states[i] for i in traj_indices]
    dones = [dones[i] for i in traj_indices]
    rewards = [rewards[i] for i in traj_indices]

    # ============ Write this to dataset ============
    num_trajs_to_save = min(num_episodes, 200)
    trajectories = []
    log("Saving to TFDS format...")
    for episode_idx in tqdm.tqdm(range(num_trajs_to_save), desc="episode"):
        # convert to list of trajectories
        trajectory = {
            "observations": observations[episode_idx][:, -1],  # this was frame-stacked
            "actions": actions[episode_idx],
            "rewards": rewards[episode_idx],
        }
        tmp = rewards[episode_idx]
        trajectory.update(
            {
                "discount": np.ones_like(tmp),
                "is_last": np.zeros_like(tmp),
                "is_first": np.zeros_like(tmp),
                "is_terminal": np.zeros_like(tmp),
            }
        )

        trajectory["is_last"][-1] = 1
        trajectory["is_terminal"][-1] = 1
        trajectory["is_first"][0] = 1
        trajectories.append(trajectory)

    log(f"Saving {len(trajectories)} trajectories to TFDS format")
    save_file = Path(cfg.data_dir) / "tensorflow_datasets" / "metaworld" / cfg.env_id
    save_dataset(trajectories, save_file, env_name="metaworld", save_imgs=False)

    # ============ Save a video of one of the trajectories ============
    visualize = False
    if visualize:
        env = env_fn(cfg.env_id, 0, **cfg.env_kwargs)
        env.reset()

        for episode_indx, (obs, act, env_state, done) in tqdm.tqdm(
            enumerate(zip(observations, actions, env_states, dones))
        ):
            if episode_indx != num_episodes - 1:
                continue
            # if episode_indx >= cfg.num_episodes_replay:
            #     break

            frames = []
            episode_return = np.sum(rewards[episode_indx])
            log(
                f"Replaying episode {episode_indx}, length: {len(obs)}, return: {episode_return}"
            )

            for t in range(len(obs)):
                obs_t = obs[t]
                act_t = act[t]
                done_t = done[t]
                reward_t = rewards[t]
                env.set_env_state((env_state[t][0], env_state[t][1]))
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
