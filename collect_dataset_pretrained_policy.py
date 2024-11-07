"""
Helper file to collect a dataset of trajectories from a pretrained policy


"""

import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import tqdm
from clam.scripts.data import save_dataset
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from utils.env_utils import env_fn
from utils.logger import log


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    log("start collecting dataset", "blue")

    exp_dir = Path("/scr/aliang80/sbx") / Path(cfg.ckpt_file_name)
    # if not exp_dir.exists():
    #     raise ValueError(f"Experiment directory {exp_dir} does not exist")

    log(f"loading pretrained policy from {exp_dir}", "green")

    policy_cls = PPO if cfg.algo_name == "ppo" else SAC
    ckpt_model = str(exp_dir / cfg.ckpt_dir / f"ckpt_{cfg.ckpt_step}_steps")
    model = policy_cls.load(ckpt_model)

    # collect trajectories
    log("creating environments for data collection", "yellow")

    data_collection_env = env_fn(
        env_id=cfg.env_id,
        env_idx=0,
        save_trajectory=True,
        save_imgs=cfg.save_imgs,
        **cfg.env_kwargs,
    )

    log(f"collecting {cfg.n_eval_episodes} trajectories", "yellow")
    start = time.time()
    evaluate_policy(
        model,
        data_collection_env,
        n_eval_episodes=cfg.n_eval_episodes,
        render=False,
        deterministic=False,
    )
    time_elapsed = time.time() - start
    log(f"collecting {cfg.n_eval_episodes} trajectories took {time_elapsed:.2f}s")

    # extract the saved trajectories
    trajectories = data_collection_env.trajectories

    # write trajectories to dataset
    log(f"Saving to TFDS format, num trajectories: {len(trajectories)}...")

    final_trajectories = []

    for trajectory in tqdm.tqdm(trajectories, desc="episode"):
        # convert to list of trajectories
        new_trajectory = {
            "observations": trajectory["observations"][:-1],  # this was frame-stacked
            "actions": trajectory["actions"],
            "rewards": trajectory["rewards"],
        }
        tmp = new_trajectory["rewards"]
        new_trajectory.update(
            {
                "discount": np.ones_like(tmp),
                "is_last": np.zeros_like(tmp),
                "is_first": np.zeros_like(tmp),
                "is_terminal": np.zeros_like(tmp),
            }
        )

        new_trajectory["is_last"][-1] = 1
        new_trajectory["is_terminal"][-1] = 1
        new_trajectory["is_first"][0] = 1

        if cfg.save_imgs:
            new_trajectory["images"] = trajectory["images"][:-1]

        final_trajectories.append(new_trajectory)

    log(f"Saving {len(final_trajectories)} trajectories to TFDS format")

    if cfg.save_imgs:
        dataset_name = f"{cfg.env_id}_imgs"
    else:
        dataset_name = cfg.env_id
    save_file = Path(cfg.data_dir) / "tensorflow_datasets" / "metaworld" / dataset_name

    save_dataset(
        final_trajectories,
        save_file,
        env_name="metaworld",
        save_imgs=cfg.save_imgs,
        framestack=cfg.env_kwargs.n_frame_stack,
    )

    # visualize one of the trajectories
    # save frames as video
    episode_indx = 0
    frames = final_trajectories[episode_indx]["images"]
    log(f"Replaying episode {episode_indx}, length: {len(frames)}, {frames.shape}")

    video_f = (
        exp_dir / cfg.video_dir / "sample_vid" / f"replay_episode_{episode_indx}.mp4"
    )
    video_f.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving video to {video_f}")
    writer = cv2.VideoWriter(
        str(video_f),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.fps,
        (frames.shape[-2], frames.shape[-3]),
    )
    for img in frames:
        if len(img.shape) == 4:
            img = img[-1]  # with framestack we take the last frame

        # convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)
    writer.release()


if __name__ == "__main__":
    main()
