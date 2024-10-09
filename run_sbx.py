import sys
from functools import partial
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig
from sbx import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
from wandb.integration.sb3 import WandbCallback

from utils.callbacks import (
    SaveEvalVideoCallback,
)
from utils.env_utils import env_fn
from utils.general_utils import omegaconf_to_dict, print_dict
from utils.logger import log


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    log("start")
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # vec_env_cls = SubprocVecEnv if cfg.n_train_envs > 1 else DummyVecEnv
    if cfg.debug:
        log("RUNNING IN DEBUG MODE", "red")
        # set some debug config values
        cfg.total_timesteps = 20
        cfg.num_evals = 1
        cfg.num_save_ckpts = 1
        cfg.eval_freq = -1
        cfg.save_ckpt_freq = -1
        cfg.use_wandb = False
        cfg.n_train_envs = 1
        cfg.n_eval_envs = 1

        vec_env_cls = DummyVecEnv
        vec_env_kwargs = {}
    else:
        vec_env_cls = SubprocVecEnv
        vec_env_kwargs = {"start_method": "fork"}

    train_envs = vec_env_cls(
        [
            partial(env_fn, env_id=cfg.env_id, env_idx=i, **cfg.env_kwargs)
            for i in range(cfg.n_train_envs)
        ],
        **vec_env_kwargs,
    )
    eval_envs = vec_env_cls(
        [
            partial(env_fn, env_id=cfg.env_id, env_idx=i + 10000, **cfg.env_kwargs)
            for i in range(cfg.n_eval_envs)
        ],
        **vec_env_kwargs,
    )

    exp_dir = Path(cfg.results_dir) / cfg.exp_name
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)

    ckpt_dir = str(exp_dir / cfg.ckpt_dir)
    video_dir = str(exp_dir / cfg.video_dir)
    log_dir = str(exp_dir / cfg.log_dir)
    eval_dir = str(exp_dir / cfg.eval_dir)

    # initialize SBX policy
    log("initializing policy...")

    policy_cls = PPO if cfg.algo_name == "ppo" else SAC

    model = policy_cls(
        "MlpPolicy",
        train_envs,
        verbose=1,
        device="cuda",
        **cfg.algo,
        # train_freq=(1, "episode"),  # update every episode
    )

    if cfg.mode == "train":
        log(f"Training envs: {train_envs}")
        log(f"Evaluation envs: {eval_envs}")

        callback_list = []
        if cfg.use_wandb:
            wandb_run = wandb.init(
                **cfg.wandb,
                name=cfg.exp_name,
                config=omegaconf_to_dict(cfg),
            )
            wandb_callback = WandbCallback(
                gradient_save_freq=cfg.gradient_save_freq,
                model_save_path=ckpt_dir,
                verbose=2,
            )
            callback_list.append(wandb_callback)

        if cfg.eval_freq != -1 or cfg.num_evals != -1:
            if cfg.num_evals != -1:
                eval_freq = max(
                    cfg.total_timesteps // cfg.num_evals // cfg.n_train_envs, 1
                )
            else:
                eval_freq = cfg.eval_freq

            log(f"Evaluating every {eval_freq} timesteps")

            eval_callback = EvalCallback(
                eval_envs,
                best_model_save_path=eval_dir,
                log_path=eval_dir,
                eval_freq=eval_freq,
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
                render=False,
            )

            callback_list.append(eval_callback)
        if cfg.save_video_freq != -1:
            save_video_callback = SaveEvalVideoCallback(
                env_fn=env_fn,
                save_freq=max(cfg.save_video_freq // cfg.n_train_envs, 1),
                video_log_dir=video_dir,
                n_eval_episodes=cfg.n_eval_episodes,
            )

            callback_list.append(save_video_callback)

        # for saving checkpoints
        if cfg.save_ckpt_freq != -1 or cfg.num_save_ckpts != -1:
            if cfg.num_save_ckpts != -1:
                save_freq = max(
                    cfg.total_timesteps // cfg.num_save_ckpts // cfg.n_train_envs, 1
                )
            else:
                save_freq = cfg.save_ckpt_freq

            log(f"Saving checkpoint every: {save_freq}")
            ckpt_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=ckpt_dir,
                name_prefix=cfg.ckpt_prefix,
                save_replay_buffer=cfg.save_replay_buffer,
                verbose=2,
            )
            callback_list.append(ckpt_callback)

        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        log(f"callback_list: {callback_list}")
        log(f"start training for {cfg.env_id}, num timesteps: {cfg.total_timesteps}")

        model.learn(
            total_timesteps=cfg.total_timesteps,
            progress_bar=cfg.progress_bar,
            callback=callback_list,
        )

        log("finished training, saving model one last time...")
    elif cfg.mode == "eval":
        log(f"running evaluation... load model from {cfg.ckpt_file_name}")

        exp_dir = Path(cfg.results_dir) / cfg.ckpt_file_name
        model_ckpt_file = exp_dir / "ckpts" / f"ckpt_{cfg.ckpt_step}_steps.zip"
        model.load(model_ckpt_file)

        # evaluate the policy
        log("evaluating policy...")

        eval_env = DummyVecEnv([partial(env_fn, env_id=cfg.env_id, env_idx=0)])

        video_log_dir = str(exp_dir / cfg.video_dir)
        if not Path(video_log_dir).exists():
            Path(video_log_dir).mkdir(parents=True)

        eval_env = VecVideoRecorder(
            eval_env,
            video_log_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=500,
            name_prefix="eval_video",
        )

        mean_rew, std_rew = evaluate_policy(
            model, eval_env, n_eval_episodes=cfg.n_eval_episodes, render=False
        )
        log(f"mean reward: {mean_rew}, std reward: {std_rew}")

    # end program
    log("end")
    sys.exit(0)


if __name__ == "__main__":
    main()
