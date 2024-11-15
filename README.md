# Train

```
Train Drawer-Open-v2
python3 train.py mode=train env_id=drawer-open-v2 use_wandb=True hydra/launcher=slurm --multirun debug=True

Multirun over different environments
python3 train.py mode=train env_id=drawer-open-v2,pick-place-v2,peg-insert-side-v2,button-press-v2 use_wandb=True hydra/launcher=slurm --multirun

python3 train.py mode=train env_id=drawer-open-v2,pick-place-v2 use_wandb=True hydra/launcher=slurm --multirun
```

```
Replay buffer

python3 load_sbx_replay_buffer.py ckpt_file_name=sac_e-drawer-open-v2_t-5000000_s-123 ckpt_step=500000 env_id=drawer-open-v2
```

# Collect dataset from trained policy
```
CUDA_VISIBLE_DEVICES=7 python3 collect_dataset_pretrained_policy.py \
    ckpt_file_name="results/2024-10-30-05-36-13/sac_e-pick-place-v2_t-10000000_s-123" \
    ckpt_step=5000000 \
    env_id=button-press-v2 \
    n_eval_episodes=100 \
    save_imgs=True
```