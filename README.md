# Train

```
Train Drawer-Open-v2
python3 run_sbx.py mode=train env_id=drawer-open-v2 use_wandb=True hydra/launcher=slurm --multirun debug=True

Multirun over different environments
python3 run_sbx.py mode=train env_id=drawer-open-v2,pick-place-v2,peg-insert-v2,button-push-v2 use_wandb=True hydra/launcher=slurm --multirun
```

