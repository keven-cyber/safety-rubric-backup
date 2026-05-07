# quick start

## ENVIRONMENT

```bash
conda create -n Alpha python=3.10
conda activate Alpha
pip install -r requirements.txt
wandb login
```

## scripts prepare and training

Open `./examples/ppo_trainer/qwen3B-instruct.sh` and fill your checkpoint save dir, and start running qwen3B-instruct.

```bash
bash ./examples/ppo_trainer/qwen3B-instruct.sh
```

## model merge

Open [merge script](./scripts/merge_han.sh), filled in your specific checkpoint path and hf_model_path to get the final model


## More Details

please refer to [Verl Official readme](./VERL_README.md)



# Evaluation

Open `./Evaluation`