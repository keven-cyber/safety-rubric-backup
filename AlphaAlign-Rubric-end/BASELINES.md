# Baselines

The retained AlphaAlign and NSPO code supports the native method entry points `train_val_alphaalign.sh` and `train_val_nspo.sh`. The current four-model launcher is `scripts/train_cross_model_twelve_p1.sh`.

The shared training dataset is `Evaluation/rubricalign-targeted-c-v4-3120`. Policy snapshots, auxiliary model revisions and data revisions are specified in `scripts/cross_model_twelve_models.json` and `configs/cross_model_twelve_dependency_models.json`. Native baseline rewards, prompts and the NSPO projection procedure remain method-specific.

Provide your own workspace paths, model access and any API credentials through environment variables. This distribution contains no machine-specific account files or credentials. See the root README for environment requirements and entry points.
