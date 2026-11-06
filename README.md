# RubricAlign

Current implementation for **Rubric-Align: Safety Alignment through Dynamically Co-Evolving Rubrics**.

## Repository contents

- `AlphaAlign-Rubric-end/verl/`: training, reward computation and rubric evolution.
- `AlphaAlign-Rubric-end/scripts/`: experiment entry points, dataset preparation and checkpoint handling.
- `AlphaAlign-Rubric-end/AlphaAlign/` and `NSPO/`: baseline implementations.
- `AlphaAlign-Rubric-end/Evaluation/`: safety and general evaluation.
- `data/`: dataset documentation and one consolidated integrity record.
- `RELEASE_PROVENANCE.json`: source version and export-only changes.

This snapshot is based on source commit `b515aaccffe5149a7ac6fb00b54ad8da671eb386` and includes the current TensorBoard implementation and P1 configuration. Earlier experiments used their own recorded source versions. This repository update is not a new verification of paper results.

## Dataset

The current corpus is `rubricalign-targeted-c-v4-3120`, version `updated_training_rubric_dataset_independent_v4_targeted_hard_negative`: **3,120 training / 304 validation examples**. The eight original Parquet files and a manifest with the internal storage path removed are included under `AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120/`; all hashes match the actual training integrity record. See [data/README.md](data/README.md) for verification and usage. Do not substitute older 2,400/2,640-row versions.

## Environment and entry points

Training requires CUDA GPUs and separate policy, judge and guard services. CPU verification does not establish GPU reproducibility. The maintained environment uses Python 3.10+, PyTorch and vLLM; native method scripts specify their respective dependencies. NSPO has separate environment requirements.

```bash
cd AlphaAlign-Rubric-end
python -m pip install -r requirements.txt
python -m pip install -e .
```

Entry points:

- `scripts/train_cross_model_twelve_p1.sh MODEL_METHOD`: four model families (`llama32_3b`, `qwen25_7b`, `qwen3_4b`, `qwen3_8b`) × `p1`, `alphaalign`, `nspo`.
- `scripts/train_rubricevolve_agentic_eight.sh p1_earlytarget`: P1 configuration.
- `train_val.sh`, `train_val_alphaalign.sh`, `train_val_nspo.sh`: method training and final evaluation.
- `scripts/cross_model_dataset.py --report PATH`: resolve the exact corpus via `FILE_PATH` and write an integrity record without overwriting an existing report.

The cross-model launcher retains the original Primus workspace/job environment variables. Adapt them explicitly for another environment; this is not a platform-independent installer. Its exported package index defaults to public PyPI. Download the exact initial model snapshots and judge/guard dependencies from the model registries in `scripts/`; no model weights are included.

P1 uses bounded residual rewards, a global standard-deviation floor of 0.25, gradual weighting, targeted correction from step 180 every two steps, a 0.05 correction cap, presentation factor 0.20, and KL loss disabled. Common settings are 3,120/304 examples, three epochs/1,170 steps, BF16, learning rate 1e-6, batch size eight and eight rollouts. Native baseline method differences are preserved.

## Credentials, provenance and licensing

API credentials must be supplied through environment variables. Hardcoded key fallbacks were removed from the export. Logs, checkpoints, weights and local machine configuration are excluded. The original experimental source remains unchanged. This distribution omits obsolete backups, unrelated harness task definitions and obsolete deployment manifests. The eight training/validation Parquet files are unchanged; only the dataset manifest storage path is redacted.

Original code and available vendored dependency licenses are retained. Third-party dataset terms and model access conditions remain applicable; a code license does not independently relicense data or models.

## Release hygiene

Use `.env.example` to identify configurable endpoints and credentials. Keep real credentials in local environment variables or ignored files. Run `python tools/check_release.py` before committing or pushing. The optional local pre-commit hook can be enabled with `git config core.hooksPath .githooks`. Retained public upstream copyright and license attributions are required and are not private operator information.
