# Rubric-Align

Code and data for **Rubric-Align: Safety Alignment through Dynamically Co-Evolving Rubrics**.

Rubric-Align trains language models with prompt-specific, natural-language rubrics. The rubrics provide criterion-level rewards and evolve during training. This repository includes rubric construction, policy training, AlphaAlign and NSPO baselines, and safety and general-capability evaluation.

## Installation

Training requires CUDA GPUs, PyTorch, vLLM, and separate judge services. Install the main package with:

```bash
cd AlphaAlign-Rubric-end
python -m pip install -r requirements.txt
python -m pip install -e .
```

The method-specific launch scripts specify additional dependencies. NSPO uses a separate environment. Configure service URLs and API keys through environment variables; see [.env.example](.env.example). Model weights must be downloaded separately.

## Dataset

The targeted-C dataset contains **3,120 training examples** and **304 validation examples**:

| Split | Benign | Sensitive | Harmful | Adaptive harmful | Total |
|---|---:|---:|---:|---:|---:|
| Train | 600 | 600 | 480 | 1,440 | 3,120 |
| Validation | 64 | 64 | 64 | 112 | 304 |

The eight Parquet files are in [Evaluation/rubricalign-targeted-c-v4-3120](AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120). See [data/README.md](data/README.md) for the file layout and integrity checks.

## Training and evaluation

The main entry points are:

| Method | Entry point |
|---|---|
| Rubric-Align | `AlphaAlign-Rubric-end/train_val.sh` |
| AlphaAlign | `AlphaAlign-Rubric-end/train_val_alphaalign.sh` |
| NSPO | `AlphaAlign-Rubric-end/train_val_nspo.sh` |

The P1 configuration is selected with:

```bash
bash scripts/train_rubricevolve_agentic_eight.sh p1_earlytarget
```

P1 uses a standard-deviation floor of 0.25, gradual reward weighting, targeted correction from step 180 every two steps, a correction cap of 0.05, and no KL loss. Shared training settings are three epochs, BF16, learning rate 1e-6, batch size eight, and eight rollouts per prompt.

The cross-model scripts support Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct, Qwen3-4B, and Qwen3-8B. Launch scripts use Primus workspace and job variables; adjust paths and service settings for your environment. Evaluation covers safety benchmarks, XSTest, MMLU, and GSM8K.

## Repository layout

- `AlphaAlign-Rubric-end/verl/`: policy training, rubric rewards, and evolution.
- `AlphaAlign-Rubric-end/scripts/`: training launchers and data utilities.
- `AlphaAlign-Rubric-end/AlphaAlign/` and `NSPO/`: baseline implementations.
- `AlphaAlign-Rubric-end/Evaluation/`: evaluation code and targeted-C data.

## License

See [LICENSE](AlphaAlign-Rubric-end/LICENSE). Vendored dependencies retain their own license notices. Dataset and model terms apply separately.
