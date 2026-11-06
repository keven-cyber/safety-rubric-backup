# Targeted-C dataset

The dataset is included at [`AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120`](../AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120).

- Train: 3,120 rows (600 benign, 600 sensitive, 480 harm, 1,440 adaptive harm).
- Validation: 304 rows (64 benign, 64 sensitive, 64 harm, 112 adaptive harm).
- Version: `updated_training_rubric_dataset_independent_v4_targeted_hard_negative`.

All eight Parquet files are byte-for-byte identical to the original training corpus. `dataset_integrity.json` records their SHA-256 values, sizes and row counts. The released manifest removes the internal source storage path; its released and original hashes are recorded separately. No training/validation examples were regenerated, re-encoded or removed.

Only the required training/validation data is distributed. Generated questions, audit dumps, inference outputs, checkpoints and caches are excluded. Upstream data terms remain applicable; the code license does not relicense third-party datasets.

```bash
cd AlphaAlign-Rubric-end
FILE_PATH="$PWD/Evaluation/rubricalign-targeted-c-v4-3120" python scripts/cross_model_dataset.py --report /tmp/rubricalign_dataset_report.json
```

Use a new report path for each verification run.
