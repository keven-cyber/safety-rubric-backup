# Targeted-C dataset

The original dataset is included at [`AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120`](../AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120).

- Name: `rubricalign-targeted-c-v4-3120`
- Version: `updated_training_rubric_dataset_independent_v4_targeted_hard_negative`
- Train: 3,120 rows (600 benign, 600 sensitive, 480 harm, 1,440 adaptive harm).
- Validation: 304 rows (64 benign, 64 sensitive, 64 harm, 112 adaptive harm).

All eight original Parquet files and the original `manifest.json` were downloaded from the historical targeted-C run's `revised_dataset` directory on 2026-11-06. Their byte sizes and SHA-256 values match the dataset integrity record captured from actual subsequent training. No data was regenerated, re-encoded or replaced with an older version. The manifest bytes are unchanged.

`expected_dataset_integrity.json` records the original expected values. `dataset_verification.json` records verification of the included files. Upstream dataset terms remain applicable; no new blanket dataset license is asserted.

The maintained launchers recognize this directory automatically. To resolve it explicitly:

```bash
cd AlphaAlign-Rubric-end
FILE_PATH="$PWD/Evaluation/rubricalign-targeted-c-v4-3120" python scripts/cross_model_dataset.py --report /tmp/rubricalign_dataset_report.json
```

The report path must not already exist.
