#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Iterable

import datasets

GSM8K_TEST_SIZE = 1319
MMLU_SUBJECT_COUNT = 57
MMLU_TEST_SIZE = 14042


def datasets_version_tuple() -> tuple[int, ...]:
    parts = []
    for part in datasets.__version__.split("."):
        number = ""
        for char in part:
            if char.isdigit():
                number += char
            else:
                break
        if number:
            parts.append(int(number))
    return tuple(parts)


def validate_dataset_dependencies() -> None:
    version = datasets_version_tuple()
    if not hasattr(datasets, "LargeList"):
        raise RuntimeError(
            f"datasets=={datasets.__version__} does not export LargeList. "
            "Install datasets>=2.21,<4, for example: pip install 'datasets==3.6.0'."
        )
    if version >= (4,):
        raise RuntimeError(
            f"datasets=={datasets.__version__} is incompatible with this ModelScope dataset preparation script. "
            "datasets 4.x no longer supports remote dataset scripts/trust_remote_code in the way GSM8K needs, "
            "and can trigger DatasetBuilder.as_dataset(verification_mode=...) compatibility errors. "
            "Install datasets>=2.21,<4, for example: pip install 'datasets==3.6.0'."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local lm-eval tasks and datasets from ModelScope downloads."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store local benchmark data and generated task YAMLs.",
    )
    parser.add_argument(
        "--harness-dir",
        type=Path,
        required=True,
        help="Path to Evaluation/lm-evaluation-harness.",
    )
    parser.add_argument("--gsm8k-id", type=str, default="")
    parser.add_argument("--mmlu-id", type=str, default="")
    parser.add_argument("--gpqa-id", type=str, default="")
    parser.add_argument("--mmlu-archive", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,mmlu,gpqa",
        help="Comma-separated datasets to prepare: gsm8k,mmlu,gpqa.",
    )
    return parser.parse_args()


def ensure_hf_dataset(obj: Any, split: str | None = None) -> datasets.Dataset:
    if isinstance(obj, datasets.Dataset):
        return obj
    if isinstance(obj, datasets.DatasetDict):
        if split is None:
            for key in ("train", "validation", "test", "dev"):
                if key in obj:
                    return obj[key]
            raise ValueError("DatasetDict has no usable split")
        return obj[split]
    if hasattr(obj, "to_hf_dataset"):
        hf_obj = obj.to_hf_dataset()
        return ensure_hf_dataset(hf_obj, split=split)
    if hasattr(obj, "dataset"):
        return ensure_hf_dataset(obj.dataset, split=split)
    if hasattr(obj, "ds_instance"):
        return ensure_hf_dataset(obj.ds_instance, split=split)
    if isinstance(obj, list):
        return datasets.Dataset.from_list(obj)
    try:
        return datasets.Dataset.from_list(list(obj))
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported ModelScope dataset object: {type(obj)}") from exc


def try_load_modelscope_dataset(
    dataset_ids: list[str],
    subset_candidates: list[str | None],
    split: str | None,
) -> datasets.Dataset:
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        MsDataset = None

    errors: list[str] = []
    for dataset_id in dataset_ids:
        for subset in subset_candidates:
            if MsDataset is not None:
                call_variants = [(dataset_id, {})]
                if "/" in dataset_id:
                    namespace, dataset_name = dataset_id.split("/", 1)
                    call_variants.append((dataset_name, {"namespace": namespace}))

                for dataset_name, extra_kwargs in call_variants:
                    kwargs = dict(extra_kwargs)
                    if subset is not None:
                        kwargs["subset_name"] = subset
                    if split is not None:
                        kwargs["split"] = split
                    kwargs["trust_remote_code"] = True
                    try:
                        obj = MsDataset.load(dataset_name, **kwargs)
                        return ensure_hf_dataset(obj, split=split)
                    except TypeError as exc:
                        if "trust_remote_code" in str(exc):
                            kwargs.pop("trust_remote_code", None)
                            try:
                                obj = MsDataset.load(dataset_name, **kwargs)
                                return ensure_hf_dataset(obj, split=split)
                            except Exception as retry_exc:
                                errors.append(
                                    f"dataset_id={dataset_id}, dataset_name={dataset_name}, subset={subset}, split={split}, err={retry_exc}"
                                )
                        else:
                            errors.append(
                                f"dataset_id={dataset_id}, dataset_name={dataset_name}, subset={subset}, split={split}, err={exc}"
                            )
                    except Exception as exc:
                        errors.append(
                            f"dataset_id={dataset_id}, dataset_name={dataset_name}, subset={subset}, split={split}, err={exc}"
                        )

            hf_kwargs: dict[str, Any] = {
                "path": dataset_id,
                "split": split,
                "trust_remote_code": True,
            }
            if subset is not None:
                hf_kwargs["name"] = subset
            try:
                return ensure_hf_dataset(datasets.load_dataset(**hf_kwargs), split=split)
            except TypeError as exc:
                if "trust_remote_code" in str(exc):
                    hf_kwargs.pop("trust_remote_code", None)
                    try:
                        return ensure_hf_dataset(datasets.load_dataset(**hf_kwargs), split=split)
                    except Exception as retry_exc:
                        errors.append(
                            f"dataset_id={dataset_id}, loader=datasets.load_dataset, subset={subset}, split={split}, err={retry_exc}"
                        )
                else:
                    errors.append(
                        f"dataset_id={dataset_id}, loader=datasets.load_dataset, subset={subset}, split={split}, err={exc}"
                    )
            except Exception as exc:
                errors.append(
                    f"dataset_id={dataset_id}, loader=datasets.load_dataset, subset={subset}, split={split}, err={exc}"
                )

    raise RuntimeError(
        "Failed to load dataset from ModelScope.\n"
        + "\n".join(errors[:20])
        + ("\n..." if len(errors) > 20 else "")
    )


def get_subjects_from_harness(harness_dir: Path) -> list[str]:
    subject_dir = harness_dir / "lm_eval" / "tasks" / "mmlu" / "default"
    subjects = []
    for path in sorted(subject_dir.glob("mmlu_*.yaml")):
        if path.name.startswith("_"):
            continue
        stem = path.stem
        if stem.startswith("mmlu_"):
            subjects.append(stem[len("mmlu_") :])
    if not subjects:
        raise RuntimeError(f"No MMLU subject YAMLs found in {subject_dir}")
    return subjects


def choose_first(record: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def normalize_gsm8k_row(row: dict[str, Any]) -> dict[str, Any]:
    question = choose_first(row, ["question", "Question"])
    answer = choose_first(row, ["answer", "Answer", "target"])
    if question is None or answer is None:
        raise ValueError(f"Unexpected GSM8K row: {row}")
    return {"question": str(question), "answer": str(answer)}


def normalize_mmlu_row(row: dict[str, Any]) -> dict[str, Any]:
    question = choose_first(row, ["question", "Question", "input"])
    choices = choose_first(row, ["choices", "Choices"])
    if not choices:
        choices = [
            choose_first(row, ["A", "a", "choice1", "Choice1"]),
            choose_first(row, ["B", "b", "choice2", "Choice2"]),
            choose_first(row, ["C", "c", "choice3", "Choice3"]),
            choose_first(row, ["D", "d", "choice4", "Choice4"]),
        ]
    choices = list(choices)
    answer = choose_first(row, ["answer", "Answer", "target", "label"])
    if isinstance(answer, str):
        answer = answer.strip()
        if answer.upper() in {"A", "B", "C", "D"}:
            answer = ["A", "B", "C", "D"].index(answer.upper())
        elif answer.isdigit():
            answer = int(answer)
    if question is None or len(choices) != 4 or answer is None:
        raise ValueError(f"Unexpected MMLU row: {row}")
    return {"question": str(question), "choices": [str(x) for x in choices], "answer": int(answer)}


def normalize_gpqa_row(row: dict[str, Any]) -> dict[str, Any]:
    question = choose_first(row, ["Question", "question"])
    if question is None:
        raise ValueError(f"Unexpected GPQA row (missing question): {row}")

    choice1 = choose_first(row, ["choice1"])
    if choice1 is not None:
        choices = [
            choose_first(row, ["choice1"]),
            choose_first(row, ["choice2"]),
            choose_first(row, ["choice3"]),
            choose_first(row, ["choice4"]),
        ]
        answer = choose_first(row, ["answer", "Answer"])
        if answer is None:
            raise ValueError(f"Unexpected GPQA row (missing answer): {row}")
        return {
            "Question": str(question),
            "choice1": str(choices[0]),
            "choice2": str(choices[1]),
            "choice3": str(choices[2]),
            "choice4": str(choices[3]),
            "answer": str(answer),
        }

    correct = choose_first(row, ["Correct Answer", "correct_answer", "answer_text"])
    incorrects = []
    if "incorrect_answers" in row and row["incorrect_answers"]:
        incorrects = list(row["incorrect_answers"])
    else:
        for key in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            value = row.get(key)
            if value not in (None, ""):
                incorrects.append(value)

    if correct is None or len(incorrects) < 3:
        raise ValueError(f"Unexpected GPQA row: {row}")

    choices = [str(incorrects[0]), str(incorrects[1]), str(incorrects[2]), str(correct)]
    rng = random.Random(str(question))
    rng.shuffle(choices)
    correct_idx = choices.index(str(correct))
    return {
        "Question": str(question),
        "choice1": choices[0],
        "choice2": choices[1],
        "choice3": choices[2],
        "choice4": choices[3],
        "answer": f"({chr(65 + correct_idx)})",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def prepare_gsm8k(output_dir: Path, gsm8k_id: str | None) -> Path:
    candidate_ids = [x for x in [gsm8k_id, "gsm8k", "modelscope/gsm8k", "AI-ModelScope/gsm8k"] if x]
    test_ds = try_load_modelscope_dataset(candidate_ids, ["main", None], "test")
    test_rows = [normalize_gsm8k_row(row) for row in test_ds]
    if len(test_rows) != GSM8K_TEST_SIZE:
        raise RuntimeError(
            f"GSM8K test split has {len(test_rows)} rows; expected {GSM8K_TEST_SIZE}"
        )
    test_path = output_dir / "gsm8k" / "test.jsonl"
    write_jsonl(test_path, test_rows)
    return test_path


def prepare_mmlu(output_dir: Path, harness_dir: Path, mmlu_id: str | None) -> list[str]:
    candidate_ids = [x for x in [mmlu_id, "cais/mmlu", "modelscope/mmlu", "AI-ModelScope/mmlu", "mmlu"] if x]
    subjects = get_subjects_from_harness(harness_dir)
    if len(subjects) != MMLU_SUBJECT_COUNT:
        raise RuntimeError(
            f"Found {len(subjects)} MMLU subjects; expected {MMLU_SUBJECT_COUNT}"
        )
    prepared_subjects = []
    total_test_rows = 0
    for subject in subjects:
        dev_ds = try_load_modelscope_dataset(candidate_ids, [subject], "dev")
        test_ds = try_load_modelscope_dataset(candidate_ids, [subject], "test")
        dev_rows = [normalize_mmlu_row(row) for row in dev_ds]
        test_rows = [normalize_mmlu_row(row) for row in test_ds]
        total_test_rows += len(test_rows)
        subject_dir = output_dir / "mmlu" / subject
        write_jsonl(subject_dir / "dev.jsonl", dev_rows)
        write_jsonl(subject_dir / "test.jsonl", test_rows)
        prepared_subjects.append(subject)
    if total_test_rows != MMLU_TEST_SIZE:
        raise RuntimeError(
            f"MMLU test splits contain {total_test_rows} rows; expected {MMLU_TEST_SIZE}"
        )
    return prepared_subjects


def prepare_mmlu_from_archive(
    output_dir: Path, harness_dir: Path, archive_path: Path
) -> list[str]:
    subjects = get_subjects_from_harness(harness_dir)
    if len(subjects) != MMLU_SUBJECT_COUNT:
        raise RuntimeError(
            f"Found {len(subjects)} MMLU subjects; expected {MMLU_SUBJECT_COUNT}"
        )

    total_test_rows = 0
    with tarfile.open(archive_path, "r:*") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for subject in subjects:
            subject_dir = output_dir / "mmlu" / subject
            for source_split, output_split in (("dev", "dev"), ("test", "test")):
                member_name = f"data/{source_split}/{subject}_{source_split}.csv"
                member = members.get(member_name)
                if member is None:
                    raise RuntimeError(f"MMLU archive is missing {member_name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not read {member_name} from {archive_path}")
                reader = csv.reader(io.TextIOWrapper(extracted, encoding="utf-8"))
                rows = []
                for csv_row in reader:
                    if len(csv_row) != 6:
                        raise RuntimeError(
                            f"Unexpected MMLU CSV row in {member_name}: {csv_row}"
                        )
                    rows.append(
                        normalize_mmlu_row(
                            {
                                "question": csv_row[0],
                                "choices": csv_row[1:5],
                                "answer": csv_row[5],
                            }
                        )
                    )
                write_jsonl(subject_dir / f"{output_split}.jsonl", rows)
                if output_split == "test":
                    total_test_rows += len(rows)

    if total_test_rows != MMLU_TEST_SIZE:
        raise RuntimeError(
            f"MMLU archive contains {total_test_rows} test rows; expected {MMLU_TEST_SIZE}"
        )
    return subjects


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_prepared_data(
    output_dir: Path, harness_dir: Path, requested_tasks: set[str]
) -> None:
    if "gsm8k" in requested_tasks:
        gsm8k_path = output_dir / "gsm8k" / "test.jsonl"
        if not gsm8k_path.is_file() or count_jsonl(gsm8k_path) != GSM8K_TEST_SIZE:
            raise RuntimeError(f"Invalid or missing local GSM8K data: {gsm8k_path}")

    if "mmlu" in requested_tasks:
        subjects = get_subjects_from_harness(harness_dir)
        if len(subjects) != MMLU_SUBJECT_COUNT:
            raise RuntimeError(
                f"Found {len(subjects)} MMLU subjects; expected {MMLU_SUBJECT_COUNT}"
            )
        total_test_rows = 0
        for subject in subjects:
            dev_path = output_dir / "mmlu" / subject / "dev.jsonl"
            test_path = output_dir / "mmlu" / subject / "test.jsonl"
            if not dev_path.is_file() or count_jsonl(dev_path) != 5:
                raise RuntimeError(f"Invalid or missing MMLU dev data: {dev_path}")
            if not test_path.is_file():
                raise RuntimeError(f"Missing MMLU test data: {test_path}")
            total_test_rows += count_jsonl(test_path)
        if total_test_rows != MMLU_TEST_SIZE:
            raise RuntimeError(
                f"Local MMLU data contains {total_test_rows} test rows; expected {MMLU_TEST_SIZE}"
            )

    required_task_files = {
        "gsm8k": output_dir / "tasks" / "gsm8k_cot_modelscope.yaml",
        "mmlu": output_dir / "tasks" / "mmlu_modelscope.yaml",
        "gpqa": output_dir / "tasks" / "gpqa_main_cot_zeroshot_modelscope.yaml",
    }
    for task in requested_tasks:
        if not required_task_files[task].is_file():
            raise RuntimeError(f"Missing local task config: {required_task_files[task]}")


def prepare_gpqa(output_dir: Path, gpqa_id: str | None) -> Path:
    candidate_ids = [x for x in [gpqa_id, "gpqa", "Idavidrein/gpqa", "modelscope/gpqa", "AI-ModelScope/gpqa"] if x]
    train_ds = try_load_modelscope_dataset(candidate_ids, ["gpqa_main", "main", None], "train")
    train_rows = [normalize_gpqa_row(row) for row in train_ds]
    train_path = output_dir / "gpqa" / "train.jsonl"
    write_jsonl(train_path, train_rows)
    return train_path


def write_tasks(output_dir: Path, requested_tasks: set[str], mmlu_subjects: list[str]) -> None:
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    if "gsm8k" in requested_tasks:
        gsm8k_test = output_dir / "gsm8k" / "test.jsonl"
        (tasks_dir / "gsm8k_cot_modelscope.yaml").write_text(
            f"""dataset_path: json
dataset_kwargs:
  data_files:
    test: {yaml_quote(str(gsm8k_test))}
task: gsm8k_cot_modelscope
output_type: generate_until
test_split: test
doc_to_target: '{{{{answer.split(''####'')[-1].strip() if answer is defined else target}}}}'
doc_to_text: |
  reasoning step by step and Put your answer in the final,
  Q: {{{{question}}}}

  A:
filter_list:
  - name: "boxed-extract"
    filter:
      - function: "regex"
        regex_pattern: "\\\\boxed{{(\\\\text{{([^}}]+)}}|([^}}]+))}}"
        group_select: -1
        fallback: "[invalid]"
      - function: "take_first"
  - name: strict-match
    filter:
      - function: regex
        regex_pattern: The answer is (\\-?[0-9\\.\\,]+).
      - function: take_first
  - name: flexible-extract
    filter:
      - function: regex
        group_select: -1
        regex_pattern: (-?[$0-9.,]{{2,}})|(-?[0-9]+)
      - function: take_first
generation_kwargs:
  max_gen_toks: 2048
  do_sample: false
  until:
    - 'Q:'
    - </s>
    - <|im_end|>
metric_list:
  - aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    metric: exact_match
    regexes_to_ignore:
      - ','
      - \\$
      - '(?s).*#### '
      - \\.$
repeats: 1
metadata:
  version: 1.0
""",
            encoding="utf-8",
        )

    task_names = []
    for subject in (mmlu_subjects if "mmlu" in requested_tasks else []):
        task_name = f"mmlu_{subject}_modelscope"
        task_names.append(task_name)
        dev_path = output_dir / "mmlu" / subject / "dev.jsonl"
        test_path = output_dir / "mmlu" / subject / "test.jsonl"
        (tasks_dir / f"{task_name}.yaml").write_text(
            f"""task: {task_name}
task_alias: {subject}
dataset_path: json
dataset_kwargs:
  data_files:
    dev: {yaml_quote(str(dev_path))}
    test: {yaml_quote(str(test_path))}
fewshot_split: dev
test_split: test
fewshot_config:
  sampler: first_n
num_fewshot: 5
output_type: multiple_choice
doc_to_text: "{{{{question.strip()}}}}\\nA. {{{{choices[0]}}}}\\nB. {{{{choices[1]}}}}\\nC. {{{{choices[2]}}}}\\nD. {{{{choices[3]}}}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
""",
            encoding="utf-8",
        )

    if "mmlu" in requested_tasks:
        (tasks_dir / "mmlu_modelscope.yaml").write_text(
            "group: mmlu_modelscope\n"
            "task:\n"
            + "".join([f"  - {name}\n" for name in task_names])
            + "aggregate_metric_list:\n"
            + "  - metric: acc\n"
            + "    weight_by_size: True\n"
            + "metadata:\n"
            + "  version: 1.0\n",
            encoding="utf-8",
        )

    if "gpqa" in requested_tasks:
        gpqa_train = output_dir / "gpqa" / "train.jsonl"
        (tasks_dir / "gpqa_main_cot_zeroshot_modelscope.yaml").write_text(
            f"""task: gpqa_main_cot_zeroshot_modelscope
dataset_path: json
dataset_kwargs:
  data_files:
    train: {yaml_quote(str(gpqa_train))}
validation_split: train
output_type: generate_until
doc_to_text: "What is the correct answer to this question:{{{{Question}}}}\\nChoices:\\n(A) {{{{choice1}}}}\\n(B) {{{{choice2}}}}\\n(C) {{{{choice3}}}}\\n(D) {{{{choice4}}}}\\nLet's think step by step, And finally put selction A, B, C or D, wrapped in ()."
doc_to_target: answer
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "(?<=The answer is )(.*)(?=.)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "multi_choice_regex"
        group_select: -1
        ignore_case: true
        ignore_punctuation: true
        regex_pattern: "(\\\\([A-Z]\\\\))"
      - function: "take_first"
generation_kwargs:
  max_gen_toks: 4096
  until:
    - "</s>"
  do_sample: false
  temperature: 0.0
num_fewshot: 0
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
metadata:
  version: 1.0
""",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    validate_dataset_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_tasks = {task.strip().lower() for task in args.tasks.split(",") if task.strip()}
    supported_tasks = {"gsm8k", "mmlu", "gpqa"}
    unknown_tasks = requested_tasks - supported_tasks
    if not requested_tasks:
        raise ValueError("--tasks must contain at least one dataset")
    if unknown_tasks:
        raise ValueError(f"Unsupported datasets in --tasks: {sorted(unknown_tasks)}")

    if args.validate_only:
        local_mmlu_subjects = (
            get_subjects_from_harness(args.harness_dir)
            if "mmlu" in requested_tasks
            else []
        )
        write_tasks(args.output_dir, requested_tasks, local_mmlu_subjects)
        validate_prepared_data(args.output_dir, args.harness_dir, requested_tasks)
        print(f"[OK] Validated local datasets under: {args.output_dir}")
        return

    if "gsm8k" in requested_tasks:
        gsm8k_path = args.output_dir / "gsm8k" / "test.jsonl"
        if gsm8k_path.is_file() and count_jsonl(gsm8k_path) == GSM8K_TEST_SIZE:
            print(f"[OK] Reusing GSM8K at {gsm8k_path}")
        else:
            gsm8k_path = prepare_gsm8k(args.output_dir, args.gsm8k_id or None)
            print(f"[OK] Prepared GSM8K at {gsm8k_path}")

    mmlu_subjects: list[str] = []
    if "mmlu" in requested_tasks:
        if args.mmlu_archive:
            mmlu_subjects = prepare_mmlu_from_archive(
                args.output_dir, args.harness_dir, args.mmlu_archive
            )
        else:
            mmlu_subjects = prepare_mmlu(args.output_dir, args.harness_dir, args.mmlu_id or None)
        print(f"[OK] Prepared MMLU subjects: {len(mmlu_subjects)}")

    if "gpqa" in requested_tasks:
        gpqa_path = prepare_gpqa(args.output_dir, args.gpqa_id or None)
        print(f"[OK] Prepared GPQA at {gpqa_path}")

    write_tasks(args.output_dir, requested_tasks, mmlu_subjects)
    print(f"[OK] Generated local tasks under: {args.output_dir / 'tasks'}")


if __name__ == "__main__":
    main()
