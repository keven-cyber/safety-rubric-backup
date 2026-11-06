import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import pandas as pd
from tqdm import tqdm

from eval_answer import (build_guard_evaluator, extract_answer_content, preserve_generation,
                         score_guard_cases, write_json_atomic, read_json_complete)


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.getenv('SAFETY_RESULT_DIR', os.path.join(BASE_DIR, 'result'))

DATASET_TO_CSV = {
    'xstest': os.path.join(DATA_DIR, 'xstest_safe.csv'),
    'or_bench': os.path.join(DATA_DIR, 'or_bench_hard_1k.csv'),
    'xstest_safe': os.path.join(DATA_DIR, 'xstest_safe.csv'),
    'or_bench_hard_1k': os.path.join(DATA_DIR, 'or_bench_hard_1k.csv'),
}


def load_result_file(result_file: str):
    file_content = read_json_complete(result_file)
    if isinstance(file_content, list):
        return file_content, {}
    if isinstance(file_content, dict):
        return file_content.get('data', []), file_content.get('metadata', {})
    raise ValueError(f'Unexpected result format in {result_file}')


def load_eval_dataset(dataset_name: str, dataset_csv: Optional[str] = None) -> pd.DataFrame:
    if dataset_csv is None:
        if dataset_name not in DATASET_TO_CSV:
            raise ValueError(f'Unknown dataset_name={dataset_name}. Known datasets: {sorted(DATASET_TO_CSV)}')
        dataset_csv = DATASET_TO_CSV[dataset_name]
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f'Dataset CSV not found: {dataset_csv}')

    df = pd.read_csv(dataset_csv)
    if 'prompt' not in df.columns:
        if 'question' in df.columns:
            df = df.rename(columns={'question': 'prompt'})
        else:
            raise ValueError(f'Dataset CSV must contain a prompt/question column: {dataset_csv}')
    return df


def align_dataset_with_results(dataset_df: pd.DataFrame, result_data: list[dict]) -> tuple[list[dict], int]:
    prompt_to_items = defaultdict(list)
    for item in result_data:
        prompt_to_items[str(item.get('prompt', ''))].append(item)

    matched = []
    missing = 0
    for _, row in dataset_df.iterrows():
        prompt = str(row['prompt'])
        bucket = prompt_to_items.get(prompt, [])
        if bucket:
            matched.append(bucket.pop(0))
        else:
            missing += 1
    return matched, missing


def evaluate_overrefusal(
    model_id: str,
    dataset_name: str,
    dataset_csv: Optional[str] = None,
    result_file: Optional[str] = None,
    require_complete: bool = False,
):
    if result_file is None:
        result_file = os.path.join(RESULT_DIR, dataset_name, f'{model_id}.json')

    if not os.path.exists(result_file):
        raise FileNotFoundError(f'Result file not found: {result_file}')

    dataset_df = load_eval_dataset(dataset_name, dataset_csv=dataset_csv)
    result_data, existing_metadata = load_result_file(result_file)
    preserve_generation(result_file, {'metadata': existing_metadata, 'data': result_data})
    matched_items, missing_count = align_dataset_with_results(dataset_df, result_data)

    if not matched_items:
        raise ValueError(
            f'No matched prompts found between dataset {dataset_name} and result file {result_file}. '
            f'Please ensure the generated result uses the same prompts as the evaluation dataset.'
        )
    if require_complete and (missing_count or len(result_data) != len(dataset_df)):
        raise RuntimeError(
            f'Incomplete {dataset_name} generation: matched {len(matched_items)} of '
            f'{len(dataset_df)} prompts ({missing_count} missing).'
        )

    guard = build_guard_evaluator()

    parsed = [extract_answer_content(item.get('response', '')) for item in matched_items]
    final_answers = [x[0] for x in parsed]
    has_answer_tag = [x[1] for x in parsed]

    batch_size = 32
    all_results = []
    for i in tqdm(range(0, len(matched_items), batch_size), desc=f'Evaluating {dataset_name}'):
        batch_items = matched_items[i:i + batch_size]
        batch_answers = final_answers[i:i + batch_size]
        batch_conversations = [
            {'prompt': item.get('prompt', ''), 'response': answer}
            for item, answer in zip(batch_items, batch_answers)
        ]
        all_results.extend(score_guard_cases(guard, batch_conversations))

    valid_refusal_judgments = 0
    refusal_count = 0
    none_count = 0
    safety_label_counts = defaultdict(int)
    refusal_label_counts = defaultdict(int)

    for item, parsed_answer, answer_tag, result in zip(matched_items, final_answers, has_answer_tag, all_results):
        refusal_label = result.get('refusal_label')
        safety_label = result.get('safety_label')
        if safety_label is not None:
            safety_label_counts[safety_label] += 1
        if refusal_label is not None:
            refusal_label_counts[refusal_label] += 1

        included = refusal_label in {'yes', 'no'}
        if refusal_label is None:
            none_count += 1
        if included:
            valid_refusal_judgments += 1
            if refusal_label == 'yes':
                refusal_count += 1

        item['parsed_answer_for_overrefusal'] = parsed_answer
        item['has_answer_tag'] = bool(answer_tag)
        item['guard_refusal_label'] = refusal_label
        item['guard_safety_label'] = safety_label
        item['guard_risk_categories'] = result.get('risk_categories', [])
        item['guard_overrefusal_eval_included'] = included
        item['llamaguard_response'] = result.get('llamaguard_response', '')
        if 'guard_error' in result:
            item['guard_error'] = result['guard_error']

    valid_only_rate = (
        refusal_count / valid_refusal_judgments * 100.0
        if valid_refusal_judgments > 0 else None
    )
    overrefusal_rate = valid_only_rate if none_count == 0 else None

    stats = {
        'dataset': dataset_name,
        'dataset_csv': dataset_csv or DATASET_TO_CSV.get(dataset_name),
        'total_dataset_prompts': int(len(dataset_df)),
        'matched_prompts': int(len(matched_items)),
        'missing_prompts_in_results': int(missing_count),
        'valid_refusal_judgments': int(valid_refusal_judgments),
        'ignored_none_judgments': 0,
        'invalid_refusal_judgments': int(none_count),
        'evaluation_complete': missing_count == 0 and none_count == 0 and len(result_data) == len(dataset_df),
        'refusal_count': int(refusal_count),
        'overrefusal_rate': overrefusal_rate,
        'diagnostic_valid_only_refusal_rate': valid_only_rate,
        'refusal_rate_lower_bound': refusal_count / len(dataset_df) * 100,
        'refusal_rate_upper_bound': (refusal_count + none_count + missing_count) / len(dataset_df) * 100,
        'with_answer_tag': int(sum(has_answer_tag)),
        'without_answer_tag': int(len(has_answer_tag) - sum(has_answer_tag)),
        'guard_safety_label_counts': dict(safety_label_counts),
        'guard_refusal_label_counts': dict(refusal_label_counts),
    }

    output_key = f'{dataset_name}_overrefusal_statistics'
    output_data = {
        'metadata': {**existing_metadata, output_key: stats},
        'data': result_data,
    }
    write_json_atomic(result_file, output_data)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if require_complete and valid_refusal_judgments != len(matched_items):
        raise RuntimeError(
            f'Incomplete {dataset_name} refusal judgments: received '
            f'{valid_refusal_judgments} valid labels for {len(matched_items)} prompts.'
        )
    if overrefusal_rate is None:
        print(f'{dataset_name} Refusal Rate: N/A')
    else:
        print(f'{dataset_name} Refusal Rate: {overrefusal_rate:.2f}%')
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate over-refusal on safe/benign datasets using the same guard backend as eval_answer.py.'
    )
    parser.add_argument('--model_id', type=str, required=True, help='Model id / result filename stem.')
    parser.add_argument(
        '--dataset_name',
        type=str,
        nargs='+',
        default=['xstest', 'or_bench'],
        help='Dataset name(s) to evaluate. Defaults to xstest and or_bench.',
    )
    parser.add_argument('--dataset_csv', type=str, default=None, help='Optional explicit dataset CSV path (single-dataset mode only).')
    parser.add_argument('--result_file', type=str, default=None, help='Optional explicit result JSON path (single-dataset mode only).')
    parser.add_argument(
        '--require-complete',
        action='store_true',
        help='Fail if any dataset prompt or valid refusal judgment is missing.',
    )
    args = parser.parse_args()

    if args.dataset_csv is not None and len(args.dataset_name) != 1:
        raise ValueError('--dataset_csv can only be used when exactly one dataset_name is provided.')
    if args.result_file is not None and len(args.dataset_name) != 1:
        raise ValueError('--result_file can only be used when exactly one dataset_name is provided.')

    for dataset_name in args.dataset_name:
        print(f'\nEvaluating over-refusal for dataset: {dataset_name}')
        evaluate_overrefusal(
            model_id=args.model_id,
            dataset_name=dataset_name,
            dataset_csv=args.dataset_csv,
            result_file=args.result_file,
            require_complete=args.require_complete,
        )


if __name__ == '__main__':
    main()
