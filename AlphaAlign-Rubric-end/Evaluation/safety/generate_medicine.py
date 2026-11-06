"""Generate model responses for medicine test JSON or CSV files."""

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SAFETY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SAFETY_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Evaluation" / "rubric_output_new_medicine" / "gpt4"
DEFAULT_OUTPUT_DIR = SAFETY_DIR / "result_medicine"


def _load_json_array_with_salvage(path: Path):
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj["data"]
        return []
    except Exception:
        pass

    start = text.find("[")
    if start < 0:
        return []

    decoder = json.JSONDecoder()
    i = start + 1
    out = []
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    print(f"[WARN] recovered partial JSON from {path}: valid_items={len(out)}")
    return out


def _load_csv_records(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    items = []
    for idx, row in enumerate(rows):
        question = str(
            row.get("harmful_medical_request")
            or row.get("question")
            or row.get("prompt")
            or ""
        ).strip()
        if not question:
            continue

        safe_response = str(row.get("safe_response") or "").strip()
        category = path.stem.replace("med_safety_demonstrations_", "")
        item = {
            "question": question,
            "prompt": question,
            "extra_info": {
                "split": "test",
                "index": idx,
                "question": question,
                "category": category,
                "source_csv": path.name,
                "reference_safe_response": safe_response,
            },
        }
        if safe_response:
            item["reference_safe_response"] = safe_response
        items.append(item)
    return items


def _discover_test_files(input_dir: str):
    p = Path(input_dir)
    files = sorted(p.rglob("*test.json"))
    if files:
        return files, False, "json"

    fallback = sorted(
        f for f in p.rglob("*.json")
        if not f.name.endswith(".safety_judged.json")
    )
    if fallback:
        print("[WARN] 未找到 *test.json，将从 .json 自动抽取测试子集（默认 20%）。")
        return fallback, True, "json"

    csv_files = sorted(p.rglob("*.csv"))
    if csv_files:
        print("[WARN] 未找到可用 JSON，回退到 CSV 输入。")
        return csv_files, False, "csv"

    return [], False, "json"


def _prompt_messages(item):
    prompt = item.get("prompt")
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return prompt
    if isinstance(prompt, dict):
        return [{"role": prompt.get("role", "user"), "content": str(prompt.get("content", ""))}]
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]

    q = item.get("question")
    if isinstance(q, str) and q.strip():
        return [{"role": "user", "content": q.strip()}]

    extra_q = (item.get("extra_info") or {}).get("question", "")
    return [{"role": "user", "content": str(extra_q).strip()}]


def _select_test_items(items, ratio=0.2):
    with_test_split = [it for it in items if str((it.get("extra_info") or {}).get("split", "")).lower() == "test"]
    if with_test_split:
        return with_test_split

    with_val_split = [it for it in items if str((it.get("extra_info") or {}).get("split", "")).lower() == "val"]
    if with_val_split:
        return with_val_split

    selected = []
    for idx, it in enumerate(items):
        extra = it.get("extra_info") or {}
        key = str(extra.get("sample_id") or extra.get("index") or it.get("question") or idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        if (h % 1000) < int(ratio * 1000):
            selected.append(it)

    if not selected and items:
        selected = items[: max(1, int(len(items) * ratio))]
    return selected


def _batch_generate(inputs, llm, batch_size=128, max_new_tokens=2048, top_p=0.9, temperature=0.6):
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        stop=["<|im_end|>"],
    )

    outputs = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating", leave=False):
        batch = inputs[i:i + batch_size]
        results = llm.generate(batch, sampling_params)
        for r in results:
            outputs.append(r.outputs[0].text)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate medicine test responses via vLLM")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--memory_utilization", type=float, default=0.8)
    parser.add_argument("--auto_test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model_name = os.path.basename(args.model_id.rstrip("/"))
    output_root = Path(args.output_dir) / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    files, auto_select_mode, input_kind = _discover_test_files(args.input_dir)
    if not files:
        print(f"[WARN] No supported input file found in {args.input_dir}")
        return

    print(f"[INFO] model={args.model_id}")
    print(f"[INFO] found {len(files)} {input_kind} input file(s)")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        device="cuda",
        gpu_memory_utilization=args.memory_utilization,
        max_model_len=2048,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    try:
        for fp in files:
            if input_kind == "csv":
                items = _load_csv_records(fp)
            else:
                items = _load_json_array_with_salvage(fp)
            if not items:
                print(f"[WARN] empty/invalid: {fp}")
                continue

            if auto_select_mode:
                before = len(items)
                items = _select_test_items(items, ratio=args.auto_test_ratio)
                print(f"[INFO] auto-select test subset for {fp.name}: {len(items)}/{before}")

            chat_msgs = [_prompt_messages(it) for it in items]
            inputs = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in chat_msgs
            ]

            responses = _batch_generate(
                inputs,
                llm,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
            )

            out_items = []
            for item, resp in zip(items, responses):
                obj = dict(item)
                obj["response"] = resp
                obj["model_name"] = model_name
                out_items.append(obj)

            if input_kind == "csv":
                base_name = fp.stem + "_test.json"
            else:
                base_name = fp.name
            if input_kind != "csv" and auto_select_mode and "test" not in base_name:
                base_name = fp.stem + "_auto_test.json"

            out_path = output_root / base_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out_items, f, ensure_ascii=False, indent=2)
            print(f"[DONE] {fp.name} -> {out_path}")

    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)


if __name__ == "__main__":
    main()
