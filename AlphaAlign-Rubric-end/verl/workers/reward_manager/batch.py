# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from verl import DataProto
from collections import defaultdict, Counter
import json
import ast




def _normalize_extra_info(extra_info):
    if isinstance(extra_info, dict):
        return dict(extra_info)
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
        except Exception:
            try:
                parsed = ast.literal_eval(extra_info)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {}


def _is_complete_extra_info(info: dict) -> bool:
    required = ["outcome_safety_rule", "outcome_useful_rule", "reasoning_process_rule"]
    for k in required:
        if k not in info:
            return False
        v = info.get(k)
        if v is None:
            return False
        if isinstance(v, (list, tuple, set)):
            if len(v) == 0:
                return False
        elif isinstance(v, str):
            if not v.strip():
                return False
    return True



def _preview_text(s: str, max_chars: int = 2000) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"... [truncated {len(s) - max_chars} chars]"
class BatchRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key='data_source', **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch['prompts']
        response_ids = data.batch['responses']
        attention_mask = data.batch['attention_mask']

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [None] * len(data)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get('extra_info', [None] * len(data))

        uids = data.non_tensor_batch.get('uid', None)
        uid_counts = Counter(list(uids)) if uids is not None else {}
        uid_seen = defaultdict(int)
        uid_to_qidx = {}

        extras_aug = []
        valid_mask = []
        for i in range(len(data)):
            extra_info = extras[i]
            uid = uids[i] if uids is not None else None
            if uid is not None:
                if uid not in uid_to_qidx:
                    uid_to_qidx[uid] = len(uid_to_qidx)
                question_index = uid_to_qidx[uid]
                uid_seen[uid] += 1
                rollout_idx = uid_seen[uid]
                rollout_n = uid_counts.get(uid, None)
            else:
                question_index = i
                rollout_idx = i + 1
                rollout_n = None

            extra_info_dict = _normalize_extra_info(extra_info)
            if not extra_info_dict:
                if extra_info is not None and not isinstance(extra_info, dict):
                    extra_info_dict["_raw_extra_info"] = extra_info

            extra_info_dict["uid"] = uid
            extra_info_dict["question_index"] = question_index
            extra_info_dict["rollout_idx"] = rollout_idx
            if rollout_n is not None:
                extra_info_dict["rollout_n"] = rollout_n

            # Keep original behavior unless evolve mode is explicitly enabled.
            # This guarantees RUBRIC_EVOLVE=0 is backward-compatible.
            evolve_on = os.getenv("RUBRIC_EVOLVE", "0").strip().lower() in {"1", "true", "yes", "on"}
            if evolve_on:
                # merge useful fields from non_tensor_batch into extra_info (rubric rules, ids, prompts)
                merge_keys = [
                    "outcome_safety_rule",
                    "outcome_useful_rule",
                    "reasoning_process_rule",
                    "id",
                    "dataset_label",
                    "data_source",
                    "prompt",
                    "instruction",
                    "instruction_prompt",
                    "question",
                    "user_request",
                ]
                for k in merge_keys:
                    vals = data.non_tensor_batch.get(k, None)
                    if vals is not None:
                        try:
                            extra_info_dict[k] = vals[i]
                        except Exception:
                            pass

            if "index" not in extra_info_dict:
                idx_val = data[i].non_tensor_batch.get('index', None)
                if idx_val is not None:
                    extra_info_dict["index"] = idx_val

            extras_aug.append(extra_info_dict)
            valid_mask.append(_is_complete_extra_info(extra_info_dict))

        scores = self.compute_score(data_sources=data_sources,
                                    solution_strs=responses_str,
                                    ground_truths=ground_truths,
                                    extra_infos=extras_aug,
                                    **self.reward_kwargs)

        return scores, valid_mask

    def __call__(self, data: DataProto, return_dict=False):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch['prompts']
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch['attention_mask']
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        uids = data.non_tensor_batch.get('uid', None)
        uid_counts = Counter(list(uids)) if uids is not None else {}
        uid_seen = defaultdict(int)
        uid_to_qidx = {}

        scores, valid_mask = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    if isinstance(value, (int, float, bool)):
                        reward_extra_info[key].append(value)
                    elif isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, (int, float, bool)):
                                reward_extra_info[f"{key}/{sub_key}"].append(sub_val)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch['responses'][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch['prompts'][i], skip_special_tokens=True)
                ground_truth = None

                uid = uids[i] if uids is not None else None
                if uid is not None:
                    if uid not in uid_to_qidx:
                        uid_to_qidx[uid] = len(uid_to_qidx)
                    question_index = uid_to_qidx[uid]
                    uid_seen[uid] += 1
                    rollout_idx = uid_seen[uid]
                    rollout_n = uid_counts.get(uid, None)
                else:
                    question_index = i
                    rollout_idx = i + 1
                    rollout_n = None

                max_chars = int(os.getenv('REWARD_DEBUG_MAX_CHARS', '2000'))
                print("\n" + "=" * 100)
                print(f"[DEBUG_SAMPLE] data_source={data_source} idx={i} question_index={question_index} rollout={rollout_idx}/{rollout_n}")
                print("[PROMPT_BEGIN]")
                print(_preview_text(prompt_str, max_chars))
                print("[PROMPT_END]")
                print("[RESPONSE_BEGIN]")
                print(_preview_text(response_str, max_chars))
                print("[RESPONSE_END]")
                print("[GROUND_TRUTH]", ground_truth)
                print("[SCORE]", scores[i])
                print("=" * 100 + "\n")
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        # For incomplete extra_info, set reward to group mean so advantage becomes 0
        uids = data.non_tensor_batch.get('uid', None)
        if uids is not None and len(uids) == len(rewards):
            id2vals = defaultdict(list)
            for i, uid in enumerate(uids):
                if valid_mask[i]:
                    id2vals[uid].append(rewards[i])
            id2mean = {uid: (sum(vals) / max(len(vals), 1)) for uid, vals in id2vals.items()}
            for i, uid in enumerate(uids):
                if not valid_mask[i]:
                    mean_val = id2mean.get(uid, 0.0)
                    rewards[i] = mean_val
                    length = valid_response_lengths[i].item()
                    reward_tensor[i, length - 1] = mean_val


        data.batch['acc'] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
