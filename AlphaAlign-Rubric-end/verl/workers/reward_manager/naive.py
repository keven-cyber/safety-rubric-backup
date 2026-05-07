# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict, Counter
import json
import ast


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # route general to reward model
        # data.batch['rm_scores'] shape will be like response
        # if 'rm_scores' in data.batch.keys():
        #     if return_dict:
        #         return {"reward_tensor": data.batch['rm_scores']}
        #     else:
        #         return data.batch['rm_scores']
        rm_scores = [0] * len(data)
        if 'rm_scores' in data.batch.keys():
            max_values, max_indices = torch.max(data.batch['rm_scores'], dim=-1)
            rm_scores = max_values.detach().cpu().tolist()

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        uids = data.non_tensor_batch.get('uid', None)
        uid_counts = Counter(list(uids)) if uids is not None else {}
        uid_seen = defaultdict(int)
        uid_to_qidx = {}

        already_print_data_sources = {}
        warned_missing_reward_model = False

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch.get('reward_model', None)
            if reward_model is None and not warned_missing_reward_model:
                print("Warning: 'reward_model' not found in non_tensor_batch; ground_truth set to None.")
                warned_missing_reward_model = True
            if isinstance(reward_model, dict):
                ground_truth = reward_model.get('ground_truth', None)
            else:
                # allow reward_model to be directly the ground_truth or other formats
                ground_truth = reward_model

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            uid = data_item.non_tensor_batch.get('uid', None)
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

            extra_info_dict = None
            if isinstance(extra_info, dict):
                extra_info_dict = dict(extra_info)
            elif isinstance(extra_info, str):
                try:
                    parsed = json.loads(extra_info)
                except Exception:
                    try:
                        parsed = ast.literal_eval(extra_info)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    extra_info_dict = parsed

            if extra_info_dict is None:
                extra_info_dict = {}
                if extra_info is not None and not isinstance(extra_info, dict):
                    extra_info_dict["_raw_extra_info"] = extra_info

            # attach rollout metadata for logging
            extra_info_dict["uid"] = uid
            extra_info_dict["question_index"] = question_index
            extra_info_dict["rollout_idx"] = rollout_idx
            if rollout_n is not None:
                extra_info_dict["rollout_n"] = rollout_n
            if "index" not in extra_info_dict:
                idx_val = data_item.non_tensor_batch.get('index', None)
                if idx_val is not None:
                    extra_info_dict["index"] = idx_val

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info_dict,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store numeric metrics; expand dict metrics for logging/validation
                for key, value in score.items():
                    if isinstance(value, (int, float, bool)):
                        reward_extra_info[key].append(value)
                    elif isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, (int, float, bool)):
                                reward_extra_info[f"{key}/{sub_key}"].append(sub_val)
                reward_extra_info['rm_scores'].append(rm_scores[i])
            else:
                reward = score

            if ground_truth is not None:
                unsafe_flag = False
                try:
                    if isinstance(ground_truth, (list, tuple, set)):
                        unsafe_flag = 'unsafe' in ground_truth
                    elif isinstance(ground_truth, dict):
                        unsafe_flag = ('unsafe' in ground_truth) or bool(ground_truth.get('unsafe', False))
                    else:
                        unsafe_flag = 'unsafe' in str(ground_truth)
                except Exception:
                    unsafe_flag = False

                if not unsafe_flag:
                    if reward > 0:
                        if rm_scores[i] > reward:
                            reward = rm_scores[i]
                        print(f'safe response. reward: {reward}')
                    else:
                        reward = reward
                        print(f'unsafe response. reward: {reward}')

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
