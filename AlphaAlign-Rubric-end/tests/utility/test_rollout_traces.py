import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token) for token in token_ids.tolist())


def test_rollout_trace_writes_decoded_rewarded_trajectory(tmp_path):
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 11, 12, 13], [0, 0, 21, 22]]),
            "responses": torch.tensor([[31, 32, 0], [41, 42, 43]]),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1]]
            ),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "token_level_scores": torch.tensor([[0.0, 0.75, 0.0], [0.0, 0.0, 0.25]]),
            "token_level_rewards": torch.tensor([[0.0, 0.70, 0.0], [0.0, 0.0, 0.20]]),
        },
        non_tensors={
            "question_index": np.array([0, 0], dtype=object),
            "rollout_idx": np.array([1, 2], dtype=object),
            "rollout_n": np.array([2, 2], dtype=object),
            "uid": np.array(["question-1", "question-1"], dtype=object),
            "data_source": np.array(["ordinary_harm", "ordinary_harm"], dtype=object),
            "raw_prompt": np.array([[{"role": "user", "content": "request"}]] * 2, dtype=object),
            "extra_info": np.array([{"sample_id": "a"}, {"sample_id": "a"}], dtype=object),
            "rubric_score_with_unsafe": np.array([0.75, 0.25], dtype=object),
        },
    )

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "rollout_trace": {
                    "enabled": True,
                    "dir": str(tmp_path),
                    "max_samples_per_step": 1,
                }
            }
        }
    )
    trainer.tokenizer = _Tokenizer()
    trainer.global_steps = 7

    trainer._dump_rollout_traces(
        data,
        epoch=2,
        reward_detail_keys=["rubric_score_with_unsafe"],
        reward_trace_info=[{"score": 0.75, "rubric_rule_passes": {"safety": [1, 1]}}],
    )

    trace_path = tmp_path / "step_00000007.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    # The trace limit is soft so a prompt's candidate group is never cut in half.
    assert len(records) == 2
    assert records[0]["global_step"] == 7
    assert records[0]["epoch"] == 2
    assert records[0]["rollout_index"] == 1
    assert records[0]["prompt"] == "11 12 13"
    assert records[0]["response"] == "31 32"
    assert records[0]["reward"] == pytest.approx(0.75)
    assert records[0]["reward_details"]["rubric_score_with_unsafe"] == 0.75
    assert records[0]["reward_result"]["rubric_rule_passes"]["safety"] == [1, 1]
