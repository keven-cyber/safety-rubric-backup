"""Contain unavailable Judge validation without inventing a score or a best model.

The caller still attempts each scheduled validation. An unavailable batch makes
that whole validation incomplete. Its frozen generations and the corresponding
checkpoint remain outstanding evidence, and final best selection refuses gaps.
This is not a generic exception handler, training-batch skip, or service repair.
"""
import json
import os
from pathlib import Path
import traceback
import uuid


UNAVAILABLE = object()


def enabled():
    return os.getenv("RUBRIC_VALIDATION_QUARANTINE_ENABLE", "0") == "1"


class JudgeScoringUnavailable(RuntimeError):
    """Only an explicitly unavailable Judge result may use this path."""
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def write(path, value):
    # Unknown objects, nonfinite numbers and storage failures remain fatal.
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def score_batch(score, snapshot, run_dir, checkpoint_root, step, batch_index, *, diagnostic_only=False):
    """Archive before scoring; return UNAVAILABLE only for typed unavailable scoring.

    A complete return is exactly the original object, with no score conversion.
    No failed response receives a valid scalar, label, or aggregate metric.
    """
    if not enabled():
        return score()
    if not run_dir or not checkpoint_root:
        raise ValueError("Validation containment requires persistent evidence and checkpoint roots")
    run = Path(run_dir)
    frozen_checkpoint = Path(checkpoint_root) / f"global_step_{step}"
    # The mandatory preflight exercises a synthetic manager fault before any
    # frozen actor is loaded. It has no policy, dataset rows or model calls.
    # Real validation snapshots must always retain their frozen-policy binding.
    if type(diagnostic_only) is not bool or (diagnostic_only and snapshot not in
            ({'diagnostic_only': True, 'model_calls': 0}, {'diagnostic_only': True})):
        raise ValueError('Only the exact zero-model-call diagnostic may omit a frozen policy')
    if os.getenv('RUBRIC_FROZEN_REVIEW_ENABLE', '0') == '1' and not diagnostic_only:
        from verl.utils.frozen_validation_review import AUTHORIZATION, actor_files
        if os.getenv('RUBRIC_FROZEN_REVIEW_AUTHORIZATION') != AUTHORIZATION:
            raise ValueError('Unauthorized new frozen-policy validation')
        frozen_checkpoint = Path(os.environ['RUBRIC_VALIDATION_FROZEN_POLICY_CHECKPOINT'])
        if frozen_checkpoint.is_symlink() or not frozen_checkpoint.is_absolute() or not frozen_checkpoint.name.endswith('_' + str(step)):
            raise ValueError('New review frozen policy binding is invalid')
        actor_files(frozen_checkpoint)
    name = f"step_{step:08d}_batch_{batch_index:05d}_{uuid.uuid4().hex}.json"
    generation = run / "validation_pending" / "generations" / name
    write(generation, dict(global_step=step, batch_index=batch_index,
                          scoring_status="pending", snapshot=snapshot))
    try:
        return score()
    except JudgeScoringUnavailable as error:
        event = dict(global_step=step, batch_index=batch_index,
            status="incomplete", generated_batch=str(generation),
            checkpoint=str(frozen_checkpoint),
            checkpoint_save_still_required=True, judge_errors=str(run / "judge_errors"),
            judge_quarantine=str(run / "judge_quarantine"),
            error_type=type(error).__name__, message=str(error),
            scoring_evidence=error.evidence, traceback=traceback.format_exc(),
            aggregate_reward=None, best_eligible=False, final_selection_blocked=True,
            dataset_modified=False, requires_frozen_checkpoint_validation_completion=True)
        write(run / "validation_pending" / name, event)
        # One immutable event per unavailable batch. Never silently clear a gap.
        write(Path(checkpoint_root) / "validation_pending" / name, event)
        print(f"[VALIDATION_UNAVAILABLE] step={step} batch={batch_index}; "
              "no aggregate or best metric; checkpoint and training continue", flush=True)
        return UNAVAILABLE


def require_complete_best(root):
    if (Path(root) / "scoring_protocol_transition.json").exists() or list((Path(root) / "protocol_history").glob("*.json")):
        raise ValueError("Final best requires same-protocol chronological validation review after scoring transition")
    pending = sorted((Path(root) / "validation_pending").glob("*.json"))
    if pending:
        raise ValueError("Best checkpoint selection is incomplete: unresolved frozen validation evidence: "
                         + ", ".join(p.name for p in pending))


def check_runtime_contract(output):
    """Exercise the real manager -> containment boundary before actor training.

    Fault injection only: no Judge/model calls, real dataset or optimizer use.
    Its diagnostic pending marker is confined to this preflight subdirectory.
    """
    if not enabled():
        raise ValueError("Validation containment is not enabled in this process")
    from types import SimpleNamespace
    import numpy as np
    import torch
    from verl import DataProto
    from verl.workers.reward_manager.batch import BatchRewardManager
    output = Path(output)
    rng = torch.get_rng_state().clone()
    data = DataProto.from_dict(tensors=dict(prompts=torch.ones(1, 2, dtype=torch.long),
        responses=torch.ones(1, 2, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long)),
        non_tensors=dict(data_source=np.array(['preflight_fault_injection'], dtype=object)))
    failed = [dict(score=-.25, judge_failed=True, judge_parse_valid=True,
                   rubric_rule_passes=dict(outcome_useful_rule=[1]))]
    manager = BatchRewardManager(SimpleNamespace(decode=lambda *a, **kw: 'Preflight only'), 0, None)
    manager.verify = lambda batch: (failed, [True])
    result = score_batch(lambda: manager(data, return_dict=True),
                         dict(diagnostic_only=True, model_calls=0),
                         output / 'run', output / 'checkpoints', 0, 0, diagnostic_only=True)
    if result is not UNAVAILABLE:
        raise ValueError("The real evaluation manager did not contain an unavailable score")
    try:
        require_complete_best(output / 'checkpoints')
    except ValueError:
        pass
    else:
        raise ValueError("Unavailable evaluation could incorrectly rank a best checkpoint")
    original = [dict(score=.25, judge_failed=False, judge_parse_valid=True)]
    manager.verify = lambda batch: (original, [True])
    valid = score_batch(lambda: manager(data, return_dict=True),
                        dict(diagnostic_only=True), output / 'run', output / 'checkpoints', 0, 1,
                        diagnostic_only=True)
    trace = valid['reward_trace_info'][0]
    if (any(trace.get(k) != v for k, v in original[0].items())
            or trace['reward_replaced_by_group_mean'] or trace['effective_reward'] != .25
            or float(valid['reward_tensor'].sum()) != .25
            or not torch.equal(rng, torch.get_rng_state()) or failed[0]['score'] != -.25):
        raise ValueError("Validation containment altered a valid result, original score or RNG")
    report = dict(passed=True, fault_injection_only=True, real_evaluation_manager_executed=True,
        unavailable_result_contained=True, incomplete_best_rejected=True,
        valid_result_and_original_failed_score_preserved=True, cpu_rng_unchanged=True,
        model_calls=0, dataset_rows_used=0, optimizer_updates=0,
        real_scheduled_validation_recovery_not_yet_exercised=True)
    write(output / 'contract.json', report)
    return report
