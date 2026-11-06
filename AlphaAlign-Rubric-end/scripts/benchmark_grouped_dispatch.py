"""One finite8-round real replay; never retry until a favorable score appears.

The72 rows are TWO explicitly labelled independent submissions of each frozen
answer. They are a load test, not72 unique examples or policy rollouts, and never
substitute for the original36-case/real8x8 mandatory admission tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from Evaluation.safety.json_evidence import write_json_atomic as atomic_json, read_json_complete
from scripts.graded_presentation_gate import environment, reward_replay, load_cases, smoke, rj, cq, pr
from scripts.benchmark_judge_dispatch import requests
from scripts.replay_presentation_rewards import actual_trainer
from verl.utils.grouped_dispatch import choose, ORDER, PROFILES, WIRE_FORMATS


def semantic_requests(directory):
    """Only declared output serialization may differ; all scored content matches."""
    from verl.utils import grouped_grounded_judge as grouped
    values=[]
    for path in sorted(Path(directory).rglob('events_*.jsonl')):
        for line in path.read_text().splitlines():
            event=json.loads(line)
            audit=event.get('request_audit') or {}
            if event.get('event') not in ('base_judge_attempt','targeted_judge_attempt'):
                continue
            if audit.get('grounding_protocol')==grouped.PROTOCOL:
                if not grouped.verified_event(event):
                    raise ValueError('Changed original grouped rule/evidence/sampling identity')
                values.append(json.dumps(dict(model=event['model'],user_sha256=audit['user_sha256'],
                    original_prompt_sha256=audit['original_prompt_sha256'],
                    original_response_sha256=hashlib.sha256(event['response'].encode()).hexdigest(),
                    temperature=audit['temperature'],max_tokens=audit['max_tokens'],seed=audit['request_seed']),sort_keys=True))
            else:
                # Targeted correction serialization has not changed at all.
                if not cq.verified_audit_event(event):
                    raise ValueError('Changed targeted correction request')
                values.append(json.dumps({k:audit.get(k) for k in ('system_sha256','user_sha256',
                    'guidance_sha256','max_tokens','temperature','request_seed','served_model')},sort_keys=True))
    return values


def both_grpo(report):
    rows = report['rows']
    groups = [r['prompt_group_sha256'] for r in rows]
    counts = Counter(groups)
    eligible = [i for i,r in enumerate(rows) if not r['quarantined'] and counts[groups[i]] > 1]
    if not eligible:
        raise ValueError('No valid stress-replay groups for actual GRPO')
    result = {}
    for name, mode, floor in [('G1','legacy',0.), ('G3','bounded_residual',.25)]:
        scores = torch.tensor([[rows[i]['damped_reward']['score']] for i in eligible], dtype=torch.float32)
        values = {k:np.array([rows[i]['damped_reward'][k] for i in eligible])
                  for k in (pr.COMPONENT, pr.SCORED, pr.CONTENT)}
        values.update(uid=np.array([groups[i] for i in eligible]),
                      targeted_refusal_adv_delta=np.zeros(len(eligible)))
        batch = SimpleNamespace(batch=dict(token_level_scores=scores.clone(), token_level_rewards=scores,
                                response_mask=torch.ones_like(scores)), non_tensor_batch=values)
        with environment(RUBRIC_EVOLVE=1, RUBRIC_EVOLVE_REWARD_MODE=mode,
                RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor, RUBRIC_EVOLVE_STD_FLOOR_SCOPE='global',
                RUBRIC_PRESENTATION_DAMPING_ENABLE=1):
            actual_trainer()(batch, 'GRPO')
        effects = batch.non_tensor_batch['presentation_advantage_contribution']
        if (not torch.isfinite(batch.batch['advantages']).all() or
                not np.isfinite(effects).all() or max(np.abs(effects), default=0.) > .050001):
            raise ValueError('Stress replay produced invalid or unbounded actual GRPO')
        result[name] = dict(eligible_indices=eligible, advantages=batch.batch['advantages'].tolist(),
                           presentation_contribution=effects.tolist(), std_floor=floor)
    return result


def verified_selection(path):
    path=Path(path)
    result=read_json_complete(path)
    pair=(result['selected_grouped_workers'],result['selected_grouped_fair'])
    if pair not in PROFILES.values():
        raise ValueError('Unknown selected grouped profile')
    if result['reason']=='incomplete_probe_retain_original_grouped_dispatch':
        if pair != PROFILES['baseline'] or result.get('selected_wire_format','object')!='object':
            raise ValueError('An incomplete probe cannot enable an optimization')
        return pair+(result.get('selected_wire_format','object'),)
    snapshots={}
    for label in ORDER:
        directory=path.parent/label
        saved=read_json_complete(directory/'timing_and_numerical.json')
        signatures,audit=requests(directory)
        consistency=cq.audit_directory(directory)
        complete=consistency['passed'] or cq.containment_verified(consistency,saved['report']['rows'])
        if (saved['requests']!=signatures or saved['response_ids']!=audit['independent_response_ids']
                or saved['evidence_complete']!=complete or saved['both_grpo']!=both_grpo(saved['report'])
                or saved.get('semantic_requests')!=semantic_requests(directory)):
            raise ValueError('Grouped tuning original requests or numerical proof changed')
        snapshots[label]=saved
    expected=choose(snapshots)
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('Grouped selection differs from the original measured evidence')
    return pair+(result['selected_wire_format'],)


def main():
    parser = argparse.ArgumentParser(); action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--output', type=Path);action.add_argument('--verify-selection',type=Path)
    args=parser.parse_args()
    if args.verify_selection:
        workers,fair,wire=verified_selection(args.verify_selection)
        print(workers,int(fair),wire);return
    args.output.mkdir(parents=True, exist_ok=False)
    result = choose({})
    result.update(predeclared_order=list(ORDER), unique_frozen_answer_count=36,
        load_repetitions=2, load_rows=72, real_policy_rollouts=False, optimizer_updates=0,
        real_judge_executed=False, no_extra_training_data=True,
        timing_includes_audit_IO_reward_and_GRPO_not_pure_GPU=True,
        note='Repeated frozen-answer stress replay only; original mandatory36 and real8x8 gates still follow')
    backend = rj._get_judge_backend()
    snapshots = {}
    try:
        if (not isinstance(backend, rj.ApiJudge) or len(backend.base_urls) != 3
                or os.getenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE') != '1'
                or backend._endpoint_max_inflight != 48):
            raise ValueError('Tuning requires the three actual grouped Judge replicas with endpoint48')
        original = [dict(c, replay_cohort='historical_sixteen') for c in load_cases()]
        frozen = smoke.FIXTURE.read_bytes()
        if hashlib.sha256(frozen).hexdigest() != smoke.FIXTURE_SHA256:
            raise ValueError('Frozen fixture changed')
        data = json.loads(frozen); smoke.validate_fixture(data)
        original += [dict(c, replay_cohort='original_twenty') for c in data['cases']]
        from scripts.validate_grouped_grounded import grammar
        syntax=[grammar(original,wire) for wire in ('object','tuple')]
        atomic_json(args.output/'syntax_preflight.json',syntax)
        if not all(s['passed'] for s in syntax):
            raise ValueError('Grouped tuning grammar contract failed')
        cases = [dict(c, stress_repetition=rep) for rep in range(2) for c in original]
        atomic_json(args.output/'load_inputs.json', cases)
        for label in ORDER:
            name = label.rsplit('_',1)[0]; workers,fair = PROFILES[name]
            directory = args.output/label; directory.mkdir()
            independent = rj.ApiJudge(','.join(backend.base_urls), backend.api_key, backend.model)
            rj._JUDGE_INSTANCE = independent
            rj._JUDGE_BACKEND = os.getenv('RUBRIC_JUDGE_BACKEND','local_api').lower().strip()
            with environment(RUBRIC_GROUPED_WIRE_FORMAT=WIRE_FORMATS[name], RUBRIC_GROUPED_TIMING_SCOPE='preflight_tuning', RUBRIC_GROUPED_API_CONCURRENCY=workers,
                RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE=int(fair),
                RUBRIC_BASE_JUDGE_TRACE_DIR=directory/'reward_attempts',
                RUBRIC_TARGETED_JUDGE_TRACE_DIR=directory/'correction_attempts'):
                start = time.monotonic()
                report = reward_replay(cases, independent, directory)
                seconds = time.monotonic()-start
                numerical = both_grpo(report)
            signatures,audit = requests(directory)
            consistency = cq.audit_directory(directory)
            complete = consistency['passed'] or cq.containment_verified(consistency,report['rows'])
            snapshot = dict(report=report,both_grpo=numerical,requests=signatures,
                semantic_requests=semantic_requests(directory),wire_format=WIRE_FORMATS[name],seconds=seconds,
                evidence_complete=complete,response_ids=audit['independent_response_ids'])
            atomic_json(directory/'timing_and_numerical.json',dict(snapshot,request_audit=audit,
                workers=workers,fair_dispatch=fair,consistency=consistency,stress_replay_only=True))
            snapshots[label] = snapshot
        result.update(choose(snapshots),real_judge_executed=True)
    except (OSError,TypeError):
        raise
    except Exception as error:
        result.update(reason='incomplete_probe_retain_original_grouped_dispatch',
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),completed_rounds=list(snapshots))
    finally:
        rj._JUDGE_INSTANCE = backend
        # An optional failed experiment may not taint the separate mandatory gate ledger.
        rj._JUDGE_BACKEND = os.getenv('RUBRIC_JUDGE_BACKEND','local_api').lower().strip()
    events=[]
    for path in args.output.rglob('events_*.jsonl'):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    response_events=sum(bool((event.get('request_audit') or {}).get('response_id')) for event in events)
    result.update(real_judge_executed=response_events>0,archived_attempt_events=len(events),
                  archived_response_id_events=response_events,
                  response_event_count_is_not_independence_or_equivalence_proof=True)
    atomic_json(args.output/'selection.json',result)
    print(json.dumps({k:result[k] for k in ('selected_grouped_workers','selected_grouped_fair','reason')}),flush=True)


if __name__ == '__main__':
    main()
