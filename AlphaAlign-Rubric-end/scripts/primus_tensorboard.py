"""TensorBoard event output; original strict JSON archives remain authoritative."""
import json
import math
import os
import runpy
from pathlib import Path


def _json(value):
    convert = runpy.run_path(str(Path(__file__).with_name('baseline_durable_tracking.py')))['json_container']
    return json.dumps(convert(value, redact=True), ensure_ascii=False, allow_nan=False, indent=2)


def event_directory():
    raw = Path(os.environ.get('TENSORBOARD_DIR', 'tensorboard_log'))
    if os.environ.get('CROSS_MODEL_EXPERIMENT_KEY'):
        base = Path(os.environ['PRIMUS_TENSORBOARD_LOG_DIR']).resolve()
        resolved = raw.resolve()
        if raw.is_symlink() or resolved == base or base not in resolved.parents:
            raise ValueError('TensorBoard needs an isolated directory under the Primus summary root')
    raw.mkdir(parents=True, exist_ok=True)
    return raw


class TensorBoardTracking:
    def __init__(self, project_name=None, experiment_name=None, config=None):
        from torch.utils.tensorboard import SummaryWriter
        # Validate/serialize before opening the event writer. No SDK fallback,
        # stringified invalid objects, or swallowed storage/numeric failures.
        text = _json(dict(project=project_name, experiment=experiment_name, config=config,
                          experiment_key=os.environ.get('CROSS_MODEL_EXPERIMENT_KEY'),
                          job_id=os.environ.get('PRIMUS_JOB_UUID'),
                          submit_timestamp=os.environ.get('PRIMUS_JOB_SUBMIT_TIMESTAMP'),
                          original_run_directory=os.environ.get('CROSS_MODEL_RUN_DIR')))
        self.writer = SummaryWriter(str(event_directory()), flush_secs=10)
        self.writer.add_text('run/config', '```json\n' + text + '\n```', 0)
        self.writer.flush()

    def log(self, data, step):
        # Check the entire batch before publishing any scalar from it.
        for value in data.values():
            if not math.isfinite(float(value)):
                raise ValueError('Nonfinite TensorBoard metric')
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def finish(self):
        self.writer.close()


def log_validation_samples(samples, step):
    from torch.utils.tensorboard import SummaryWriter
    text = _json(dict(step=step, samples=samples))
    # Validation preview frequency/count is selected by the unchanged trainer.
    with SummaryWriter(str(event_directory()), flush_secs=10) as writer:
        writer.add_text('val/generations', '```json\n' + text + '\n```', step)
        writer.flush()
