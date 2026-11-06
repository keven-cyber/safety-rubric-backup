"""Exercise real shell selection with isolated config fixtures and no model calls."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BaselineFinalCoverageTests(unittest.TestCase):
    def run_case(self, steps=(900, 990, 1080, 1170), *, scope='selected', required='900,990,1080,1170', override='', missing=(), extra_best=False, action='run_baseline_general_eval', failure_at=None):
        with tempfile.TemporaryDirectory(prefix='baseline path ') as temp:
            ckpt = Path(temp)
            policy = 'meta-llama/Llama-3.2-3B-Instruct'
            prefix = ckpt / 'models' / (policy + '-nspo-')
            models = [str(prefix) + 'global_step_' + str(n) for n in steps]
            if extra_best:
                models.append(str(prefix) + 'global_step_best_480')
            for name in models:
                if any(name.endswith('_' + str(n)) for n in missing):
                    continue
                directory = Path(name)
                directory.mkdir(parents=True)
                (directory / 'config.json').write_text('{}')
            calls = ckpt / 'calls.txt'
            env = dict(os.environ, CKPT_PATH=str(ckpt), POLICY_MODEL_ID=policy, METHOD_TAG='nspo', BASELINE_GENERAL_EVAL_SCOPE=scope, BASELINE_REQUIRED_FINAL_STEPS=required, BASELINE_GENERAL_EVAL_STEP=override, CALLS=str(calls))
            body = 'set -euo pipefail\nsource "$1"\nSAFETY_EVAL_MODELS=(' + ' '.join(shlex.quote(x) for x in models) + ')\nLAST_MERGED_MODEL=' + shlex.quote(str(prefix) + 'global_step_1170') + '\nrun_general_eval() { printf "%s\\n" "$1" >> "$CALLS"; '
            if failure_at:
                body += '[ "$1" != ' + shlex.quote(str(prefix) + 'global_step_' + str(failure_at)) + ' ]; '
            body += '}\n' + action + '\n'
            result = subprocess.run(['bash', '-c', body, 'fixture', str(ROOT / 'scripts/baseline_common.sh')], env=env, capture_output=True, text=True)
            got = calls.read_text().splitlines() if calls.exists() else []
            return result, got, models

    def test_all_four_selected_in_order(self):
        result, got, models = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(got, models)

    def test_distinct_native_best_is_additional_and_not_renamed(self):
        result, got, models = self.run_case(extra_best=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(got, models)
        self.assertTrue(got[-1].endswith('global_step_best_480'))

    def test_missing_required_model_fails_before_any_eval(self):
        result, got, _ = self.run_case(steps=(900, 1080, 1170))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checkpoint 990', result.stderr)
        self.assertEqual(got, [])

    def test_missing_config_fails_before_any_eval(self):
        result, got, _ = self.run_case(missing=(1080,))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(got, [])

    def test_best_copy_cannot_substitute_numbered_role(self):
        result, got, _ = self.run_case(required='900,990,1080,1170,480', extra_best=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checkpoint 480', result.stderr)
        self.assertEqual(got, [])

    def test_single_step_override_conflicts_with_selected_scope(self):
        result, got, _ = self.run_case(override='900')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(got, [])

    def test_required_coverage_rejects_last_scope_before_safety(self):
        result, got, _ = self.run_case(scope='last', action='validate_baseline_final_coverage')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('needs BASELINE_GENERAL_EVAL_SCOPE=selected', result.stderr)
        self.assertEqual(got, [])

    def test_legacy_last_scope_is_preserved(self):
        result, got, _ = self.run_case(scope='last', required='')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].endswith('global_step_1170'))

    def test_legacy_explicit_single_step_is_preserved(self):
        result, got, _ = self.run_case(scope='last', required='', override='900')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].endswith('global_step_900'))

    def test_invalid_required_csv_rejected(self):
        for value in ['900,,1170', 'last,best', '900,0', '900, 1170', '900,']:
            result, got, _ = self.run_case(required=value)
            self.assertNotEqual(result.returncode, 0, value)
            self.assertEqual(got, [])

    def test_empty_selected_models_rejected(self):
        result, got, _ = self.run_case(steps=(), required='')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(got, [])

    def test_eval_failure_stops_later_models(self):
        result, got, models = self.run_case(failure_at=990)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(got, models[:2])

    def test_invalid_scope_rejected(self):
        result, got, _ = self.run_case(scope='all')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(got, [])

    def test_both_native_wrappers_call_shared_scope(self):
        for name in ['train_val_alphaalign.sh', 'train_val_nspo.sh']:
            text = (ROOT / name).read_text()
            self.assertIn('  run_baseline_general_eval\nelse\n', text)
            self.assertNotIn('general_eval_target=', text)


if __name__ == '__main__':
    unittest.main()
