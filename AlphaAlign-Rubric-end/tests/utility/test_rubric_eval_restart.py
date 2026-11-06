"""CPU-only regression tests for checkpoint selection and eval-only restarts."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EvalRestartTests(unittest.TestCase):
    def run_mode(self, **env):
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith(("RUBRIC_EVAL_", "PRIMUS_"))}
        clean.update(env)
        return subprocess.run(
            ["bash", "-ec", 'source "$1"; rubric_configure_eval_mode', "bash",
             str(ROOT / "scripts/rubric_eval_mode.sh")],
            env=clean, text=True, capture_output=True,
        )

    def test_training_default(self):
        self.assertEqual(self.run_mode().returncode, 0)

    def test_invalid_mode(self):
        self.assertNotEqual(self.run_mode(RUBRIC_EVAL_ONLY="yes").returncode, 0)

    def test_missing_source_never_trains(self):
        self.assertNotEqual(self.run_mode(RUBRIC_EVAL_ONLY="1").returncode, 0)

    def test_checkpoint_shard_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args = dict(RUBRIC_EVAL_ONLY="1", RUBRIC_EVAL_SOURCE_DIR=directory,
                        RUBRIC_EVAL_STEP="990")
            self.assertNotEqual(self.run_mode(**args).returncode, 0)
            actor = Path(directory) / "global_step_990/actor"
            actor.mkdir(parents=True)
            (actor / "model_world_size_2_rank_0.pt").write_bytes(b"test")
            self.assertNotEqual(self.run_mode(**args).returncode, 0)
            (actor / "model_world_size_2_rank_1.pt").touch()
            self.assertNotEqual(self.run_mode(**args).returncode, 0)
            (actor / "model_world_size_2_rank_1.pt").write_bytes(b"test")
            result = self.run_mode(**args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified 2 nonempty model shards for step 990", result.stdout)
            self.assertNotEqual(self.run_mode(**dict(args, RUBRIC_EVAL_STEP="900")).returncode, 0)
            self.assertNotEqual(self.run_mode(**dict(args, FINAL_SAFETY_EVAL_ENABLE="0",
                                                   FINAL_GENERAL_EVAL_ENABLE="0")).returncode, 0)

    def test_actual_launcher_selector(self):
        script = (ROOT / "train_val.sh").read_text()
        selector = script.split("  is_selected=0\n", 1)[1].split('  if [ "${is_selected}"', 1)[0]
        cases = [("600,900,last,best", "600", "0", "1"),
                 ("600,900,last,best", "990", "0", "1"),
                 ("600,900,last,best", "990", "1", "1"),
                 ("600,900,last,best", "960", "0", "0"),
                 ("990", "600", "0", "0"), ("990", "990", "0", "1"),
                 ("best", "600", "1", "1"), ("best", "990", "0", "0")]
        for steps, step, best, expected in cases:
            with self.subTest(steps=steps, step=step, best=best):
                result = subprocess.run(["bash", "-ec", "is_selected=0\n" + selector +
                                         '\nprintf "%s" "$is_selected"'],
                    env=dict(os.environ, FINAL_SAFETY_EVAL_STEPS=steps,
                             LAST_STANDARD_STEP="990", checkpoint_step=step,
                             is_best_checkpoint=best), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_primus_json_config_without_command_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "task.json"
            actor = Path(directory) / "global_step_990/actor"
            actor.mkdir(parents=True)
            (actor / "model_world_size_1_rank_0.pt").write_bytes(b"test")
            args = dict(PRIMUS_JOB_CONFIG_FILE_PATH=str(config), PRIMUS_OTHER_DATA_PATH=directory)
            config.write_text(json.dumps({"rubric_eval": {"enabled": True, "step": 990}}))
            result = self.run_mode(**args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified 1 nonempty model shards for step 990", result.stdout)
            config.write_text('{"rubric_eval": {"enabled": "true", "step": 990}}')
            self.assertNotEqual(self.run_mode(**args).returncode, 0)
            config.write_text('{"rubric_eval": {"enabled": true, "step": "bad"}}')
            self.assertNotEqual(self.run_mode(**args).returncode, 0)
            config.write_text('{bad json')
            self.assertNotEqual(self.run_mode(**args).returncode, 0)
            config.write_text('{}')
            self.assertEqual(self.run_mode(**args).returncode, 0)

    def test_exported_actor_only_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_world_size_1_rank_0.pt").write_bytes(b"test")
            result = self.run_mode(RUBRIC_EVAL_ONLY="1", RUBRIC_EVAL_STEP="990",
                                   RUBRIC_EVAL_SOURCE_DIR=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified 1 nonempty model shards for step 990", result.stdout)

    def test_primus_actor_model_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            actor.mkdir()
            (actor / "model_world_size_1_rank_0.pt").write_bytes(b"test")
            config = root / "task.json"
            config.write_text(json.dumps({"rubric_eval": {
                "enabled": True, "step": 990, "source": "actor_model"}}))
            args = dict(PRIMUS_JOB_CONFIG_FILE_PATH=str(config),
                        PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR=directory,
                        PRIMUS_SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME="actor")
            self.assertEqual(self.run_mode(**args).returncode, 0)
            args["PRIMUS_SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME"] = ""
            self.assertEqual(self.run_mode(**args).returncode, 0)
            args["PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR"] = str(actor)
            self.assertEqual(self.run_mode(**args).returncode, 0)
            args["PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR"] = ""
            self.assertNotEqual(self.run_mode(**args).returncode, 0)

    def test_shell_syntax_and_training_branch(self):
        for path in ["train_val.sh", "scripts/train_safety_v4.sh", "scripts/rubric_eval_mode.sh"]:
            subprocess.run(["bash", "-n", str(ROOT / path)], check=True)
        script = (ROOT / "train_val.sh").read_text()
        block = script.split('# 1. 等待 Guard 服务\n', 1)[1].split(
            'if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ] && ! kill', 1)[0]
        # Exercise the actual branch, replacing only the expensive training call.
        block = block.replace('bash "${PROJECT_ROOT}/examples/ppo_trainer/qwen3B-instruct-GRPO.sh"',
                              'echo TRAINING_EXECUTED')
        prelude = 'wait_for_api() { :; }; wait_for_model() { :; }; stop_service_pids() { :; }; wait_for_port_release() { :; }; judge_ports=(); judge_service_pids=();\n'
        for mode in ("0", "1"):
            result = subprocess.run(["bash", "-ec", prelude +
                'if [ "${RUBRIC_EVAL_ONLY}" != "1" ]; then\n' + block],
                env=dict(os.environ, RUBRIC_EVAL_ONLY=mode, REWARD_MODE_NEEDS_GUARD="0"),
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual("TRAINING_EXECUTED" in result.stdout, mode == "0")


if __name__ == "__main__":
    unittest.main()
