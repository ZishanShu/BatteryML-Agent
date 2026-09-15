"""Tests for the streaming executor: parsing, events, and honest failure.

Milestone B. No BatteryML run is ever started: the subprocess boundary is
replaced by a fake process whose stdout/stderr are real OS pipes preloaded
with recorded output, so carriage-return handling and concurrent draining
are exercised for real while nothing trains.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_streaming -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.executor import WRAPPER  # noqa: E402
from batteryml_web.agent.streaming import (  # noqa: E402
    HEARTBEAT_MESSAGE,
    STAGE_COMPLETE,
    STAGE_EVALUATING,
    STAGE_FAILED,
    STAGE_LOADING,
    STAGE_TRAINING,
    classify_stage,
    execute_plan_streaming,
    is_training_progress,
    parse_progress_line,
)

CONFIG = "configs/external/batlinet/matr_1.yaml"
WORKSPACE = "runs/agent_streaming_unittest"


def _plan(epochs: int | None = 1) -> dict:
    command = [
        sys.executable,
        WRAPPER,
        "run",
        CONFIG,
        "--workspace",
        WORKSPACE,
        "--train",
        "--eval",
        "--device",
        "cpu",
        "--seed",
        "0",
    ]
    if epochs is not None:
        command += ["--epochs", str(epochs)]
    return {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": "batlinet",
        "model_display": "BatLiNet — Inter-cell Lifetime Prediction",
        "model_status": "READY",
        "dataset_status": "READY",
        "config_path": CONFIG,
        "config_valid": True,
        "can_execute": True,
        "validation_errors": [],
        "command": command,
        "epochs": epochs,
        "reason": "test plan",
    }


class _FakeProc:
    """A finished process whose pipes already hold the recorded output.

    Real OS pipes, so :func:`os.read` and the carriage-return splitting in
    the pump run exactly as they do against BatteryML. Payloads stay well
    under the pipe buffer so no writer thread is needed.
    """

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = self._pipe(stdout)
        self.stderr = self._pipe(stderr)
        self._returncode = returncode
        self.killed = False

    @staticmethod
    def _pipe(payload: bytes):
        read_fd, write_fd = os.pipe()
        if payload:
            os.write(write_fd, payload)
        os.close(write_fd)
        return os.fdopen(read_fd, "rb", 0)

    def wait(self):
        return self._returncode

    def kill(self):
        self.killed = True


def _popen_factory(proc: _FakeProc):
    calls: list = []

    def popen(argv, **kwargs):
        calls.append({"argv": list(argv), "kwargs": dict(kwargs)})
        return proc

    popen.calls = calls  # type: ignore[attr-defined]
    return popen


def _collect(plan: dict, proc: _FakeProc, **kwargs) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    result = execute_plan_streaming(
        plan,
        on_event=events.append,
        popen=_popen_factory(proc),
        repo_root=REPO_ROOT,
        **kwargs,
    )
    return result, [e for e in events if e.get("message") != HEARTBEAT_MESSAGE]


class TestProgressParsing(unittest.TestCase):
    def test_extension_model_tqdm_line(self) -> None:
        parsed = parse_progress_line(
            "Training BatLiNet:  10%|#         | 1/10 [00:06<00:54,  6.80s/it]"
        )
        self.assertEqual(parsed["desc"], "Training BatLiNet")
        self.assertEqual((parsed["current"], parsed["total"]), (1, 10))
        self.assertAlmostEqual(parsed["fraction"], 0.1)
        self.assertTrue(is_training_progress(parsed["desc"]))

    def test_upstream_typo_desc_is_recognised(self) -> None:
        parsed = parse_progress_line("Traning:   1%|          | 1/100 [00:00<00:30,  3.20it/s]")
        self.assertEqual((parsed["current"], parsed["total"]), (1, 100))
        self.assertTrue(is_training_progress(parsed["desc"]))
        self.assertEqual(classify_stage("Traning:   1%|  | 1/100 [00:00<00:30]"), STAGE_TRAINING)

    def test_single_epoch_smoke_line(self) -> None:
        parsed = parse_progress_line("Training BatLiNet: 100%|##########| 1/1 [00:06<00:00,  6.80s/it]")
        self.assertEqual((parsed["current"], parsed["total"]), (1, 1))
        self.assertEqual(parsed["fraction"], 1.0)

    def test_data_reading_bar_is_progress_but_not_training(self) -> None:
        parsed = parse_progress_line("Reading train data: 100%|##########| 41/41 [00:02<00:00]")
        self.assertEqual((parsed["current"], parsed["total"]), (41, 41))
        self.assertFalse(is_training_progress(parsed["desc"]))
        self.assertEqual(
            classify_stage("Reading train data: 100%|#| 41/41 [00:02<00:00]"), STAGE_LOADING
        )

    def test_no_counter_means_no_progress_not_a_guess(self) -> None:
        for line in (
            "",
            "Loading data from data/processed/MATR",
            "Traceback (most recent call last):",
            "AssertionError: data/processed/MATR",
            "processed 3/4 of the way there",
            "{'RMSE': 794.3201293945312, 'MAE': 767.9190673828125}",
        ):
            with self.subTest(line=line):
                self.assertIsNone(parse_progress_line(line))
                self.assertIsNone(classify_stage(line))

    def test_zero_total_yields_no_fraction(self) -> None:
        parsed = parse_progress_line("Training X:   0%|          | 0/0 [00:00<?, ?it/s]")
        self.assertIsNone(parsed["fraction"])

    def test_pipeline_prose_markers_map_to_loading(self) -> None:
        self.assertEqual(classify_stage("Seed is set to 0."), STAGE_LOADING)
        self.assertEqual(
            classify_stage("Load datasets from cache cache/battery_cache_abc.pkl."),
            STAGE_LOADING,
        )


class TestStreamingExecution(unittest.TestCase):
    def test_carriage_return_updates_become_separate_events(self) -> None:
        stderr = (
            b"\rTraining BatLiNet:   0%|          | 0/1 [00:00<?, ?it/s]"
            b"\rTraining BatLiNet: 100%|##########| 1/1 [00:06<00:00,  6.80s/it]\n"
        )
        result, events = _collect(_plan(), _FakeProc(stdout=b"Seed is set to 0.\n", stderr=stderr))
        training = [e for e in events if e["type"] == "training_progress"]
        self.assertEqual([(e["current"], e["total"]) for e in training], [(0, 1), (1, 1)])
        self.assertAlmostEqual(training[-1]["fraction"], 1.0)
        self.assertTrue(result["success"])

    def test_full_output_is_preserved_including_carriage_returns(self) -> None:
        stderr = b"\rTraning:   1%|  | 1/100 [00:00<00:30]\rTraning: 100%|##| 100/100 [00:30<00:00]\n"
        stdout = b"Seed is set to 0.\nsome other line\n"
        result, _ = _collect(_plan(), _FakeProc(stdout=stdout, stderr=stderr))
        self.assertEqual(result["stdout"], stdout.decode())
        self.assertEqual(result["stderr"], stderr.decode())
        self.assertIn("\r", result["stderr"])

    def test_stage_sequence_is_observed_not_invented(self) -> None:
        stdout = b"Seed is set to 0.\n"
        stderr = (
            b"Reading train data: 100%|##| 41/41 [00:02<00:00]\n"
            b"\rTraining BatLiNet:   0%|  | 0/1 [00:00<?, ?it/s]"
            b"\rTraining BatLiNet: 100%|##| 1/1 [00:06<00:00]\n"
        )
        result, events = _collect(_plan(), _FakeProc(stdout=stdout, stderr=stderr))
        stages = [e["stage"] for e in events]
        self.assertIn(STAGE_LOADING, stages)
        self.assertIn(STAGE_TRAINING, stages)
        self.assertIn(STAGE_EVALUATING, stages)
        self.assertEqual(result["final_stage"], STAGE_COMPLETE)
        self.assertEqual(events[-1]["type"], "completed")

    def test_argv_list_and_no_shell(self) -> None:
        proc = _FakeProc()
        popen = _popen_factory(proc)
        execute_plan_streaming(_plan(), popen=popen, repo_root=REPO_ROOT)
        call = popen.calls[0]  # type: ignore[attr-defined]
        self.assertIsInstance(call["argv"], list)
        self.assertIs(call["kwargs"]["shell"], False)
        self.assertEqual(call["argv"][2], "run")
        self.assertIn("--epochs", call["argv"])

    def test_failed_run_is_reported_honestly_and_never_retried(self) -> None:
        proc = _FakeProc(
            stderr=b"torch.cuda.OutOfMemoryError: CUDA out of memory.\n", returncode=1
        )
        popen = _popen_factory(proc)
        result = execute_plan_streaming(_plan(), popen=popen, repo_root=REPO_ROOT)
        self.assertEqual(len(popen.calls), 1)  # type: ignore[attr-defined]
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["final_stage"], STAGE_FAILED)
        self.assertIn("no retry performed", result["error"])
        self.assertIn("CUDA out of memory", result["stderr"])
        self.assertEqual(result["artifacts"]["prediction_files"], [])

    def test_no_metric_is_ever_read_out_of_the_output(self) -> None:
        stdout = b"{'RMSE': 794.3201293945312, 'MAE': 767.9190673828125, 'MAPE': 1.35}\n"
        result, events = _collect(_plan(), _FakeProc(stdout=stdout))
        self.assertNotIn("metrics", result)
        for event in events:
            self.assertNotIn("metrics", event)
            for key in ("current", "total", "fraction"):
                value = event.get(key)
                self.assertTrue(value is None or isinstance(value, (int, float)))
        # The line is still preserved verbatim as output.
        self.assertIn("794.3201293945312", result["stdout"])

    def test_refused_plan_never_starts_a_process(self) -> None:
        bad = _plan()
        bad["can_execute"] = False
        popen = _popen_factory(_FakeProc())
        events: list[dict] = []
        result = execute_plan_streaming(
            bad, on_event=events.append, popen=popen, repo_root=REPO_ROOT
        )
        self.assertEqual(popen.calls, [])  # type: ignore[attr-defined]
        self.assertFalse(result["started"])
        self.assertFalse(result["success"])
        self.assertEqual(result["final_stage"], STAGE_FAILED)
        self.assertEqual(events[-1]["type"], "completed")

    def test_shell_string_command_is_refused(self) -> None:
        bad = _plan()
        bad["command"] = "python batteryml_web/core/run_batteryml_cli.py run ..."
        popen = _popen_factory(_FakeProc())
        result = execute_plan_streaming(bad, popen=popen, repo_root=REPO_ROOT)
        self.assertEqual(popen.calls, [])  # type: ignore[attr-defined]
        self.assertIn("shell", str(result["error"]).lower())

    def test_callback_exception_does_not_stop_the_run(self) -> None:
        def boom(event: dict) -> None:
            raise RuntimeError("display bug")

        result = execute_plan_streaming(
            _plan(),
            on_event=boom,
            popen=_popen_factory(_FakeProc(stdout=b"Seed is set to 0.\n")),
            repo_root=REPO_ROOT,
        )
        self.assertTrue(result["success"])

    def test_no_callback_is_supported(self) -> None:
        result = execute_plan_streaming(
            _plan(),
            popen=_popen_factory(_FakeProc(stdout=b"Seed is set to 0.\n")),
            repo_root=REPO_ROOT,
        )
        self.assertTrue(result["success"])

    def test_result_shape_matches_the_blocking_executor(self) -> None:
        from batteryml_web.agent import executor as blocking

        import inspect

        source = inspect.getsource(blocking.execute_plan)
        result, _ = _collect(_plan(), _FakeProc())
        for key in (
            "started",
            "completed",
            "success",
            "returncode",
            "command",
            "workspace",
            "stdout",
            "stderr",
            "error",
            "artifacts",
            "dry_run",
        ):
            self.assertIn(key, result, f"{key} missing; build_result_context reads this shape")
            self.assertIn(f'"{key}"', source)


class TestStreamingIsBackendOnly(unittest.TestCase):
    def test_no_streamlit_import(self) -> None:
        source = (REPO_ROOT / "batteryml_web" / "agent" / "streaming.py").read_text(encoding="utf-8")
        self.assertNotIn("import streamlit", source)
        self.assertNotIn("shell=True", source)

    def test_no_metric_names_are_parsed(self) -> None:
        source = (REPO_ROOT / "batteryml_web" / "agent" / "streaming.py").read_text(encoding="utf-8")
        for token in ("RMSE", "MAE", "MAPE"):
            # The names may be discussed in prose but never compiled into a regex.
            self.assertNotIn(f"'{token}'", source)
            self.assertNotIn(f'"{token}"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
