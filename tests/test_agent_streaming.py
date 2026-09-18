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
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.executor import WRAPPER  # noqa: E402
from batteryml_web.agent.streaming import (  # noqa: E402
    HEARTBEAT_MESSAGE,
    INDETERMINATE_STAGES,
    STAGE_COMPLETE,
    STAGE_EVALUATING,
    STAGE_EXTRACTING_TEST,
    STAGE_EXTRACTING_TRAIN,
    STAGE_FAILED,
    STAGE_LOADING,
    STAGE_PREPARING,
    STAGE_READING_TEST,
    STAGE_READING_TRAIN,
    STAGE_TRAINING,
    classify_stage,
    execute_plan_streaming,
    is_indeterminate_stage,
    is_training_progress,
    parse_progress_line,
    stage_for_description,
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

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"",
                 returncode: int = 0, delay: float = 0.0):
        self.stdout = self._pipe(stdout, delay)
        self.stderr = self._pipe(stderr, delay)
        self._returncode = returncode
        self.killed = False

    @staticmethod
    def _pipe(payload: bytes, delay: float = 0.0):
        """A pipe holding ``payload``, optionally staying open afterwards.

        ``delay`` keeps the write end open for that many seconds *after* the
        payload is written, which makes the run go quiet without ending —
        the only condition under which a heartbeat is emitted at all. Without
        it every fake run reaches EOF before the first heartbeat interval.
        """
        read_fd, write_fd = os.pipe()
        if payload:
            os.write(write_fd, payload)
        if delay > 0:
            def _close_later() -> None:
                time.sleep(delay)
                os.close(write_fd)

            threading.Thread(target=_close_later, daemon=True).start()
        else:
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
            classify_stage("Reading train data: 100%|#| 41/41 [00:02<00:00]"),
            STAGE_READING_TRAIN,
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
        """One stream, because a sequence is what this asserts.

        The seed line used to be fed on stdout while the bars went to stderr,
        and the two reader threads race: whichever pipe is drained first
        decides whether "Loading data" is ever reported at all, because once
        a counter has advanced the stage the never-walk-backwards rule
        rightly refuses to go back to the coarser prose stage. That made the
        test pass on luck — it failed the moment thread timing shifted — and
        it was asserting an ordering the OS does not guarantee, since a real
        run's stdout and stderr genuinely can interleave either way.

        Two pipes cannot express "before". One can, so the marker that comes
        first in a real run is written first here.
        """
        stderr = (
            b"Seed is set to 0.\n"
            b"Reading train data: 100%|##| 41/41 [00:02<00:00]\n"
            b"\rTraining BatLiNet:   0%|  | 0/1 [00:00<?, ?it/s]"
            b"\rTraining BatLiNet: 100%|##| 1/1 [00:06<00:00]\n"
        )
        result, events = _collect(_plan(), _FakeProc(stderr=stderr))
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


class TestTheStagesAreTheBarsBatteryMLPrints(unittest.TestCase):
    """Every stage name traces to a literal ``desc=`` in the source.

    A stage the pipeline never announces would be a label invented by the
    display layer, so each of these is checked against the exact line
    BatteryML emits — including the upstream ``Traning`` typo, which is real
    output and must not be "fixed" by refusing to recognise it.
    """

    CASES = (
        ("Reading train data:  48%|####      | 20/41 [00:01<00:01]",
         STAGE_READING_TRAIN, 20, 41),
        ("Reading test data: 100%|##########| 42/42 [00:02<00:00]",
         STAGE_READING_TEST, 42, 42),
        ("Reading test data:  71%|#######   | 30/42 [00:01<00:01]",
         STAGE_READING_TEST, 30, 42),
        ("Extracting train cycle samples:  24%|##        | 10/41 [00:03<00:09]",
         STAGE_EXTRACTING_TRAIN, 10, 41),
        ("Extracting train cycle samples:  49%|####      | 20/41 [00:05<00:05]",
         STAGE_EXTRACTING_TRAIN, 20, 41),
        ("Extracting test cycle samples:  48%|####      | 20/42 [00:05<00:05]",
         STAGE_EXTRACTING_TEST, 20, 42),
        ("Training BatLiNet:   4%|#         | 4/100 [00:12<04:48]",
         STAGE_TRAINING, 4, 100),
        ("Traning:   4%|#         | 4/100 [00:12<04:48]",
         STAGE_TRAINING, 4, 100),
    )

    def test_each_real_bar_yields_its_stage_and_its_counter(self) -> None:
        for line, stage, current, total in self.CASES:
            with self.subTest(line=line):
                parsed = parse_progress_line(line)
                self.assertIsNotNone(parsed, "the counter was not parsed at all")
                self.assertEqual((parsed["current"], parsed["total"]), (current, total))
                self.assertEqual(classify_stage(line), stage)

    def test_the_percentage_comes_from_the_counter_not_from_tqdm(self) -> None:
        """``fraction`` is ``current / total``, recomputed, not tqdm's round.

        tqdm prints a rounded integer percent; the bar is drawn from the
        exact ratio, so 20/41 is 48.78% rather than the printed 48%.
        """
        parsed = parse_progress_line("Reading train data:  48%|####      | 20/41 [00:01<00:01]")
        self.assertAlmostEqual(parsed["fraction"], 20 / 41)
        self.assertAlmostEqual(parsed["fraction"] * 100, 48.78, places=2)

        full = parse_progress_line("Reading test data: 100%|##########| 42/42 [00:02<00:00]")
        self.assertEqual(full["fraction"], 1.0)

        extracting = parse_progress_line(
            "Extracting train cycle samples:  24%|##        | 10/41 [00:03<00:09]"
        )
        self.assertAlmostEqual(extracting["fraction"] * 100, 24.39, places=2)

        for line in (
            "Training BatLiNet:   4%|#         | 4/100 [00:12<04:48]",
            "Traning:   4%|#         | 4/100 [00:12<04:48]",
        ):
            with self.subTest(line=line):
                self.assertAlmostEqual(parse_progress_line(line)["fraction"], 0.04)

    def test_evaluating_is_indeterminate_and_gets_no_counter(self) -> None:
        self.assertTrue(is_indeterminate_stage(STAGE_EVALUATING))
        for stage in INDETERMINATE_STAGES:
            with self.subTest(stage=stage):
                self.assertTrue(is_indeterminate_stage(stage))
        for stage in (STAGE_READING_TRAIN, STAGE_EXTRACTING_TEST, STAGE_TRAINING):
            with self.subTest(stage=stage):
                self.assertFalse(is_indeterminate_stage(stage))

    def test_unrelated_stdout_fabricates_nothing(self) -> None:
        for line in (
            "Evaluating the model now",
            "Saving results to runs/soh_smeen",
            "processed 3/4 of the way there",
            "Traceback (most recent call last):",
            "  File \"batteryml/pipeline.py\", line 42, in train",
            "AssertionError: data/processed/MATR",
            "20/41",
            "Downloading 20/41 files",
            HEARTBEAT_MESSAGE,
        ):
            with self.subTest(line=line):
                self.assertIsNone(
                    parse_progress_line(line),
                    "a percentage was invented from a line with no tqdm bar",
                )
                self.assertIsNone(classify_stage(line))

    def test_an_unknown_tqdm_bar_reports_progress_but_moves_no_stage(self) -> None:
        line = "Compacting shards:  50%|#####     | 5/10 [00:01<00:01]"
        self.assertIsNotNone(parse_progress_line(line))
        self.assertIsNone(stage_for_description("Compacting shards"))
        self.assertIsNone(classify_stage(line))


class TestTheStageSequenceForARealRun(unittest.TestCase):
    """The four data bars, in the order ``Task.build`` emits them."""

    DATA_BARS = (
        b"Reading train data: 100%|##########| 41/41 [00:02<00:00]\n"
        b"Reading test data: 100%|##########| 42/42 [00:02<00:00]\n"
        b"Extracting train cycle samples: 100%|##########| 41/41 [00:09<00:00]\n"
        b"Extracting test cycle samples: 100%|##########| 42/42 [00:09<00:00]\n"
    )

    def _stages(self, stderr: bytes, stdout: bytes = b"Seed is set to 0.\n") -> list[str]:
        _, events = _collect(_plan(), _FakeProc(stdout=stdout, stderr=stderr))
        stages: list[str] = []
        for event in events:
            stage = event.get("stage")
            if stage and (not stages or stages[-1] != stage):
                stages.append(str(stage))
        return stages

    def test_the_four_data_stages_are_reported_separately(self) -> None:
        stages = self._stages(self.DATA_BARS)
        for stage in (
            STAGE_READING_TRAIN,
            STAGE_READING_TEST,
            STAGE_EXTRACTING_TRAIN,
            STAGE_EXTRACTING_TEST,
        ):
            with self.subTest(stage=stage):
                self.assertIn(stage, stages)
        order = [stages.index(s) for s in (
            STAGE_READING_TRAIN, STAGE_READING_TEST,
            STAGE_EXTRACTING_TRAIN, STAGE_EXTRACTING_TEST,
        )]
        self.assertEqual(order, sorted(order), f"stages arrived out of order: {stages}")

    def test_a_sklearn_run_reaches_training_with_no_training_counter(self) -> None:
        """Ridge and its siblings print nothing while fitting.

        A full *test extraction* counter means ``Task.build`` has returned
        and ``model.fit`` is what remains, so the run leaves the extraction
        stage rather than sitting on "Extracting test samples" for the whole
        fit. Nothing waits for an epoch counter that is never coming.
        """
        stages = self._stages(self.DATA_BARS)
        self.assertIn(STAGE_TRAINING, stages)
        self.assertGreater(stages.index(STAGE_TRAINING), stages.index(STAGE_EXTRACTING_TEST))

    def test_a_sklearn_run_still_terminates_and_releases(self) -> None:
        """No training event must never mean a run stuck at RUNNING."""
        result, events = _collect(_plan(), _FakeProc(stderr=self.DATA_BARS))
        self.assertTrue(result["success"])
        self.assertEqual(result["final_stage"], STAGE_COMPLETE)
        self.assertEqual(events[-1]["type"], "completed")

    def test_a_failed_run_leaves_running_immediately(self) -> None:
        result, events = _collect(
            _plan(),
            _FakeProc(stderr=self.DATA_BARS, returncode=1),
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["final_stage"], STAGE_FAILED)
        self.assertEqual(events[-1]["type"], "completed")
        self.assertEqual(events[-1]["stage"], STAGE_FAILED)
        self.assertIsNone(events[-1]["fraction"], "a failed run may not report 100%")

    def test_a_late_reading_bar_does_not_drag_the_stage_backwards(self) -> None:
        stderr = (
            self.DATA_BARS
            + b"\rTraning: 100%|##########| 100/100 [00:30<00:00]\n"
            + b"Reading test data: 100%|##########| 42/42 [00:02<00:00]\n"
        )
        _, events = _collect(_plan(), _FakeProc(stderr=stderr))
        self.assertEqual(events[-1]["stage"], STAGE_COMPLETE)
        after_training = [
            e["stage"] for e in events
            if e["stage"] not in (STAGE_COMPLETE, STAGE_PREPARING)
        ]
        self.assertNotIn(
            STAGE_READING_TEST, after_training[after_training.index(STAGE_TRAINING):],
            "a bar printed during evaluation reset the stage to a data stage",
        )

    def test_the_counter_belongs_to_the_stage_that_printed_it(self) -> None:
        """A new stage never inherits the previous stage's counter."""
        stderr = (
            b"Reading train data: 100%|##########| 41/41 [00:02<00:00]\n"
            b"Extracting train cycle samples:   2%|          | 1/41 [00:00<00:09]\n"
        )
        _, events = _collect(_plan(), _FakeProc(stderr=stderr))
        extracting = [e for e in events if e["stage"] == STAGE_EXTRACTING_TRAIN]
        self.assertTrue(extracting)
        self.assertEqual((extracting[-1]["current"], extracting[-1]["total"]), (1, 41))


class TestAgainstRecordedRealOutput(unittest.TestCase):
    """The parser, against output a real BatteryML run actually produced.

    Every segment below is copied verbatim from the stderr of a
    ``bin/batteryml.py run ... --train --eval`` invocation of the SOH route
    with Ridge, on a synthetic fixture dataset of 12 cells. It is recorded
    rather than regenerated because this container has no real dataset — and
    recorded rather than hand-written because hand-written tqdm is the one
    thing that cannot show the parser is right about tqdm.

    Note the Unicode block characters: real tqdm draws ``█``, not ``#``.
    """

    RECORDED = [
        "Reading train data:   0%|          | 0/6 [00:00<?, ?it/s]",
        "Reading train data: 100%|██████████| 6/6 [00:00<00:00, 2012.78it/s]",
        "Reading test data:   0%|          | 0/6 [00:00<?, ?it/s]",
        "Reading test data: 100%|██████████| 6/6 [00:00<00:00, 2791.24it/s]",
        "Extracting train cycle samples:   0%|          | 0/6 [00:00<?, ?it/s]",
        "Extracting train cycle samples: 100%|██████████| 6/6 [00:00<00:00, 1691.59it/s]",
        "Extracting test cycle samples:   0%|          | 0/6 [00:00<?, ?it/s]",
        "Extracting test cycle samples: 100%|██████████| 6/6 [00:00<00:00, 1985.78it/s]",
    ]

    #: stdout of the same run. The metrics line is here on purpose: the
    #: parser must take nothing from it.
    RECORDED_STDOUT = [
        "runs/scratch_progress_probe",
        "Seed is set to 0.",
        "Seed is set to 0.",
        "{'MAE': 0.04196943450665128, 'RMSE': 0.04855779900033929}",
    ]

    def test_the_unicode_bar_real_tqdm_draws_is_parsed(self) -> None:
        parsed = parse_progress_line(self.RECORDED[1])
        self.assertIsNotNone(parsed, "real tqdm's █ bar was not recognised")
        self.assertEqual((parsed["current"], parsed["total"]), (6, 6))
        self.assertEqual(parsed["fraction"], 1.0)

    def test_the_recorded_run_produces_the_four_data_stages_then_training(self) -> None:
        from batteryml_web.agent.streaming import _ProgressState

        state = _ProgressState()
        seen: list[str] = []
        for segment in self.RECORDED:
            _, progress, entered = state.observe(segment)
            self.assertIsNotNone(progress, f"no counter parsed from {segment!r}")
            owner = state.counter_stage
            if owner and (not seen or seen[-1] != owner):
                seen.append(owner)
            if entered and (not seen or seen[-1] != entered):
                seen.append(entered)
        self.assertEqual(
            seen,
            [
                STAGE_READING_TRAIN,
                STAGE_READING_TEST,
                STAGE_EXTRACTING_TRAIN,
                STAGE_EXTRACTING_TEST,
                STAGE_TRAINING,
            ],
        )
        # A one-shot sklearn fit prints no epoch counter, and none is waited
        # for: the run reaches Training from the data stages alone.
        self.assertEqual(state.stage, STAGE_TRAINING)

    def test_the_metrics_line_is_not_read_for_anything(self) -> None:
        """Not for a stage, not for a counter, and certainly not for a value.

        Metric values come only from the canonical ``scores`` BatteryML
        saved; the streaming layer never reads a number out of stdout
        (CLAUDE.md §3).
        """
        metrics_line = self.RECORDED_STDOUT[-1]
        self.assertIsNone(parse_progress_line(metrics_line))
        self.assertIsNone(classify_stage(metrics_line))

    def test_the_prose_lines_place_the_run_without_a_counter(self) -> None:
        self.assertEqual(classify_stage("Seed is set to 0."), STAGE_LOADING)
        self.assertTrue(is_indeterminate_stage(STAGE_LOADING))


class TestHeartbeatsAreMarkedAndCarryNoNewInformation(unittest.TestCase):
    """The "Running." spam, cut off at its source.

    The heartbeat exists so an elapsed-time display can tick while the run is
    quiet. It repeats the last observed state and advances nothing, so a
    consumer that keeps a log must be able to recognise and skip it — by a
    flag, not by matching its text.
    """

    def _all_events(self, proc: _FakeProc) -> list[dict]:
        events: list[dict] = []
        execute_plan_streaming(
            _plan(),
            on_event=events.append,
            popen=_popen_factory(proc),
            repo_root=REPO_ROOT,
        )
        return events

    def test_every_event_declares_whether_it_is_a_heartbeat(self) -> None:
        for event in self._all_events(_FakeProc(stdout=b"Seed is set to 0.\n")):
            with self.subTest(event=event.get("message")):
                self.assertIn("heartbeat", event)
                self.assertIsInstance(event["heartbeat"], bool)

    def test_only_heartbeats_carry_the_heartbeat_message(self) -> None:
        for event in self._all_events(_FakeProc(stdout=b"Seed is set to 0.\n")):
            if event.get("message") == HEARTBEAT_MESSAGE:
                self.assertTrue(event["heartbeat"])
            if event.get("heartbeat"):
                self.assertEqual(event["message"], HEARTBEAT_MESSAGE)

    def test_a_heartbeat_repeats_the_last_observed_counter(self) -> None:
        """It may echo the last counter; it may never advance one.

        The echo is what keeps the bar from blanking every 0.2 s between two
        tqdm updates. What it must never do is move.
        """
        stderr = b"Extracting train cycle samples:  49%|####      | 20/41 [00:05<00:05]\n"
        events = self._all_events(_FakeProc(stderr=stderr, delay=0.5))
        beats = [e for e in events if e.get("heartbeat")]
        self.assertTrue(beats, "no heartbeat was emitted during a quiet run")
        after = [
            e for e in beats
            if e.get("stage") == STAGE_EXTRACTING_TRAIN and e.get("current") is not None
        ]
        for event in after:
            with self.subTest(event=event):
                self.assertEqual((event["current"], event["total"]), (20, 41))
                self.assertAlmostEqual(event["fraction"], 20 / 41)

    def test_no_real_output_line_is_a_heartbeat(self) -> None:
        stdout = b"Seed is set to 0.\nsomething the pipeline said\n"
        stderr = b"Reading train data: 100%|##########| 41/41 [00:02<00:00]\n"
        for event in self._all_events(_FakeProc(stdout=stdout, stderr=stderr)):
            if event.get("heartbeat"):
                continue
            with self.subTest(message=event.get("message")):
                self.assertNotEqual(event.get("message"), HEARTBEAT_MESSAGE)


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
