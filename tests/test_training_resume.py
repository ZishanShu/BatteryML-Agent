"""Resume is a continuation, not a restart — proved on a tiny real model.

The claim "training resumed correctly" is only worth making if an
uninterrupted run and an interrupted-then-resumed run end in the same place.
So these tests train a small deterministic network two ways and compare the
parameters, rather than asserting that a checkpoint file has the right keys.

Everything runs on CPU in a temporary directory in well under a second. No
GPU, no dataset, no Streamlit.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_training_resume -v
"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from batteryml_web.core.training_checkpoint import (  # noqa: E402
    ALLOWED_OVERRIDES,
    BLOCKING_IDENTITY_FIELDS,
    KIND_FUTURE_VERSION,
    KIND_RESUMABLE,
    KIND_UNREADABLE,
    KIND_WEIGHTS_ONLY,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    build_checkpoint,
    capture_rng_state,
    classify_checkpoint,
    config_fingerprint,
    restore_rng_state,
    save_checkpoint,
    split_fingerprint,
    validate_resume,
)


class TinyNet(nn.Module):
    """The smallest thing that still has an optimizer state worth restoring."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.layer = nn.Linear(width, 1)

    def forward(self, x):
        return self.layer(x)


def _seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(count: int = 6, width: int = 4):
    """Fixed data, so any difference between runs comes from training state."""
    generator = torch.Generator().manual_seed(1234)
    return [
        (torch.randn(3, width, generator=generator),
         torch.randn(3, 1, generator=generator))
        for _ in range(count)
    ]


def _train(model, optimizer, data, epochs, start_epoch=0, shuffle=True):
    """Train, drawing the per-epoch order from the global torch RNG.

    The shuffle matters: it is what makes the RNG state part of the
    trajectory, so a resume that does not restore RNG diverges here.
    """
    loss_fn = nn.MSELoss()
    for _ in range(start_epoch, epochs):
        order = torch.randperm(len(data)) if shuffle else torch.arange(len(data))
        for index in order.tolist():
            x, y = data[index]
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
    return model


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def _identity(**overrides):
    base = {
        "model_key": "tinynet",
        "dataset": "SYNTHETIC",
        "task": "RUL prediction",
        "config_fingerprint": config_fingerprint({"lr": 0.01, "width": 4}),
        "split_fingerprint": split_fingerprint(6, 2),
        "seed": "0",
    }
    base.update(overrides)
    return base


class TestResumeReproducesUninterruptedTraining(unittest.TestCase):
    """A9: the whole point — B must land where A landed."""

    EPOCHS = 8
    STOP_AT = 3

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        self.data = _batches()

    def _uninterrupted(self):
        _seed_everything(0)
        model = TinyNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        _train(model, optimizer, self.data, self.EPOCHS)
        return _params(model), optimizer.state_dict()

    def _interrupted_then_resumed(self):
        _seed_everything(0)
        model = TinyNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        _train(model, optimizer, self.data, self.STOP_AT)

        payload = build_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=self.STOP_AT,
            target_total_epochs=self.EPOCHS,
            training_identity=_identity(),
        )
        path = save_checkpoint(payload, self.directory / "resume.ckpt")

        # A genuinely fresh process state: new objects, and generators
        # deliberately disturbed so only a real restore can recover them.
        del model, optimizer
        _seed_everything(9999)
        for _ in range(17):
            torch.randn(5)

        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        restored = TinyNet()
        restored.load_state_dict(loaded["model_state"])
        new_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
        new_optimizer.load_state_dict(loaded["optimizer_state"])
        warnings = restore_rng_state(loaded["rng_state"])
        self.assertEqual(warnings, [], "RNG restore reported a problem")

        _train(
            restored, new_optimizer, self.data, self.EPOCHS,
            start_epoch=loaded["epoch"],
        )
        return _params(restored), new_optimizer.state_dict(), loaded

    def test_final_parameters_match(self) -> None:
        expected, _ = self._uninterrupted()
        actual, _, _ = self._interrupted_then_resumed()
        self.assertEqual(len(expected), len(actual))
        for index, (want, have) in enumerate(zip(expected, actual)):
            self.assertTrue(
                torch.allclose(want, have, atol=1e-6),
                f"parameter {index} diverged: max |diff| = "
                f"{(want - have).abs().max().item():.3e}",
            )

    def test_optimizer_state_continues_rather_than_restarting(self) -> None:
        _, expected_state = self._uninterrupted()
        _, actual_state, _ = self._interrupted_then_resumed()
        expected_steps = [s["step"] for s in expected_state["state"].values()]
        actual_steps = [s["step"] for s in actual_state["state"].values()]
        self.assertEqual(expected_steps, actual_steps)
        # ...and Adam's moments match, which is what a warm start would lose.
        for key in expected_state["state"]:
            for moment in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(
                    torch.allclose(
                        expected_state["state"][key][moment],
                        actual_state["state"][key][moment],
                        atol=1e-6,
                    ),
                    f"{moment} diverged for parameter {key}",
                )

    def test_a_warm_start_does_not_reproduce_it(self) -> None:
        """The control: weights-only really is different, so the test bites."""
        expected, _ = self._uninterrupted()

        _seed_everything(0)
        model = TinyNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        _train(model, optimizer, self.data, self.STOP_AT)
        weights = model.state_dict()

        warm = TinyNet()
        warm.load_state_dict(weights)
        # Fresh optimizer, epoch counter back at zero: the old behaviour.
        _train(warm, torch.optim.Adam(warm.parameters(), lr=0.01), self.data, self.EPOCHS)
        divergences = [
            not torch.allclose(a, b, atol=1e-6)
            for a, b in zip(expected, _params(warm))
        ]
        self.assertTrue(
            any(divergences),
            "a weights-only warm start matched the uninterrupted run, so this "
            "suite cannot tell resume from warm start",
        )

    def test_the_resume_starts_at_the_right_epoch(self) -> None:
        _, _, loaded = self._interrupted_then_resumed()
        self.assertEqual(loaded["epoch"], self.STOP_AT)
        validation = validate_resume(
            classify_checkpoint(self.directory / "resume.ckpt"),
            _identity(),
            target_total_epochs=self.EPOCHS,
        )
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(validation["start_epoch"], self.STOP_AT + 1)
        self.assertEqual(validation["remaining"], self.EPOCHS - self.STOP_AT)


class TestCheckpointClassification(unittest.TestCase):
    """A3: a weights-only file is never promoted to resumable."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def _resumable(self, **kwargs) -> Path:
        payload = build_checkpoint(
            model_state=TinyNet().state_dict(),
            optimizer_state={"state": {}, "param_groups": []},
            epoch=kwargs.pop("epoch", 5),
            training_identity=_identity(**kwargs.pop("identity", {})),
            **kwargs,
        )
        return save_checkpoint(payload, self.directory / "r.ckpt")

    def test_a_bare_state_dict_is_weights_only(self) -> None:
        path = self.directory / "legacy.ckpt"
        torch.save(TinyNet().state_dict(), str(path))
        verdict = classify_checkpoint(path)
        self.assertEqual(verdict["kind"], KIND_WEIGHTS_ONLY)
        self.assertIn("cannot reproduce an interrupted training", verdict["message"])

    def test_a_weights_only_file_is_refused_for_resume(self) -> None:
        path = self.directory / "legacy.ckpt"
        torch.save(TinyNet().state_dict(), str(path))
        result = validate_resume(classify_checkpoint(path), _identity())
        self.assertFalse(result["ok"])
        self.assertIn("optimizer/epoch/RNG", result["errors"][0])

    def test_a_resumable_checkpoint_is_recognised(self) -> None:
        verdict = classify_checkpoint(self._resumable())
        self.assertEqual(verdict["kind"], KIND_RESUMABLE)
        self.assertEqual(verdict["epoch"], 5)
        self.assertEqual(verdict["schema_version"], SCHEMA_VERSION)

    def test_a_corrupt_file_fails_safely(self) -> None:
        path = self.directory / "broken.ckpt"
        path.write_bytes(b"not a checkpoint at all")
        verdict = classify_checkpoint(path)
        self.assertEqual(verdict["kind"], KIND_UNREADABLE)
        self.assertFalse(validate_resume(verdict, _identity())["ok"])

    def test_a_future_schema_is_refused_rather_than_read_optimistically(self) -> None:
        path = self.directory / "future.ckpt"
        torch.save({SCHEMA_KEY: SCHEMA_VERSION + 1, "epoch": 3}, str(path))
        verdict = classify_checkpoint(path)
        self.assertEqual(verdict["kind"], KIND_FUTURE_VERSION)
        self.assertFalse(validate_resume(verdict, _identity())["ok"])

    def test_the_write_is_atomic_and_leaves_no_scratch_file(self) -> None:
        self._resumable()
        leftovers = [p.name for p in self.directory.iterdir() if p.name.startswith(".ckpt-")]
        self.assertEqual(leftovers, [])

    def test_no_secret_shaped_field_exists_in_the_schema(self) -> None:
        payload = build_checkpoint(
            model_state={}, optimizer_state={}, epoch=1,
            training_identity=_identity(),
        )
        blob = repr(list(payload)) + repr(list(payload["training_identity"]))
        for secret in ("api_key", "token", "password", "secret"):
            self.assertNotIn(secret, blob.lower())


class TestResumeCompatibilityIsEnforced(unittest.TestCase):
    """A4: a checkpoint cannot be resumed into a different experiment."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        payload = build_checkpoint(
            model_state={}, optimizer_state={}, epoch=40,
            target_total_epochs=100, training_identity=_identity(),
        )
        self.checkpoint = classify_checkpoint(
            save_checkpoint(payload, Path(self._tmp.name) / "r.ckpt")
        )

    def test_a_compatible_resume_is_allowed(self) -> None:
        result = validate_resume(self.checkpoint, _identity(), target_total_epochs=100)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["start_epoch"], 41)

    def test_a_different_model_is_blocked(self) -> None:
        result = validate_resume(self.checkpoint, _identity(model_key="batlinet"))
        self.assertFalse(result["ok"])
        self.assertIn("model_key", result["errors"][0])

    def test_a_different_dataset_or_split_is_blocked(self) -> None:
        for field, value in (
            ("dataset", "CALCE"),
            ("split_fingerprint", split_fingerprint(99, 99)),
            ("config_fingerprint", config_fingerprint({"lr": 0.5})),
            ("seed", "7"),
            ("task", "SOH estimation"),
        ):
            with self.subTest(field=field):
                result = validate_resume(self.checkpoint, _identity(**{field: value}))
                self.assertFalse(result["ok"])
                self.assertTrue(any(field in e for e in result["errors"]))

    def test_every_blocking_field_is_actually_checked(self) -> None:
        """Guards the list itself: adding a field must make it enforced."""
        for field in BLOCKING_IDENTITY_FIELDS:
            with self.subTest(field=field):
                result = validate_resume(
                    self.checkpoint, _identity(**{field: "definitely-different"})
                )
                self.assertFalse(result["ok"], f"{field} is listed but not enforced")

    def test_raising_the_epoch_target_is_an_allowed_override(self) -> None:
        self.assertIn("target_total_epochs", ALLOWED_OVERRIDES)
        result = validate_resume(self.checkpoint, _identity(), target_total_epochs=200)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["remaining"], 160)

    def test_a_target_already_reached_is_refused_with_a_usable_message(self) -> None:
        result = validate_resume(self.checkpoint, _identity(), target_total_epochs=40)
        self.assertFalse(result["ok"])
        self.assertIn("Raise the target above 40", result["errors"][0])

    def test_total_target_semantics_not_additional_epochs(self) -> None:
        """A5: 40 done of 100 leaves 60 — not another 100."""
        result = validate_resume(self.checkpoint, _identity(), target_total_epochs=100)
        self.assertEqual(result["completed_epochs"], 40)
        self.assertEqual(result["target_total_epochs"], 100)
        self.assertEqual(result["remaining"], 60)


class TestRngCaptureAndRestore(unittest.TestCase):
    """A6: the generators that drive training are really restored."""

    def test_a_captured_state_reproduces_the_same_draws(self) -> None:
        _seed_everything(3)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.randn(3).tolist())

        _seed_everything(4321)
        for _ in range(11):
            torch.randn(2)
        self.assertEqual(restore_rng_state(state), [])
        actual = (random.random(), float(np.random.rand()), torch.randn(3).tolist())
        self.assertEqual(expected, actual)

    def test_a_missing_rng_state_is_reported_not_ignored(self) -> None:
        warnings = restore_rng_state(None)
        self.assertTrue(warnings)
        self.assertIn("not deterministic", warnings[0])

    def test_cuda_states_are_empty_rather_than_invented_without_a_gpu(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("this assertion is about the CPU-only case")
        self.assertEqual(capture_rng_state()["torch_cuda"], [])


if __name__ == "__main__":
    unittest.main()
