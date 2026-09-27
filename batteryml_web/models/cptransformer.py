from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from batteryml.builders import FEATURE_EXTRACTORS, MODELS
from batteryml.data.battery_data import BatteryData
from batteryml.feature.base import BaseFeatureExtractor
from batteryml.models.nn_model import NNModel


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BATTERYLIFE_ROOT = _REPO_ROOT / "third_party" / "BatteryLife"
_CPTRANSFORMER_SOURCE = _BATTERYLIFE_ROOT / "models" / "CPTransformer.py"

if not _CPTRANSFORMER_SOURCE.is_file():
    # Reached only when something imports this module directly. The normal
    # path — batteryml_web.models.register_extension_models() — consults
    # EXTERNAL_SOURCES first and skips this module when the vendored source
    # is absent, so one un-vendored optional wrapper no longer stops the
    # whole registry (and with it the CLI) on a fresh clone.
    raise FileNotFoundError(
        f"BatteryLife CPTransformer source not found: "
        f"{_CPTRANSFORMER_SOURCE}. This wrapper needs the BatteryLife "
        f"repository vendored at third_party/BatteryLife (pinned commit in "
        f"docs/EXTERNAL_MODEL_VERSIONS.txt); third_party/ is gitignored, so "
        f"a fresh clone does not carry it. Every other model runs without "
        f"it — CPTransformer alone is unavailable until it is vendored."
    )

if str(_BATTERYLIFE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BATTERYLIFE_ROOT))

from models import CPTransformer  # noqa: E402


@FEATURE_EXTRACTORS.register()
class BatteryLifeCycleCurveFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self,
        early_cycle_threshold: int = 100,
        charge_discharge_length: int = 300,
        current_threshold_c: float = 0.01,
    ):
        if early_cycle_threshold <= 0:
            raise ValueError("early_cycle_threshold must be positive.")

        if charge_discharge_length <= 0:
            raise ValueError("charge_discharge_length must be positive.")

        if charge_discharge_length % 2 != 0:
            raise ValueError(
                "charge_discharge_length must be even."
            )

        self.early_cycle_threshold = int(early_cycle_threshold)
        self.charge_discharge_length = int(charge_discharge_length)
        self.current_threshold_c = float(current_threshold_c)

    @staticmethod
    def _backfill_invalid_rows(
        voltage,
        current,
        charge_capacity,
        discharge_capacity,
        time_s,
    ):
        arrays = np.column_stack([
            voltage,
            current,
            charge_capacity,
            discharge_capacity,
            time_s,
        ]).astype(np.float64, copy=True)

        bad_rows = (
            (arrays[:, 2] < 0)
            | (arrays[:, 3] < 0)
            | (~np.isfinite(arrays).all(axis=1))
        )

        arrays[bad_rows, :] = np.nan

        for col in range(arrays.shape[1]):
            values = arrays[:, col]
            next_value = np.nan

            for idx in range(len(values) - 1, -1, -1):
                if np.isfinite(values[idx]):
                    next_value = values[idx]
                elif np.isfinite(next_value):
                    values[idx] = next_value

        if not np.isfinite(arrays).all():
            raise ValueError(
                "A cycle still contains non-finite values "
                "after backward fill."
            )

        return tuple(arrays[:, i] for i in range(arrays.shape[1]))

    def _resample_segment(
        self,
        voltage,
        current,
        capacity,
    ):
        target_length = self.charge_discharge_length // 2

        if len(voltage) < 2:
            raise ValueError(
                "A charge/discharge segment contains "
                "fewer than two samples."
            )

        raw_base = np.arange(
            1,
            len(voltage) + 1,
            dtype=np.float64,
        )

        interp_base = np.linspace(
            1,
            len(voltage) + 1,
            num=target_length,
            endpoint=True,
        )

        return (
            np.interp(interp_base, raw_base, voltage),
            np.interp(interp_base, raw_base, current),
            np.interp(interp_base, raw_base, capacity),
        )

    def _process_cycle(
        self,
        cell_data: BatteryData,
        cycle_index: int,
    ):
        cycle = cell_data.cycle_data[cycle_index]

        voltage = np.asarray(
            cycle.voltage_in_V,
            dtype=np.float64,
        )

        current = np.asarray(
            cycle.current_in_A,
            dtype=np.float64,
        )

        charge_capacity = np.asarray(
            cycle.charge_capacity_in_Ah,
            dtype=np.float64,
        )

        discharge_capacity = np.asarray(
            cycle.discharge_capacity_in_Ah,
            dtype=np.float64,
        )

        time_s = np.asarray(
            cycle.time_in_s,
            dtype=np.float64,
        )

        lengths = {
            len(voltage),
            len(current),
            len(charge_capacity),
            len(discharge_capacity),
            len(time_s),
        }

        if len(lengths) != 1:
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "array lengths do not match."
            )

        if len(voltage) < 4:
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "too few measurements."
            )

        nominal_capacity = float(
            cell_data.nominal_capacity_in_Ah
        )

        if nominal_capacity <= 0:
            raise ValueError(
                f"{cell_data.cell_id}: "
                "nominal_capacity_in_Ah must be positive."
            )

        (
            voltage,
            current,
            charge_capacity,
            discharge_capacity,
            time_s,
        ) = self._backfill_invalid_rows(
            voltage,
            current,
            charge_capacity,
            discharge_capacity,
            time_s,
        )

        current_in_c = current / nominal_capacity

        charge_indices = np.flatnonzero(
            current_in_c >= self.current_threshold_c
        )

        discharge_indices = np.flatnonzero(
            current_in_c <= -self.current_threshold_c
        )

        if len(charge_indices) == 0:
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "no charge segment detected."
            )

        if len(discharge_indices) == 0:
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "no discharge segment detected."
            )

        charge_end_index = int(charge_indices[-1])

        if charge_end_index < 2:
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "charge segment is too short."
            )

        charge_voltage = voltage[:charge_end_index]
        charge_current = current[:charge_end_index]
        charge_capacity_part = charge_capacity[:charge_end_index]

        discharge_voltage = voltage[charge_end_index:]
        discharge_current = current[charge_end_index:]
        discharge_capacity_part = discharge_capacity[charge_end_index:]

        discharge_keep = (
            np.abs(discharge_current / nominal_capacity)
            > self.current_threshold_c
        )

        discharge_voltage = discharge_voltage[discharge_keep]
        discharge_current = discharge_current[discharge_keep]
        discharge_capacity_part = (
            discharge_capacity_part[discharge_keep]
        )

        (
            charge_voltage,
            charge_current,
            charge_capacity_part,
        ) = self._resample_segment(
            charge_voltage,
            charge_current,
            charge_capacity_part,
        )

        (
            discharge_voltage,
            discharge_current,
            discharge_capacity_part,
        ) = self._resample_segment(
            discharge_voltage,
            discharge_current,
            discharge_capacity_part,
        )

        voltage_curve = np.concatenate([
            charge_voltage,
            discharge_voltage,
        ])

        current_curve = np.concatenate([
            charge_current,
            discharge_current,
        ])

        capacity_curve = np.concatenate([
            charge_capacity_part,
            discharge_capacity_part,
        ])

        voltage_scale = float(np.max(voltage_curve))

        if voltage_scale <= 0:
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "invalid maximum voltage."
            )

        voltage_curve = voltage_curve / voltage_scale
        current_curve = current_curve / nominal_capacity
        capacity_curve = capacity_curve / nominal_capacity

        curve = np.stack(
            [
                voltage_curve,
                current_curve,
                capacity_curve,
            ],
            axis=0,
        )

        if curve.shape != (
            3,
            self.charge_discharge_length,
        ):
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                f"unexpected curve shape {curve.shape}."
            )

        if not np.isfinite(curve).all():
            raise ValueError(
                f"{cell_data.cell_id} cycle {cycle_index + 1}: "
                "feature contains NaN/Inf."
            )

        return curve.astype(np.float32)

    def process_cell(
        self,
        cell_data: BatteryData,
    ) -> torch.Tensor:

        feature = np.zeros(
            (
                self.early_cycle_threshold,
                4,
                self.charge_discharge_length,
            ),
            dtype=np.float32,
        )

        usable_cycles = min(
            len(cell_data.cycle_data),
            self.early_cycle_threshold,
        )

        skipped_cycles = []

        for cycle_index in range(usable_cycles):
            try:
                curve = self._process_cycle(
                    cell_data,
                    cycle_index,
                )
            except ValueError as exc:
                skipped_cycles.append(
                    (
                        cycle_index + 1,
                        str(exc),
                    )
                )
                continue

            feature[
                cycle_index,
                :3,
                :
            ] = curve

            feature[
                cycle_index,
                3,
                :
            ] = 1.0

        valid_cycles = int(
            feature[:, 3, 0].sum()
        )

        if valid_cycles == 0:
            raise ValueError(
                f"{cell_data.cell_id}: "
                "no valid early cycles were available "
                "for CPTransformer."
            )

        if skipped_cycles:
            warnings.warn(
                f"{cell_data.cell_id}: masked "
                f"{len(skipped_cycles)}/{self.early_cycle_threshold} "
                "malformed early cycle(s).",
                RuntimeWarning,
            )

        return torch.from_numpy(feature)


@MODELS.register()
class CPTransformerRULPredictor(NNModel):

    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 100,
        workspace=None,
        evaluate_freq: int = 10,
        checkpoint_freq: int = 100,
        train_batch_size=None,
        test_batch_size=None,
        lr: float = 5e-5,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 6,
        d_layers: int = 4,
        n_heads: int = 4,
        factor: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        early_cycle_threshold: int = 100,
        charge_discharge_length: int = 300,
        output_num: int = 1,
    ):
        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            workspace=workspace,
            evaluate_freq=evaluate_freq,
            checkpoint_freq=checkpoint_freq,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            lr=lr,
        )

        self.early_cycle_threshold = int(
            early_cycle_threshold
        )

        self.charge_discharge_length = int(
            charge_discharge_length
        )

        configs = SimpleNamespace(
            d_model=int(d_model),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            d_layers=int(d_layers),
            n_heads=int(n_heads),
            factor=int(factor),
            dropout=float(dropout),
            activation=str(activation),
            early_cycle_threshold=(
                self.early_cycle_threshold
            ),
            charge_discharge_length=(
                self.charge_discharge_length
            ),
            output_num=int(output_num),
        )

        self.network = CPTransformer.Model(configs)

    def forward(
        self,
        feature: torch.Tensor,
        label: torch.Tensor | None = None,
        return_loss: bool = False,
        **kwargs,
    ):
        if feature.ndim != 4:
            raise ValueError(
                "CPTransformer feature must have shape "
                "[batch, cycles, 4, samples]."
            )

        if feature.shape[1] != (
            self.early_cycle_threshold
        ):
            raise ValueError(
                "Unexpected number of cycles: "
                f"{feature.shape[1]} != "
                f"{self.early_cycle_threshold}"
            )

        if feature.shape[2] != 4:
            raise ValueError(
                "Feature channel dimension must be 4."
            )

        if feature.shape[3] != (
            self.charge_discharge_length
        ):
            raise ValueError(
                "Unexpected curve length: "
                f"{feature.shape[3]} != "
                f"{self.charge_discharge_length}"
            )

        cycle_curve_data = feature[:, :, :3, :]
        curve_attn_mask = feature[:, :, 3, 0]

        cycle_curve_data = (
            cycle_curve_data
            * curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
        )

        prediction = self.network(
            cycle_curve_data,
            curve_attn_mask,
        ).reshape(-1)

        if not return_loss:
            return prediction

        if label is None:
            raise ValueError(
                "label is required when return_loss=True."
            )

        target = label.reshape(-1)

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction/label shape mismatch: "
                f"{prediction.shape} vs {target.shape}"
            )

        return F.mse_loss(
            prediction,
            target,
        )
