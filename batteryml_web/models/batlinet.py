from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from batteryml.builders import FEATURE_EXTRACTORS, MODELS
from batteryml.data.battery_data import BatteryData
from batteryml.feature.base import BaseFeatureExtractor
from batteryml.models.base import BaseModel


# ============================================================
# Feature extractor
# ============================================================

@FEATURE_EXTRACTORS.register()
class BatLiNetFeatureExtractor(BaseFeatureExtractor):
    """
    BatteryML adapter for BatLiNet early-cycle features.

    Output shape for one battery:
        [6, max_cycle_index - min_cycle_index + 1, interp_dim]

    Six channels:
        0: V(Q_charge)
        1: V(Q_discharge)
        2: I(Q_charge)
        3: I(Q_discharge)
        4: delta V(Q)
        5: R(Q)

    The selected diff_base cycle is subtracted from all cycles to
    construct the intra-cell degradation representation.
    """

    def __init__(
        self,
        interp_dim: int = 1000,
        diff_base: int = 5,
        min_cycle_index: int = 1,
        max_cycle_index: int = 20,
        max_capacity: float = 1.2,
        current_threshold_a: float = 0.1,
    ):
        self.interp_dim = int(interp_dim)
        self.diff_base = int(diff_base)
        self.min_cycle_index = int(min_cycle_index)
        self.max_cycle_index = int(max_cycle_index)
        self.max_capacity = float(max_capacity)
        self.current_threshold_a = float(current_threshold_a)

        self.num_cycles = (
            self.max_cycle_index
            - self.min_cycle_index
            + 1
        )

        if self.num_cycles <= 0:
            raise ValueError(
                "max_cycle_index must be >= min_cycle_index"
            )

        if not 0 <= self.diff_base < self.num_cycles:
            raise ValueError(
                f"diff_base={self.diff_base} must index one of "
                f"{self.num_cycles} selected cycles."
            )

    def _interpolate(
        self,
        capacity,
        values,
    ):
        capacity = np.asarray(
            capacity,
            dtype=np.float64,
        )

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        valid = (
            np.isfinite(capacity)
            & np.isfinite(values)
        )

        capacity = capacity[valid]
        values = values[valid]

        if len(capacity) < 2:
            raise ValueError(
                "fewer than two valid samples for interpolation"
            )

        order = np.argsort(capacity)

        capacity = capacity[order]
        values = values[order]

        # Remove duplicated capacity values.
        capacity, unique_idx = np.unique(
            capacity,
            return_index=True,
        )

        values = values[unique_idx]

        if len(capacity) < 2:
            raise ValueError(
                "fewer than two unique capacity samples"
            )

        grid = np.linspace(
            0.0,
            self.max_capacity,
            self.interp_dim,
        )

        return np.interp(
            grid,
            capacity,
            values,
            left=values[0],
            right=values[-1],
        )

    def _process_cycle(
        self,
        cell_data: BatteryData,
        cycle,
    ):
        nominal_capacity = float(
            cell_data.nominal_capacity_in_Ah
        )

        if nominal_capacity <= 0:
            raise ValueError(
                f"{cell_data.cell_id}: invalid nominal capacity"
            )

        current = np.asarray(
            cycle.current_in_A,
            dtype=np.float64,
        )

        voltage = np.asarray(
            cycle.voltage_in_V,
            dtype=np.float64,
        )

        qc = (
            np.asarray(
                cycle.charge_capacity_in_Ah,
                dtype=np.float64,
            )
            / nominal_capacity
        )

        qd = (
            np.asarray(
                cycle.discharge_capacity_in_Ah,
                dtype=np.float64,
            )
            / nominal_capacity
        )

        n = {
            len(current),
            len(voltage),
            len(qc),
            len(qd),
        }

        if len(n) != 1:
            raise ValueError(
                "current/voltage/capacity lengths differ"
            )

        charge_mask = (
            current > self.current_threshold_a
        )

        discharge_mask = (
            current < -self.current_threshold_a
        )

        if charge_mask.sum() < 2:
            raise ValueError(
                "insufficient charge samples"
            )

        if discharge_mask.sum() < 2:
            raise ValueError(
                "insufficient discharge samples"
            )

        qc_part = qc[charge_mask]
        qd_part = qd[discharge_mask]

        ic = current[charge_mask]
        id_ = current[discharge_mask]

        vc = voltage[charge_mask]
        vd = voltage[discharge_mask]

        # V(Qc), V(Qd), I(Qc), I(Qd)
        v_charge = self._interpolate(
            qc_part,
            vc,
        )

        v_discharge = self._interpolate(
            qd_part,
            vd,
        )

        i_charge = self._interpolate(
            qc_part,
            ic,
        )

        i_discharge = self._interpolate(
            qd_part,
            id_,
        )

        # Align charge/discharge directions.
        v_discharge_rev = v_discharge[::-1]
        i_discharge_rev = i_discharge[::-1]

        delta_v = (
            v_charge
            - v_discharge_rev
        )

        resistance = (
            delta_v
            / (
                i_charge
                - i_discharge_rev
                + 1e-3
            )
        )

        result = np.stack(
            [
                v_charge,
                v_discharge,
                i_charge,
                i_discharge,
                delta_v,
                resistance,
            ],
            axis=0,
        )

        if not np.isfinite(result).all():
            raise ValueError(
                "BatLiNet feature contains NaN/Inf"
            )

        return result.astype(np.float32)

    def process_cell(
        self,
        cell_data: BatteryData,
    ) -> torch.Tensor:

        by_number = {
            int(cycle.cycle_number): cycle
            for cycle in cell_data.cycle_data
            if np.isfinite(cycle.cycle_number)
        }

        features = []

        for cycle_number in range(
            self.min_cycle_index,
            self.max_cycle_index + 1,
        ):
            cycle = by_number.get(cycle_number)

            if cycle is None:
                raise ValueError(
                    f"{cell_data.cell_id}: "
                    f"missing cycle {cycle_number}"
                )

            try:
                cycle_feature = self._process_cycle(
                    cell_data,
                    cycle,
                )

            except ValueError as exc:
                raise ValueError(
                    f"{cell_data.cell_id} "
                    f"cycle {cycle_number}: {exc}"
                ) from exc

            features.append(
                cycle_feature
            )

        # [cycle, channel, point]
        feature = np.stack(
            features,
            axis=0,
        )

        # Intra-cell difference.
        feature = (
            feature
            - feature[[self.diff_base]]
        )

        # BatLiNet Conv2D expects:
        # [channel, cycle, point]
        feature = np.transpose(
            feature,
            (1, 0, 2),
        )

        expected = (
            6,
            self.num_cycles,
            self.interp_dim,
        )

        if feature.shape != expected:
            raise ValueError(
                f"{cell_data.cell_id}: "
                f"{feature.shape} != {expected}"
            )

        return torch.from_numpy(
            feature.copy()
        ).float()


# ============================================================
# Original BatLiNet architecture
# ============================================================

class _ConvModule(nn.Module):

    def __init__(
        self,
        din,
        dout,
        kernel_size,
        act_fn="relu",
    ):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(
            din,
            dout,
            kernel_size,
        )

        self.pool1 = nn.AvgPool2d(
            kernel_size,
        )

        self.conv2 = nn.Conv2d(
            dout,
            dout,
            kernel_size,
        )

        self.pool2 = nn.AvgPool2d(
            kernel_size,
        )

        self.act_fn = getattr(
            torch,
            act_fn,
        )

    def output_shape(
        self,
        H,
        W,
    ):
        kh, kw = self.kernel_size

        H = H - kh + 1
        W = W - kw + 1

        H = int(
            (H - kh) / kh + 1
        )
        W = int(
            (W - kw) / kw + 1
        )

        H = H - kh + 1
        W = W - kw + 1

        H = int(
            (H - kh) / kh + 1
        )
        W = int(
            (W - kw) / kw + 1
        )

        return H, W

    def forward(
        self,
        x,
    ):
        x = self.conv1(x)
        x = self.act_fn(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act_fn(x)

        x = self.pool2(x)

        return x


def _build_module(
    in_channels,
    channels,
    input_height,
    input_width,
    kernel_size,
    act_fn,
):
    encoder = _ConvModule(
        in_channels,
        channels,
        kernel_size,
        act_fn,
    )

    H, W = encoder.output_shape(
        input_height,
        input_width,
    )

    projection = nn.Conv2d(
        channels,
        channels,
        (H, W),
    )

    return nn.Sequential(
        encoder,
        projection,
        nn.ReLU(),
    )


class _BatLiNetCore(nn.Module):

    def __init__(
        self,
        in_channels=6,
        channels=32,
        input_height=20,
        input_width=1000,
        alpha=0.5,
        kernel_size=1,
        act_fn="relu",
    ):
        super().__init__()

        if isinstance(
            kernel_size,
            int,
        ):
            kernel_size = (
                kernel_size,
                kernel_size,
            )

        self.alpha = float(alpha)
        self.channels = int(channels)

        self.ori_module = _build_module(
            in_channels,
            channels,
            input_height,
            input_width,
            kernel_size,
            act_fn,
        )

        self.sup_module = _build_module(
            in_channels,
            channels,
            input_height,
            input_width,
            kernel_size,
            act_fn,
        )

        # Same shared linear regressor design.
        self.fc = nn.Linear(
            channels,
            1,
            bias=False,
        )

    def forward(
        self,
        feature,
        label,
        support_feature,
        support_label,
    ):
        B, S, C, H, W = (
            support_feature.size()
        )

        x_ori = self.ori_module(
            feature
        )

        x_sup = self.sup_module(
            support_feature.view(
                -1,
                C,
                H,
                W,
            )
        )

        y_ori = self.fc(
            x_ori.view(
                B,
                self.channels,
            )
        ).view(-1)

        y_sup = self.fc(
            x_sup.view(
                B,
                S,
                self.channels,
            )
        ).view(
            B,
            S,
        )

        # Inter-cell lifetime estimate:
        # Δlife + known reference life
        y_sup = (
            y_sup
            + support_label.view(
                B,
                S,
            )
        )

        if self.training:
            y_sup_agg = (
                y_sup.mean(
                    dim=1
                )
            )
        else:
            # Match local BatLiNet source:
            # median aggregation at inference.
            y_sup_agg = (
                y_sup.median(
                    dim=1
                )[0]
            )

        intra_loss = (
            (
                y_ori.view(-1)
                - label.view(-1)
            ) ** 2
        ).mean()

        inter_loss = (
            (
                y_sup_agg.view(-1)
                - label.view(-1)
            ) ** 2
        ).mean()

        loss = (
            (1.0 - self.alpha)
            * intra_loss
            +
            self.alpha
            * inter_loss
        )

        outputs = (
            (1.0 - self.alpha)
            * y_ori
            +
            self.alpha
            * y_sup_agg
        )

        return outputs, loss


# ============================================================
# BatteryML model wrapper
# ============================================================

@MODELS.register()
class BatLiNetRULPredictor(
    BaseModel,
    nn.Module,
):

    def __init__(
        self,
        workspace=None,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 5e-5,
        evaluate_freq: int = 10,
        checkpoint_freq: int = 10,
        channels: int = 32,
        alpha: float = 0.5,
        train_support_size: int = 8,
        test_support_size: int = 16,
        input_height: int = 20,
        input_width: int = 1000,
        in_channels: int = 6,
    ):
        nn.Module.__init__(self)
        BaseModel.__init__(
            self,
            workspace,
        )

        self.train_epochs = int(
            epochs
        )

        self.batch_size = int(
            batch_size
        )

        self.lr = float(lr)

        self.evaluate_freq = int(
            evaluate_freq
        )

        self.checkpoint_freq = int(
            checkpoint_freq
        )

        self.train_support_size = int(
            train_support_size
        )

        self.test_support_size = int(
            test_support_size
        )

        self.network = _BatLiNetCore(
            in_channels=in_channels,
            channels=channels,
            input_height=input_height,
            input_width=input_width,
            alpha=alpha,
        )

    def _sample_support_indices(
        self,
        target_indices,
        total_count,
        support_size,
    ):
        B = len(target_indices)

        if total_count < 2:
            raise ValueError(
                "BatLiNet requires at least "
                "two training cells."
            )

        S = min(
            support_size,
            total_count - 1,
        )

        result = []

        for target_index in target_indices.tolist():

            candidates = torch.arange(
                total_count,
                device=target_indices.device,
            )

            candidates = candidates[
                candidates != target_index
            ]

            perm = torch.randperm(
                len(candidates),
                device=target_indices.device,
            )

            selected = candidates[
                perm[:S]
            ]

            result.append(
                selected
            )

        return torch.stack(
            result,
            dim=0,
        )

    def _training_support(
        self,
        features,
        labels,
        target_indices,
    ):
        support_indices = (
            self._sample_support_indices(
                target_indices,
                len(features),
                self.train_support_size,
            )
        )

        target = features[
            target_indices
        ]

        refs = features[
            support_indices
        ]

        # Inter-cell feature:
        # target cell - reference cell
        support_feature = (
            target.unsqueeze(1)
            - refs
        )

        support_label = labels[
            support_indices
        ]

        return (
            support_feature,
            support_label,
        )

    def _test_support(
        self,
        target_features,
        train_features,
        train_labels,
    ):
        S = min(
            self.test_support_size,
            len(train_features),
        )

        # Deterministic reference pool.
        support_features = (
            train_features[:S]
        )

        support_labels = (
            train_labels[:S]
        )

        B = len(
            target_features
        )

        refs = (
            support_features
            .unsqueeze(0)
            .expand(
                B,
                -1,
                -1,
                -1,
                -1,
            )
        )

        support_feature = (
            target_features.unsqueeze(1)
            - refs
        )

        support_label = (
            support_labels
            .unsqueeze(0)
            .expand(
                B,
                -1,
            )
        )

        return (
            support_feature,
            support_label,
        )

    def fit(
        self,
        dataset,
        timestamp=None,
    ):
        device = (
            dataset.train_data.feature.device
        )

        self.network.to(
            device
        )

        features = (
            dataset.train_data.feature
        )

        labels = (
            dataset.train_data.label
            .reshape(-1)
        )

        n = len(labels)

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.lr,
        )

        timestamp = (
            timestamp
            or "UnknownTime"
        )

        global_seed = int(
            torch.initial_seed()
            % (2**31)
        )

        latest = None

        for epoch in tqdm(
            range(self.train_epochs),
            desc="Training BatLiNet",
        ):
            self.network.train()

            order = torch.randperm(
                n,
                device=device,
            )

            for start in range(
                0,
                n,
                self.batch_size,
            ):
                indices = order[
                    start:
                    start
                    + self.batch_size
                ]

                x = features[
                    indices
                ]

                y = labels[
                    indices
                ]

                (
                    sup_x,
                    sup_y,
                ) = self._training_support(
                    features,
                    labels,
                    indices,
                )

                _, loss = self.network(
                    x,
                    y,
                    sup_x,
                    sup_y,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (
                self.checkpoint_freq > 0
                and (epoch + 1)
                % self.checkpoint_freq
                == 0
                and self.workspace is not None
            ):
                filename = (
                    f"{timestamp}_"
                    f"seed_{global_seed}_"
                    f"epoch_{epoch + 1}.ckpt"
                )

                latest = (
                    self.workspace
                    / filename
                )

                self.dump_checkpoint(
                    latest
                )

            if (
                self.evaluate_freq > 0
                and (epoch + 1)
                % self.evaluate_freq
                == 0
            ):
                pred = self.predict(
                    dataset
                )

                rmse = dataset.evaluate(
                    pred,
                    "RMSE",
                )

                print(
                    f"[{epoch + 1}/"
                    f"{self.train_epochs}] "
                    f"RMSE {rmse:.2f}",
                    flush=True,
                )

        if (
            latest is not None
            and self.workspace is not None
        ):
            self.link_latest_checkpoint(
                latest
            )

        return self

    @torch.no_grad()
    def predict(
        self,
        dataset,
        data_type="test",
    ):
        self.network.eval()

        train_features = (
            dataset.train_data.feature
        )

        train_labels = (
            dataset.train_data.label
            .reshape(-1)
        )

        if data_type == "train":
            target_features = (
                dataset.train_data.feature
            )
        else:
            target_features = (
                dataset.test_data.feature
            )

        predictions = []

        for start in range(
            0,
            len(target_features),
            self.batch_size,
        ):
            x = target_features[
                start:
                start
                + self.batch_size
            ]

            (
                sup_x,
                sup_y,
            ) = self._test_support(
                x,
                train_features,
                train_labels,
            )

            # label is required only for the auxiliary
            # loss calculation inside the original
            # forward. It does not enter prediction.
            dummy_label = torch.zeros(
                len(x),
                device=x.device,
                dtype=x.dtype,
            )

            outputs, _ = self.network(
                x,
                dummy_label,
                sup_x,
                sup_y,
            )

            predictions.append(
                outputs.reshape(-1)
            )

        return torch.cat(
            predictions,
            dim=0,
        )

    def dump_checkpoint(
        self,
        path,
    ):
        torch.save(
            self.network.state_dict(),
            path,
        )

    def load_checkpoint(
        self,
        path,
    ):
        state = torch.load(
            path,
            map_location="cpu",
        )

        self.network.load_state_dict(
            state
        )

    def to(
        self,
        device,
    ):
        self.network.to(
            device
        )
        return self
