from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


MODEL_DISPLAY_NAMES = {
    "batlinet": "BatLiNet — Inter-cell Lifetime Prediction",

    "cptransformer": "CPTransformer — CyclePatch Transformer",

    "xgb": "XGBoost — Extreme Gradient Boosting",
    "xgboost": "XGBoost — Extreme Gradient Boosting",
    "rf": "Random Forest",
    "random forest": "Random Forest",
    "random": "Random Forest",
    "mlp": "MLP — Multilayer Perceptron",
    "cnn": "CNN — Convolutional Neural Network",
    "lstm": "LSTM — Long Short-Term Memory",
    "transformer": "Transformer",
    "gpr": "GPR — Gaussian Process Regression",
    "plsr": "PLSR — Partial Least Squares Regression",
    "pcr": "PCR — Principal Component Regression",
    "ridge": "Ridge Regression",
    "linear regression": "Linear Regression",
    "svm": "SVM — Support Vector Regression",
    "elastic net": "Elastic Net Regression",
    "extra trees": "Extra Trees",
    "knn": "KNN — k-Nearest Neighbors Regression",
    "lasso": "Lasso Regression",
    "huber": "Huber Regression",
    "kernel ridge": "Kernel Ridge Regression",
    "dummy": "Dummy Baseline",
    "discharge model": "Discharge Feature Model",
    "full model": "Full Feature Model",
    "variance model": "Variance Feature Model",
}

MODEL_KEY_ALIASES = {
    "batlinet": "batlinet",

    "cptransformer": "cptransformer",

    "xgboost": "xgb",
    "rf": "rf",
    "random forest": "rf",
    "random": "rf",
    "mlp": "mlp",
    "cnn": "cnn",
    "lstm": "lstm",
    "transformer": "transformer",
    "gpr": "gpr",
    "plsr": "plsr",
    "pcr": "pcr",
    "ridge": "ridge",
    "linear regression": "linear_regression",
    "svm": "svm",
    "elastic net": "elastic_net",
    "extra trees": "extra_trees",
    "knn": "knn",
    "lasso": "lasso",
    "huber": "huber",
    "kernel ridge": "kernel_ridge",
    "dummy": "dummy",
    "discharge model": "discharge_model",
    "full model": "full_model",
    "variance model": "variance_model",
}

# Registered BatteryML predictor class name -> lookup key in the tables above.
# Used as the fallback when a config does not live in a per-model directory.
# The three feature-family directories (discharge/full/variance model) all
# declare LinearRegressionRULPredictor, so the class name alone cannot tell
# them apart; the directory name is authoritative and is consulted first.
MODEL_CLASS_KEYS = {
    "batlinetrulpredictor": "batlinet",

    "cptransformerrulpredictor": "cptransformer",

    "xgboostrulpredictor": "xgb",
    "randomforestrulpredictor": "rf",
    "mlprulpredictor": "mlp",
    "cnnrulpredictor": "cnn",
    "lstmrulpredictor": "lstm",
    "transformerrulpredictor": "transformer",
    "gaussianprocessrulpredictor": "gpr",
    "plsrrulpredictor": "plsr",
    "pcrrulpredictor": "pcr",
    "ridgerulpredictor": "ridge",
    "linearregressionrulpredictor": "linear regression",
    "svmrulpredictor": "svm",
    "elasticnetrulpredictor": "elastic net",
    "extratreesrulpredictor": "extra trees",
    "knnrulpredictor": "knn",
    "lassorulpredictor": "lasso",
    "huberrulpredictor": "huber",
    "kernelridgerulpredictor": "kernel ridge",
    "dummyrulpredictor": "dummy",
}


def normalize_model_token(text: str) -> str:
    """Fold separator and case variants into a single lookup form.

    The configs/ tree names its per-model directories with underscores
    ("discharge_model"), while MODEL_DISPLAY_NAMES / MODEL_KEY_ALIASES spell
    the same names with spaces ("discharge model"). Normalizing both sides
    through this function removes the mismatch at the source instead of
    special-casing individual names.
    """
    return " ".join(str(text).replace("_", " ").replace("-", " ").lower().split())


_NORMALIZED_DISPLAY_NAMES = {
    normalize_model_token(key): display for key, display in MODEL_DISPLAY_NAMES.items()
}
_NORMALIZED_KEY_ALIASES = {
    normalize_model_token(key): alias for key, alias in MODEL_KEY_ALIASES.items()
}


@dataclass(frozen=True)
class ConfigEntry:
    dataset: str
    task: str
    model_key: str
    model_display: str
    split_preset: str
    split_label: str
    config_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def config_exists(self) -> bool:
        return self.config_path.exists()


class ConfigRegistry:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.configs_root = self.repo_root / "configs"
        self.entries: list[ConfigEntry] = []

    def build(self) -> "ConfigRegistry":
        self.entries = []
        for path in sorted(self.configs_root.rglob("*.yaml")):
            entry = self._parse_entry(path)
            if entry:
                self.entries.append(entry)
        return self

    def _parse_entry(self, path: Path) -> ConfigEntry | None:
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        data: dict[str, Any] = {}
        try:
            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            pass
        # A config that declares itself a legacy example is not offered as a
        # runnable choice. configs/soh/dummy/* demonstrate label plumbing with
        # DummyRegressor and one label per cell; listing them as the platform's
        # SOH support would claim a capability that does not exist yet. The
        # files stay, and the notebook that uses them keeps working — they are
        # only withheld from the catalog.
        if isinstance(data, dict) and bool(data.get("legacy_example")):
            return None
        hay = f"{path} {path.name} {text}".lower()
        dataset = self._infer_dataset(hay, data)
        task = self._infer_task(hay, data)
        model_key, model_display = self._infer_model(path, data)
        split_preset = self._infer_split_preset(path)
        split_label = self._split_label(split_preset)
        if not (dataset and task and model_key and split_preset):
            return None
        return ConfigEntry(dataset, task, model_key, model_display, split_preset, split_label, path, {"yaml": data})

    #: Datasets a config's ``cell_data_path`` may name. Order is irrelevant
    #: now that matching is structural; it was not when this searched the
    #: whole YAML, where the first entry won whatever the config said.
    _DATASETS = ("MATR", "HUST", "RWTH", "CALCE", "SNL", "UL_PUR", "HNEI")

    def _infer_dataset(self, hay: str, data: dict[str, Any]) -> str | None:
        """The dataset this config trains on, from ``cell_data_path``.

        Structure, not substrings — the same repair ``_infer_task`` needed,
        for the same reason and with a worse symptom.

        This used to search ``f"{path} {path.name} {text}"`` for each dataset
        name in order, with ``MATR`` first. Every CNN config names
        ``VoltageCapacityMatrixFeatureExtractor``, and *Matrix* contains
        **matr** — so all eight CNN configs were registered as MATR,
        including ``hust.yaml`` (``data/processed/HUST``) and ``snl.yaml``
        (``data/processed/SNL``). Two consequences, both observed:

        * CNN + MATR offered eight configs of which five read another
          dataset's directory. A plan built from one of those says MATR and
          runs HUST, so it is refused by the data-requiring validator, or —
          if it starts — dies on ``AssertionError: data/processed/HUST``.
          Whether a CNN run worked depended on which config was picked.
        * ``HUST`` and ``SNL`` offered no CNN at all, although each has a
          config written for it.

        ``cell_data_path`` is what ``Task.build`` actually loads, so it
        cannot disagree with the run the way a filename or a class name can.

        A config whose ``cell_data_path`` is a **list** spans several
        datasets (``mix``, ``cruh``, ``crush``). It is not any one dataset's
        config and is not offered under one; it stays on disk and remains
        runnable from the CLI. Returning ``MATR`` for ``mix.yaml`` because
        MATR appears among its six directories is exactly the mislabelling
        this function exists to stop.
        """
        split = data.get("train_test_split", {}) if isinstance(data, dict) else {}
        cdp = split.get("cell_data_path") if isinstance(split, dict) else None
        if isinstance(cdp, (list, tuple)):
            return None
        if cdp:
            # The last path segment, so "data/processed/MATR" resolves on
            # "MATR" and never on a parent directory that happens to carry a
            # dataset name.
            leaf = str(cdp).rstrip("/").rsplit("/", 1)[-1].upper()
            for ds in self._DATASETS:
                if leaf == ds:
                    return ds
            return None
        # No declared data path. Fall back to the *filename* alone — never
        # the file's text, which is what produced the Matrix/MATR collision.
        stem = Path(str(hay).strip().split()[0]).stem.upper() if hay else ""
        for ds in self._DATASETS:
            if stem == ds or stem.startswith(f"{ds}_"):
                return ds
        return None

    #: Label annotator class name -> canonical task. This is the authoritative
    #: signal: the annotator *is* what the config trains against, so it cannot
    #: disagree with the task the way a filename or a stray parameter can.
    _TASK_BY_ANNOTATOR = {
        "sohlabelannotator": "SOH estimation",
        "cyclesohlabelannotator": "SOH estimation",
        "rullabelannotator": "RUL prediction",
    }

    #: Parameter names that contain a task word but say nothing about the task.
    #: ``eol_soh`` is RUL's end-of-life *threshold* — the SOH value at which a
    #: cell is declared dead — and every RUL baseline config carries it.
    _TASK_DECOYS = ("eol_soh",)

    def _infer_task(self, hay: str, data: dict[str, Any]) -> str | None:
        """The canonical task this config trains, from structure not substrings.

        Substring matching over the whole YAML used to decide this, and it was
        wrong for 42 of the 44 configs it called SOH: every RUL baseline sets
        ``label: {name: RULLabelAnnotator, eol_soh: 0.9}``, and ``eol_soh``
        contains ``soh``. Lifetime prediction was being advertised as state-of-
        health estimation across the whole catalog.

        The hierarchy is now:

        1. an explicit ``task:`` field, when a config declares one;
        2. the label annotator's registered class name;
        3. the ``configs/<task>/`` directory, documented as a fallback for a
           config that names neither.

        A parameter name is never a signal.
        """
        if isinstance(data, dict):
            declared = str(data.get("task", "") or "").strip()
            if declared:
                canonical = self._canonical_task(declared)
                if canonical:
                    return canonical

            label = data.get("label", {})
            if isinstance(label, dict):
                annotator = str(label.get("name", "") or "").strip().lower()
                if annotator in self._TASK_BY_ANNOTATOR:
                    return self._TASK_BY_ANNOTATOR[annotator]
                # An unregistered annotator whose own name states the task.
                if annotator.startswith("soh") or "stateofhealth" in annotator:
                    return "SOH estimation"
                if annotator.startswith("rul"):
                    return "RUL prediction"

        # Fallback: the directory the repository files the config under. The
        # decoy parameters are stripped first so they cannot leak back in.
        path_hay = hay
        for decoy in self._TASK_DECOYS:
            path_hay = path_hay.replace(decoy, "")
        if "configs/soh/" in path_hay or "configs\\soh\\" in path_hay:
            return "SOH estimation"
        if "configs/rul/" in path_hay or "configs\\rul\\" in path_hay:
            return "RUL prediction"
        return None

    @staticmethod
    def _canonical_task(declared: str) -> str | None:
        lowered = declared.strip().lower()
        if lowered.startswith("soh") or "state of health" in lowered:
            return "SOH estimation"
        if lowered.startswith("rul") or "remaining useful life" in lowered:
            return "RUL prediction"
        return None

    def _infer_model(self, path: Path, data: dict[str, Any]) -> tuple[str | None, str | None]:
        """Resolve (model_key, model_display) for a config file.

        Resolution order, most authoritative first:

        1. the config's parent directory name — configs/ is organized one
           directory per model, so this is the declared identity;
        2. the ``model.name`` class declared inside the YAML.

        Neither step scans the raw file text. Substring-scanning the whole
        file matched commented-out lines: e.g. the disabled
        ``# name: 'RandomTrainTestSplitter'`` line in the dummy cruh/crush/snl
        configs matched the "random" -> "rf" alias and reported three Dummy
        baselines as Random Forest.
        """
        token = normalize_model_token(path.parent.name)
        display = _NORMALIZED_DISPLAY_NAMES.get(token)
        if display is not None:
            return _NORMALIZED_KEY_ALIASES.get(token, token.replace(" ", "_")), display

        model_name = ""
        if isinstance(data, dict):
            model_section = data.get("model")
            if isinstance(model_section, dict):
                model_name = str(model_section.get("name", "")).strip().lower()
        key = MODEL_CLASS_KEYS.get(model_name)
        if key is not None:
            token = normalize_model_token(key)
            return (
                _NORMALIZED_KEY_ALIASES.get(token, token.replace(" ", "_")),
                _NORMALIZED_DISPLAY_NAMES[token],
            )
        return None, None

    def _infer_split_preset(self, path: Path) -> str | None:
        name = path.name.lower()
        if name.startswith("matr_1"):
            return "official_matr_split_1"
        if name.startswith("matr_2"):
            return "official_matr_split_2"
        return path.stem

    def _split_label(self, preset: str) -> str:
        return {
            "official_matr_split_1": "Official BatteryML split 1",
            "official_matr_split_2": "Official BatteryML split 2",
        }.get(preset, preset.replace("_", " ").title())

    def get_entries(self, dataset: str | None = None, task: str | None = None) -> list[ConfigEntry]:
        entries = self.entries
        if dataset:
            entries = [e for e in entries if e.dataset == dataset]
        if task:
            entries = [e for e in entries if e.task == task]
        return sorted(entries, key=lambda e: (e.model_display, e.split_label, e.config_path.as_posix()))

    def get_models_for_dataset_task(self, dataset: str, task: str) -> list[dict[str, str]]:
        seen = {}
        for e in self.get_entries(dataset, task):
            seen.setdefault(e.model_key, e.model_display)
        return [{"model_key": k, "model_display": v} for k, v in seen.items()]

    def get_configs_for_selection(self, dataset: str, task: str, model_key: str) -> list[ConfigEntry]:
        return [e for e in self.get_entries(dataset, task) if e.model_key == model_key]

    def get_split_preview(self, config_path: Path) -> dict[str, Any]:
        preview = {"config_path": str(config_path), "status": "not_parsed"}
        try:
            text = config_path.read_text(encoding="utf-8", errors="ignore")
            data = yaml.safe_load(text) or {}
            preview["split_source"] = "BatteryML official MATR splitter" if "matr" in str(config_path).lower() else "unknown"
            preview["split_config"] = config_path.stem
            preview["status"] = "parsed"
            task = data.get("train_test_split", {}) if isinstance(data, dict) else {}
            preview["train_test_split"] = task
        except Exception as exc:
            preview["error"] = str(exc)
        out = self.repo_root / "data" / "metadata" / "split_preview_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
        return preview


@lru_cache(maxsize=1)
def get_config_registry(repo_root: str | None = None) -> ConfigRegistry:
    return ConfigRegistry(Path(repo_root) if repo_root else None).build()
