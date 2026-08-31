from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


MODEL_DISPLAY_NAMES = {
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
    "dummy": "Dummy Baseline",
    "discharge model": "Discharge Feature Model",
    "full model": "Full Feature Model",
    "variance model": "Variance Feature Model",
}

MODEL_KEY_ALIASES = {
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
        hay = f"{path} {path.name} {text}".lower()
        dataset = self._infer_dataset(hay, data)
        task = self._infer_task(hay, data)
        model_key, model_display = self._infer_model(path, data)
        split_preset = self._infer_split_preset(path)
        split_label = self._split_label(split_preset)
        if not (dataset and task and model_key and split_preset):
            return None
        return ConfigEntry(dataset, task, model_key, model_display, split_preset, split_label, path, {"yaml": data})

    def _infer_dataset(self, hay: str, data: dict[str, Any]) -> str | None:
        for ds in ("MATR", "HUST", "RWTH", "CALCE", "SNL", "UL_PUR"):
            if ds.lower() in hay:
                return ds
        split = data.get("train_test_split", {}) if isinstance(data, dict) else {}
        cdp = split.get("cell_data_path", "") if isinstance(split, dict) else ""
        for ds in ("MATR", "HUST", "RWTH", "CALCE", "SNL", "UL_PUR"):
            if ds.lower() in str(cdp).lower():
                return ds
        return None

    def _infer_task(self, hay: str, data: dict[str, Any]) -> str | None:
        label = data.get("label", {}) if isinstance(data, dict) else {}
        label_name = str(label.get("name", "")).lower() if isinstance(label, dict) else ""
        if any(x in hay or x in label_name for x in ("soh", "state of health")):
            return "SOH estimation"
        if any(x in hay or x in label_name for x in ("rul", "remaining useful life")):
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
