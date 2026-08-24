"""BatteryML Configuration Catalog Builder.

Dynamically scans configs/ directory and builds a catalog of available
models, datasets, tasks with their metadata and readiness status.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum


class TaskType(Enum):
    RUL = "RUL"
    SOH = "SOH"
    SOC = "SOC"
    UNKNOWN = "unknown"


class Status(Enum):
    READY = "ready"                    # Config + processed data exists
    CONFIG_ONLY = "config_only"        # Config exists but no data at all
    NEED_PREPROCESS = "need_preprocess"  # Raw exists but processed missing
    MISSING = "missing"                # Config exists but raw missing
    EXPERIMENTAL = "experimental"      # Implementation incomplete


@dataclass
class ModelConfig:
    """Represents a single model configuration."""
    config_path: Path
    model_name: str
    model_class: str
    dataset: str
    task: TaskType
    feature_extractor: str
    train_test_split: str
    device_support: List[str] = field(default_factory=lambda: ["cpu"])
    is_experimental: bool = False
    is_debug: bool = False
    status: Status = Status.MISSING
    raw_exists: bool = False
    processed_exists: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "model_name": self.model_name,
            "model_class": self.model_class,
            "dataset": self.dataset,
            "task": self.task.value,
            "feature_extractor": self.feature_extractor,
            "train_test_split": self.train_test_split,
            "device_support": self.device_support,
            "is_experimental": self.is_experimental,
            "is_debug": self.is_debug,
            "status": self.status.value,
            "raw_exists": self.raw_exists,
            "processed_exists": self.processed_exists,
            "notes": self.notes,
        }


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    raw_path: Path
    processed_path: Path
    raw_exists: bool = False
    processed_exists: bool = False
    configs: List[ModelConfig] = field(default_factory=list)
    custom: bool = False  # True if user-uploaded custom dataset

    @property
    def status(self) -> Status:
        if self.configs:
            # Aggregate status from configs
            statuses = [c.status for c in self.configs]
            if Status.READY in statuses:
                return Status.READY
            elif Status.NEED_PREPROCESS in statuses:
                return Status.NEED_PREPROCESS
            elif Status.CONFIG_ONLY in statuses:
                return Status.CONFIG_ONLY
        return Status.MISSING


class ConfigCatalog:
    """Catalog of all available BatteryML configurations."""

    # Known dataset names from BatteryML
    KNOWN_DATASETS = [
        "MATR", "HUST", "CALCE", "RWTH", "SNL", "HNEI", "UL_PUR", "CLO",
        "CRUH", "CRUSH", "MIX", "OX", "ARBIN", "NEWARE"
    ]

    # Model class to task mapping
    MODEL_TASK_PATTERNS = {
        # RUL models
        "RUL": ["RUL", "rul"],
        # SOH models
        "SOH": ["SOH", "soh", "StateOfHealth"],
        # SOC models
        "SOC": ["SOC", "soc", "StateOfCharge"],
    }

    # NN model patterns that support CUDA
    NN_MODEL_PATTERNS = ["LSTM", "GRU", "CNN", "MLP", "Transformer", "RNN"]
    # Sklearn/XGB patterns that are CPU-first
    SKLEARN_PATTERNS = ["XGB", "RandomForest", "Ridge", "Linear", "PLSR", "PCR",
                        "Dummy", "GPR", "GaussianProcess", "SVM", "ElasticNet"]

    # Debug/Smoke test models
    DEBUG_MODELS = ["DummyRULPredictor", "DummySOHPredictor"]

    # Experimental models
    EXPERIMENTAL_MODELS = ["BatLiNet"]

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(".")
        self.configs_dir = self.root_dir / "configs"
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.datasets: Dict[str, DatasetInfo] = {}
        self.configs: List[ModelConfig] = []
        self._catalog_built = False

    def build_catalog(self, include_debug: bool = False) -> "ConfigCatalog":
        """Scan all configs and build the catalog."""
        if not self.configs_dir.exists():
            print(f"Configs directory not found: {self.configs_dir}")
            return self

        # First pass: scan all yaml files
        for config_file in self.configs_dir.rglob("*.yaml"):
            try:
                config = self._parse_config(config_file)
                if config is None:
                    continue

                # Skip debug models unless explicitly included
                if config.is_debug and not include_debug:
                    continue

                self.configs.append(config)

                # Add to dataset
                if config.dataset not in self.datasets:
                    self.datasets[config.dataset] = self._create_dataset_info(config.dataset)
                self.datasets[config.dataset].configs.append(config)

            except Exception as e:
                print(f"Error parsing {config_file}: {e}")

        # Second pass: check data existence for all datasets
        self._check_all_data_status()

        self._catalog_built = True
        return self

    def _parse_config(self, config_path: Path) -> Optional[ModelConfig]:
        """Parse a single config file and extract metadata."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
        except Exception:
            return None

        if not content or not isinstance(content, dict):
            return None

        # Extract model info
        model_info = content.get("model", {})
        model_class = model_info.get("name", "Unknown")

        # Extract dataset from cell_data_path or path
        dataset = self._infer_dataset(config_path, content)
        if not dataset:
            return None  # Skip configs we can't identify

        # Extract task from label or model name
        task = self._infer_task(content, model_class)

        # Extract feature extractor
        feature_info = content.get("feature", {})
        feature_extractor = feature_info.get("name", "Unknown")

        # Extract train test split
        split_info = content.get("train_test_split", {})
        train_test_split = split_info.get("name", "Unknown")

        # Determine device support
        device_support = self._infer_device_support(model_class)

        # Check if debug model
        is_debug = any(dm in model_class for dm in self.DEBUG_MODELS)

        # Check if experimental
        is_experimental = any(em in model_class for em in self.EXPERIMENTAL_MODELS)

        # Generate model display name
        model_name = self._generate_model_name(model_class, dataset, config_path)

        return ModelConfig(
            config_path=config_path,
            model_name=model_name,
            model_class=model_class,
            dataset=dataset,
            task=task,
            feature_extractor=feature_extractor,
            train_test_split=train_test_split,
            device_support=device_support,
            is_experimental=is_experimental,
            is_debug=is_debug,
        )

    def _infer_dataset(self, config_path: Path, content: Dict) -> Optional[str]:
        """Infer dataset from config path or content."""
        # Try cell_data_path first
        split_info = content.get("train_test_split", {})
        cell_data_path = split_info.get("cell_data_path", "")

        if cell_data_path:
            # Extract from path like 'data/processed/MATR'
            parts = Path(cell_data_path).parts
            for part in parts:
                upper_part = part.upper()
                for ds in self.KNOWN_DATASETS:
                    if ds.upper() == upper_part:
                        return ds

        # Try to infer from file path
        path_str = str(config_path).upper()
        for ds in self.KNOWN_DATASETS:
            if ds.upper() in path_str:
                return ds

        # Try filename
        filename = config_path.stem.upper()
        for ds in self.KNOWN_DATASETS:
            if ds.upper() in filename:
                return ds

        return None

    def _infer_task(self, content: Dict, model_class: str) -> TaskType:
        """Infer task type from label name or model class."""
        label_info = content.get("label", {})
        label_name = label_info.get("name", "")

        # Check label name
        if label_name:
            if "RUL" in label_name:
                return TaskType.RUL
            if "SOH" in label_name or "StateOfHealth" in label_name:
                return TaskType.SOH
            if "SOC" in label_name or "StateOfCharge" in label_name:
                return TaskType.SOC

        # Check model class
        model_upper = model_class.upper()
        if "RUL" in model_upper:
            return TaskType.RUL
        if "SOH" in model_upper:
            return TaskType.SOH
        if "SOC" in model_upper:
            return TaskType.SOC

        return TaskType.UNKNOWN

    def _infer_device_support(self, model_class: str) -> List[str]:
        """Infer supported devices from model class."""
        model_upper = model_class.upper()

        # Check if NN model (supports both CPU and CUDA)
        for pattern in self.NN_MODEL_PATTERNS:
            if pattern.upper() in model_upper:
                return ["cpu", "cuda"]

        # Sklearn-based models - CPU first, CUDA experimental for XGB
        for pattern in self.SKLEARN_PATTERNS:
            if pattern.upper() in model_upper:
                if "XGB" in model_upper:
                    return ["cpu", "cuda"]  # XGBoost has GPU support
                return ["cpu"]  # Others CPU only

        # Default to CPU only
        return ["cpu"]

    def _generate_model_name(self, model_class: str, dataset: str, config_path: Path) -> str:
        """Generate a display name for the model."""
        # Get method from path (e.g., xgb, lstm, ridge)
        method = config_path.parent.name
        return f"{model_class} ({method})"

    def _create_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Create a DatasetInfo for a known dataset."""
        raw_path = self.raw_dir / dataset_name
        processed_path = self.processed_dir / dataset_name

        return DatasetInfo(
            name=dataset_name,
            raw_path=raw_path,
            processed_path=processed_path,
            raw_exists=raw_path.exists() and any(raw_path.iterdir()),
            processed_exists=processed_path.exists() and any(processed_path.iterdir()),
        )

    def _check_all_data_status(self):
        """Update status for all configs based on data existence."""
        for config in self.configs:
            if config.dataset in self.datasets:
                ds_info = self.datasets[config.dataset]
                config.raw_exists = ds_info.raw_exists
                config.processed_exists = ds_info.processed_exists

                # Determine status
                if config.is_experimental:
                    config.status = Status.EXPERIMENTAL
                elif config.processed_exists:
                    config.status = Status.READY
                elif config.raw_exists:
                    config.status = Status.NEED_PREPROCESS
                else:
                    # Check if any config for this dataset has data
                    any_processed = any(c.processed_exists for c in ds_info.configs)
                    any_raw = any(c.raw_exists for c in ds_info.configs)

                    if any_processed:
                        config.status = Status.READY  # Data exists for other split
                    elif any_raw:
                        config.status = Status.NEED_PREPROCESS
                    else:
                        config.status = Status.CONFIG_ONLY

    def get_datasets(self) -> List[str]:
        """Get all dataset names."""
        return sorted(self.datasets.keys())

    def get_tasks_for_dataset(self, dataset: str) -> List[TaskType]:
        """Get available tasks for a dataset."""
        if dataset not in self.datasets:
            return []

        tasks = set()
        for config in self.datasets[dataset].configs:
            if not config.is_debug:  # Exclude debug models from task list
                tasks.add(config.task)
        return sorted(tasks, key=lambda t: t.value)

    def get_models_for_dataset_task(self, dataset: str, task: TaskType,
                                     include_debug: bool = False) -> List[ModelConfig]:
        """Get models available for a dataset+task combination."""
        if dataset not in self.datasets:
            return []

        models = []
        for config in self.datasets[dataset].configs:
            if config.task == task:
                if config.is_debug and not include_debug:
                    continue
                models.append(config)

        return sorted(models, key=lambda m: (m.is_experimental, m.model_name))

    def get_configs_for_model(self, dataset: str, task: TaskType,
                              model_class: str) -> List[ModelConfig]:
        """Get all config variants for a specific model class."""
        if dataset not in self.datasets:
            return []

        return [
            c for c in self.datasets[dataset].configs
            if c.task == task and c.model_class == model_class
        ]

    def get_debug_models(self) -> List[ModelConfig]:
        """Get all debug/smoke test models."""
        return [c for c in self.configs if c.is_debug]

    def get_experimental_models(self) -> List[ModelConfig]:
        """Get all experimental models."""
        return [c for c in self.configs if c.is_experimental]

    def check_compatibility(self, dataset: str, task: TaskType,
                           model_class: str) -> tuple[bool, str]:
        """Check if a model is compatible with a task for a dataset."""
        if dataset not in self.datasets:
            return False, f"Dataset {dataset} not found in catalog"

        models = self.get_models_for_dataset_task(dataset, task)
        model_classes = [m.model_class for m in models]

        if model_class not in model_classes:
            # Check if model exists but for different task
            all_dataset_models = [
                c for c in self.datasets[dataset].configs
                if c.model_class == model_class
            ]
            if all_dataset_models:
                available_tasks = list(set(c.task.value for c in all_dataset_models))
                return False, f"Model {model_class} is designed for tasks: {available_tasks}, not {task.value}"
            return False, f"Model {model_class} not available for dataset {dataset}"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """Export catalog to dictionary."""
        return {
            "datasets": {
                name: {
                    "name": info.name,
                    "raw_path": str(info.raw_path),
                    "processed_path": str(info.processed_path),
                    "raw_exists": info.raw_exists,
                    "processed_exists": info.processed_exists,
                    "status": info.status.value,
                    "config_count": len(info.configs),
                }
                for name, info in self.datasets.items()
            },
            "configs": [c.to_dict() for c in self.configs],
            "stats": {
                "total_datasets": len(self.datasets),
                "total_configs": len(self.configs),
                "ready_configs": len([c for c in self.configs if c.status == Status.READY]),
                "experimental_configs": len([c for c in self.configs if c.is_experimental]),
                "debug_configs": len([c for c in self.configs if c.is_debug]),
            }
        }


# Global catalog instance
catalog_instance: Optional[ConfigCatalog] = None


def get_catalog(root_dir: Path = None, force_rebuild: bool = False,
                include_debug: bool = False) -> ConfigCatalog:
    """Get or create the global catalog instance."""
    global catalog_instance
    if catalog_instance is None or force_rebuild:
        catalog_instance = ConfigCatalog(root_dir).build_catalog(include_debug=include_debug)
    return catalog_instance


def refresh_catalog(root_dir: Path = None, include_debug: bool = False) -> ConfigCatalog:
    """Force refresh the catalog."""
    return get_catalog(root_dir, force_rebuild=True, include_debug=include_debug)
