"""Core utilities for BatteryML Web."""

from .gpu_checker import (
    check_nvidia_smi,
    check_torch_cuda,
    check_xgboost_gpu,
    get_gpu_readiness_report,
    validate_device_choice,
    list_available_configs,
    get_gpu_demo_configs,
)
from .readiness import (
    DatasetReadiness,
    DatasetStatus,
    ModelReadiness,
    ModelStatus,
    assess_dataset,
    assess_external_model,
    assess_model,
    assess_registry_model,
    get_workflow_context,
    list_dataset_readiness,
    list_model_readiness,
)

__all__ = [
    "check_nvidia_smi",
    "check_torch_cuda",
    "check_xgboost_gpu",
    "get_gpu_readiness_report",
    "validate_device_choice",
    "list_available_configs",
    "get_gpu_demo_configs",
    "DatasetReadiness",
    "DatasetStatus",
    "ModelReadiness",
    "ModelStatus",
    "assess_dataset",
    "assess_external_model",
    "assess_model",
    "assess_registry_model",
    "get_workflow_context",
    "list_dataset_readiness",
    "list_model_readiness",
]
