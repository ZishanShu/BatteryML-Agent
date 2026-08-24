from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def check_nvidia_smi() -> Dict[str, Any]:
    """Check if nvidia-smi is available and get GPU info."""
    result = {
        "available": False,
        "command": "nvidia-smi",
        "output": None,
        "error": None,
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            result["available"] = True
            result["output"] = proc.stdout
        else:
            result["error"] = proc.stderr
    except FileNotFoundError:
        result["error"] = "nvidia-smi command not found"
    except subprocess.TimeoutExpired:
        result["error"] = "nvidia-smi timed out"
    except Exception as e:
        result["error"] = str(e)
    return result


def check_torch_cuda() -> Dict[str, Any]:
    """Check PyTorch CUDA availability using Python subprocess."""
    result = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "error": None,
    }

    script = '''
import sys
try:
    import torch
    print("TORCH_VERSION:", torch.__version__)
    print("CUDA_AVAILABLE:", torch.cuda.is_available())
    print("DEVICE_COUNT:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"DEVICE_{i}:", torch.cuda.get_device_name(i))
    print("SUCCESS: True")
except ImportError as e:
    print("ERROR: torch not installed -", str(e))
    print("SUCCESS: False")
except Exception as e:
    print("ERROR:", str(e))
    print("SUCCESS: False")
'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = proc.stdout + proc.stderr

        # Parse output
        for line in output.split("\n"):
            if line.startswith("TORCH_VERSION:"):
                result["torch_version"] = line.split(":", 1)[1].strip()
                result["torch_installed"] = True
            elif line.startswith("CUDA_AVAILABLE:"):
                result["cuda_available"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("DEVICE_COUNT:"):
                try:
                    result["device_count"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("DEVICE_"):
                try:
                    _, device_str = line.split(":", 1)
                    result["device_names"].append(device_str.strip())
                except ValueError:
                    pass
            elif line.startswith("ERROR:"):
                result["error"] = line.split(":", 1)[1].strip() if ":" in line else line

        # Determine success from the last line
        if "SUCCESS: True" not in output:
            if result["error"] is None and not result["torch_installed"]:
                result["error"] = "PyTorch not installed or import failed"

    except subprocess.TimeoutExpired:
        result["error"] = "PyTorch CUDA check timed out"
    except Exception as e:
        result["error"] = f"Failed to run torch check: {str(e)}"

    return result


def check_xgboost_gpu() -> Dict[str, Any]:
    """Check if XGBoost GPU support is available."""
    result = {
        "xgboost_installed": False,
        "gpu_support": False,
        "test_passed": False,
        "error": None,
    }

    script = '''
import sys
try:
    import xgboost as xgb
    print("XGBOOST_VERSION:", xgb.__version__)
    print("XGBOOST_INSTALLED: True")
    # Try a minimal GPU test
    try:
        import numpy as np
        # Minimal test: check if tree_method 'gpu_hist' is available
        params = {'tree_method': 'gpu_hist', 'max_depth': 2}
        # Try to create a DMatrix and run a minimal training
        dtrain = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=np.array([0, 1]))
        bst = xgb.train(params, dtrain, num_boost_round=1)
        print("GPU_TEST_PASSED: True")
    except Exception as e:
        print("GPU_TEST_PASSED: False")
        print("GPU_TEST_ERROR:", str(e))
    print("SUCCESS: True")
except ImportError as e:
    print("XGBOOST_INSTALLED: False")
    print("ERROR: xgboost not installed -", str(e))
    print("SUCCESS: False")
'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = proc.stdout + proc.stderr

        for line in output.split("\n"):
            if line.startswith("XGBOOST_INSTALLED:"):
                result["xgboost_installed"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("XGBOOST_VERSION:"):
                result["xgboost_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("GPU_TEST_PASSED:"):
                result["test_passed"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("GPU_TEST_ERROR:"):
                result["error"] = line.split(":", 1)[1].strip()

        if result["xgboost_installed"] and result["test_passed"]:
            result["gpu_support"] = True

    except subprocess.TimeoutExpired:
        result["error"] = "XGBoost GPU check timed out"
    except Exception as e:
        result["error"] = f"Failed to run xgboost check: {str(e)}"

    return result


def get_gpu_readiness_report() -> Dict[str, Any]:
    """Get a comprehensive GPU readiness report."""
    report = {
        "nvidia_smi": check_nvidia_smi(),
        "torch_cuda": check_torch_cuda(),
        "xgboost_gpu": check_xgboost_gpu(),
        "overall_ready": False,
        "recommendations": [],
    }

    # Determine overall readiness
    torch_ok = report["torch_cuda"]["cuda_available"]
    nvidia_ok = report["nvidia_smi"]["available"]

    if torch_ok and nvidia_ok:
        report["overall_ready"] = True

    # Generate recommendations
    if not nvidia_ok:
        report["recommendations"].append(
            "nvidia-smi not found. Check if NVIDIA drivers are installed and nvidia-smi is in PATH."
        )

    if not report["torch_cuda"]["torch_installed"]:
        report["recommendations"].append(
            "PyTorch not installed. Install with: pip install torch"
        )
    elif not torch_ok:
        report["recommendations"].append(
            "PyTorch CUDA not available. Check: 1) CUDA drivers installed, "
            "2) PyTorch installed with CUDA support (pip install torch --index-url https://download.pytorch.org/whl/cu118), "
            "3) Container has GPU mapping (docker run --gpus all ...)"
        )

    return report


def validate_device_choice(device: str) -> Dict[str, Any]:
    """Validate if the chosen device can be used."""
    report = get_gpu_readiness_report()

    result = {
        "requested_device": device,
        "can_run": False,
        "will_use": device,
        "error": None,
        "gpu_report": report,
    }

    if device == "cpu":
        result["can_run"] = True
        result["will_use"] = "cpu"
    elif device in ["cuda", "gpu"]:
        if report["overall_ready"]:
            result["can_run"] = True
            result["will_use"] = "cuda"
        else:
            result["can_run"] = False
            result["error"] = "GPU requested but CUDA is not available"
    elif device == "auto":
        if report["overall_ready"]:
            result["can_run"] = True
            result["will_use"] = "cuda"
        else:
            result["can_run"] = True
            result["will_use"] = "cpu"
    else:
        result["error"] = f"Unknown device: {device}"

    return result


def list_available_configs() -> List[Dict[str, Any]]:
    """List available config files and identify GPU-native configs."""
    configs = []
    config_dirs = [
        Path("configs/baselines"),
        Path("configs"),
    ]

    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        for config_file in config_dir.rglob("*.yaml"):
            try:
                import yaml
                content = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                model_name = ""
                if isinstance(content.get("model"), dict):
                    model_name = content["model"].get("name", "")
                elif isinstance(content.get("model"), str):
                    model_name = content["model"]

                # Identify GPU-native configs (deep learning models)
                gpu_native_models = ["cnn", "lstm", "transformer", "mlp"]
                is_gpu_native = any(m in model_name.lower() for m in gpu_native_models)

                # Infer dataset from path or train_test_split
                dataset = content.get("dataset", "unknown")
                if dataset == "unknown":
                    # Try to get from train_test_split cell_data_path
                    train_test_split = content.get("train_test_split", {})
                    if isinstance(train_test_split, dict):
                        cell_data_path = train_test_split.get("cell_data_path", "")
                        if cell_data_path:
                            # Extract dataset from path like "data/processed/MATR"
                            parts = cell_data_path.split("/")
                            if len(parts) >= 3:
                                dataset = parts[-1].upper()
                # Also infer from file path
                if dataset == "unknown":
                    path_lower = str(config_file).lower()
                    for ds in ["matr", "hust", "calce", "snl", "rwth", "crush", "cruh", "ox", "hnei", "ul_pur", "arbin", "neware", "clobattery"]:
                        if ds in path_lower:
                            dataset = ds.upper()
                            break

                configs.append({
                    "path": str(config_file),
                    "name": config_file.name,
                    "model": model_name,
                    "is_gpu_native": is_gpu_native,
                    "dataset": dataset,
                })
            except Exception:
                pass

    return configs


def get_gpu_demo_configs() -> List[Dict[str, Any]]:
    """Get configs suitable for GPU demo (MATR dataset, GPU-native models)."""
    all_configs = list_available_configs()
    gpu_configs = [
        c for c in all_configs
        if c.get("is_gpu_native") and "matr" in c.get("dataset", "").lower()
    ]
    return gpu_configs
