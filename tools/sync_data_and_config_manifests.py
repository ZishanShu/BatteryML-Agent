from pathlib import Path
import json
import os
import re
import shutil
import subprocess

# This checkout, not a fixed machine. The path here used to be the absolute
# location of one developer's server, so on anyone else's clone the two
# manifests below were written outside the repository — and the web UI, which
# reads data/metadata/dataset_manifest.json and nothing else to learn what
# datasets exist, showed "No datasets are available yet" no matter how much
# data had been preprocessed.
#
# parents[1] because this file is tools/<name>.py, so its parent is tools/
# and its grandparent is the repository root. BATTERYML_ROOT overrides it for
# the case this tool was presumably written for: metadata for a checkout other
# than the one holding the script.
ROOT = Path(os.environ.get("BATTERYML_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW = ROOT / "data" / "raw"
V12 = RAW / "BatteryLife" / "v12"
META = ROOT / "data" / "metadata"

V12.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)


def file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def size_human(path: Path) -> str:
    if not path.exists():
        return "-"
    try:
        return subprocess.getoutput(f"du -sh '{path}' 2>/dev/null | awk '{{print $1}}'") or "-"
    except Exception:
        return "-"


def safe_link(src: Path, dst: Path):
    src = src.resolve()

    if not src.exists() or file_count(src) == 0:
        print(f"[SKIP] source missing or empty: {src}")
        return False

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            current = dst.resolve()
            if current == src:
                print(f"[OK] link exists: {dst} -> {src}")
                return True
            dst.unlink()
        elif dst.is_dir() and file_count(dst) == 0:
            shutil.rmtree(dst)
        else:
            print(f"[SKIP] target already non-empty: {dst}")
            return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)
    print(f"[LINK] {dst} -> {src}")
    return True


def find_best_dir(root: Path, keywords):
    if not root.exists():
        return None

    hits = []
    kws = [k.lower() for k in keywords]

    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        n = file_count(p)
        if n == 0:
            continue

        s = str(p).lower()
        if any(k in s for k in kws):
            hits.append((n, len(str(p)), p))

    if not hits:
        return None

    hits.sort(reverse=True)
    return hits[0][2]


def normalize_batterylife_layout():
    print("\n=== Normalize BatteryLife v12 layout ===")

    battery_raw = RAW / "BatteryLife_Raw"

    mapping = {
        "Zn-ion": [
            RAW / "BatteryLife_Raw" / "Zn-ion",
            RAW / "BatteryLife_Raw" / "Zn_ion",
            find_best_dir(battery_raw, ["zn", "zinc"]),
        ],
        "Na-ion": [
            RAW / "BatteryLife_Raw" / "Na-ion",
            RAW / "BatteryLife_Raw" / "Na_ion",
            find_best_dir(battery_raw, ["na", "sodium"]),
        ],
        "CALB": [
            RAW / "BatteryLife_Raw" / "CALB",
            find_best_dir(battery_raw, ["calb"]),
        ],
        "Farasis_processed": [
            RAW / "BatteryLife" / "v12" / "Farasis_processed",
        ],
        "Farasis_raw": [
            RAW / "BatteryLife" / "v12" / "Farasis_raw",
        ],
        "XJTU_complete": [
            RAW / "XJTU",
        ],
        "Tongji": [
            RAW / "Tongji",
        ],
        "SDU": [
            RAW / "SDU",
        ],
        "ISU_ILCC": [
            RAW / "ISU_ILCC",
        ],
    }

    for name, candidates in mapping.items():
        dst = V12 / name

        linked = False
        for src in candidates:
            if src is None:
                continue
            if src.exists() and file_count(src) > 0:
                safe_link(src, dst)
                linked = True
                break

        if not linked:
            print(f"[MISS] {name}: no non-empty source found")


def infer_dataset_from_config(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"_[0-9]+$", "", name)

    known = {
        "matr": "MATR",
        "hust": "HUST",
        "snl": "SNL",
        "clo": "CLO",
        "cruh": "CRUH",
        "crush": "CRUSH",
        "mix": "MIX",
        "calce": "CALCE",
        "rwth": "RWTH",
    }

    for k, v in known.items():
        if name == k or k in str(path).lower():
            return v

    return name.upper()


def infer_model_from_config(path: Path) -> str:
    s = str(path).lower()

    rules = [
        ("sklearn/xgb", "XGBoost"),
        ("sklearn/rf", "Random Forest"),
        ("nn_models/mlp", "MLP"),
        ("nn_models/lstm", "LSTM"),
        ("nn_models/cnn", "CNN"),
        ("nn_models/transformer", "Transformer"),
        ("sklearn/ridge", "Ridge"),
        ("sklearn/gpr", "GPR"),
        ("sklearn/plsr", "PLSR"),
        ("sklearn/pcr", "PCR"),
        ("sklearn/dummy", "Dummy"),
        ("sklearn/discharge_model", "Discharge model"),
        ("sklearn/full_model", "Full model"),
        ("sklearn/variance_model", "Variance model"),
    ]

    for key, model in rules:
        if key in s:
            return model

    return path.parent.name


def infer_task_from_config(path: Path) -> str:
    s = str(path).lower()
    if "/configs/soh/" in s or "configs/soh/" in s:
        return "SOH estimation"
    return "RUL prediction"


def infer_device_recommendation(path: Path) -> str:
    s = str(path).lower()
    if "nn_models" in s:
        return "cuda_or_cpu"
    return "cpu"


def build_model_manifest():
    print("\n=== Build model config manifest ===")

    configs_root = ROOT / "configs"
    items = []

    for path in sorted(configs_root.rglob("*.yaml")):
        rel = path.relative_to(ROOT)
        item = {
            "dataset": infer_dataset_from_config(path),
            "task": infer_task_from_config(path),
            "model": infer_model_from_config(path),
            "config_path": str(rel),
            "exists": path.exists(),
            "device_recommendation": infer_device_recommendation(path),
        }
        items.append(item)

    out = META / "model_config_manifest.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WRITE] {out} ({len(items)} configs)")


def add_dataset_entry(entries, name, source, path: Path, family=None, version=None):
    entries.append({
        "name": name,
        "source": source,
        "battery_family": family,
        "version": version,
        "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
        "exists": path.exists(),
        "file_count": file_count(path),
        "size": size_human(path),
        "status": "present" if path.exists() and file_count(path) > 0 else "missing_or_empty",
    })


def build_dataset_manifest():
    print("\n=== Build dataset manifest ===")

    entries = []

    batteryml_datasets = [
        "CALCE", "MATR", "HUST", "RWTH", "OX", "SNL", "HNEI", "UL_PUR",
        "MICH", "MICH_EXP", "ISU_ILCC", "XJTU", "Tongji", "Stanford", "SDU"
    ]

    for name in batteryml_datasets:
        add_dataset_entry(
            entries,
            name=name,
            source="BatteryML/raw",
            path=RAW / name,
            family="Li-ion" if name not in ["OX"] else None,
            version=None,
        )

    batterylife_v12_family = {
        "Zn-ion": "Zn-ion",
        "Na-ion": "Na-ion",
        "CALB": "Li-ion",
        "Farasis_processed": "Li-ion",
        "Farasis_raw": "Li-ion",
        "XJTU_complete": "Li-ion",
        "Tongji": "Li-ion",
        "SDU": "Li-ion",
        "ISU_ILCC": "Li-ion",
    }

    for p in sorted(V12.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir() or p.is_symlink():
            add_dataset_entry(
                entries,
                name=p.name,
                source="BatteryLife/v12",
                path=p,
                family=batterylife_v12_family.get(p.name),
                version="v12",
            )

    out = META / "dataset_manifest.json"
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WRITE] {out} ({len(entries)} datasets)")


def print_summary():
    print("\n=== Dataset summary ===")
    manifest = json.loads((META / "dataset_manifest.json").read_text())
    for x in manifest:
        if x["status"] == "present":
            print(f"{x['source']:18s} {x['name']:20s} {x['size']:>8s} files={x['file_count']:>5} path={x['path']}")

    print("\n=== Model config summary ===")
    models = json.loads((META / "model_config_manifest.json").read_text())
    for x in models:
        if x["dataset"] == "MATR" and x["task"] == "RUL prediction":
            print(f"{x['dataset']:8s} {x['task']:16s} {x['model']:18s} {x['config_path']}")


if __name__ == "__main__":
    normalize_batterylife_layout()
    build_dataset_manifest()
    build_model_manifest()
    print_summary()
