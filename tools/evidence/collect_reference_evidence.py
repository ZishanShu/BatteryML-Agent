#!/usr/bin/env python3
"""Collect existing BatteryML evidence without running BatteryML or an LLM.

Python 3.10+, standard library only. Current Git/environment snapshots are not
historical execution evidence. Pickles and model files are copied as bytes only.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import selectors
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

SCHEMA = "batteryml-existing-evidence/1"
SENSITIVE_NAME = re.compile(r"(^\.env($|\.)|credential|secret|password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|cookies?|\.pem$|\.key$|\.pypirc$|\.netrc$)", re.I)
SECRET_VALUE = re.compile(r"(?i)[\"']?\b(?:api[_-]?key|authorization|proxy-authorization|password|passwd|access[_-]?token|refresh[_-]?token|client[_-]?secret|set-cookie|cookie|token|secret|credentials?)[\"']?\s*[:=]\s*[\"']?([^\s\"',;}]+)")
SENSITIVE_JSON_KEY = re.compile(r"^(?:apikey|authorization|proxyauthorization|password|passwd|accesstoken|refreshtoken|idtoken|clientsecret|secret|token|credential|credentials|cookie|setcookie)$", re.I)
URL = re.compile(r"(?:https?|ftp)://[^\s<>\"']+", re.I)
PLACEHOLDER = re.compile(r"^(?:null|none|false|true|redacted|masked|not[_-]?(?:set|recorded|sent)|\*+|<.*>|\[.*\])$", re.I)
ARTIFACT_DIRS = {"artifacts", "models", "model", "checkpoints", "checkpoint", "transforms", "transformers", "scalers", "metadata", "splits"}
SOURCE_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".sh"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_url(value):
    """Fail closed for authenticated, signed, tokenized or ambiguous URLs."""
    if not isinstance(value, str):
        return False
    try:
        p = urlsplit(value.rstrip(").,;"))
        decoded = unquote(value)
        if p.scheme not in {"http", "https", "ftp", "file"}:
            return False
        if p.username is not None or p.password is not None:
            return False
        if p.query or p.fragment:  # Includes signed URLs and unknown query tokens.
            return False
        if re.search(r"/(?:t|token|auth|apikey|api-key)/[^/]+", decoded, re.I):
            return False
        if re.search(r"(?:token|secret|password|signature|credential|api[_-]?key)[=:]", decoded, re.I):
            return False
        return bool(p.path) if p.scheme == "file" else bool(p.hostname)
    except ValueError:
        return False


def unsafe_text(text):
    text = re.sub(r"\\u([a-fA-F0-9]{4})", lambda m: chr(int(m.group(1), 16)), text)
    for match in SECRET_VALUE.finditer(text):
        value = match.group(1)
        if not PLACEHOLDER.fullmatch(value):
            return "possible_credential_assignment"
    if re.search(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text):
        return "possible_credential_literal"
    if any(not safe_url(m.group()) for m in URL.finditer(text)):
        return "authenticated_tokenized_or_query_url"
    return None


def unsafe_json(value):
    """Inspect decoded JSON keys, including escaped credential key spellings."""
    if isinstance(value, dict):
        for key, item in value.items():
            normal = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if SENSITIVE_JSON_KEY.fullmatch(normal) and item not in (None, False, ""):
                if not isinstance(item, str) or not PLACEHOLDER.fullmatch(item):
                    return "possible_credential_JSON_key"
            issue = unsafe_json(item)
            if issue:
                return issue
    elif isinstance(value, list):
        for item in value:
            issue = unsafe_json(item)
            if issue:
                return issue
    elif isinstance(value, str):
        return unsafe_text(value)
    return None


def unsafe_path(path):
    return any(SENSITIVE_NAME.search(part) for part in path.parts)


def existing_absolute(value, kind):
    p = Path(value)
    if not p.is_absolute() or not p.exists() or p.is_symlink():
        raise ValueError("{} must be an existing absolute non-symlink path".format(kind))
    p = p.resolve()
    if kind in {"repo", "run-dir"} and not p.is_dir():
        raise ValueError("{} must be a directory".format(kind))
    if kind == "llm-record" and not p.is_file():
        raise ValueError("llm-record must be a regular JSON file")
    return p


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def category(path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "run.json":
        return "run_manifest"
    if name.startswith("config") and suffix in {".yaml", ".yml"}:
        return "workspace_config"
    if name.startswith("prediction") and suffix == ".pkl":
        return "predictions"
    if suffix in {".csv", ".json"} and re.search(r"(?:test|held[_-]?out).*(?:ids?|identit)|(?:ids?).*(?:test|held[_-]?out)", name):
        return "test_identities"
    if re.search(r"fitted|transform|scaler|normaliz|encoder", name) and suffix in {".pkl", ".pickle", ".joblib", ".json", ".npz", ".npy", ".pt", ".pth", ".bin"}:
        return "fitted_transforms"
    if suffix in {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".ubj"}:
        return "model_artifacts"
    if re.search(r"model|checkpoint|estimator|^best|^latest", name) and suffix in {".pkl", ".pickle", ".joblib", ".bin", ".json"}:
        return "model_artifacts"
    return None


def candidates(run):
    # No unrestricted recursive copy: top-level files and whitelisted folders,
    # at most two folder levels below the run. Never follow directory symlinks.
    queue = [(run, 0)]
    while queue:
        folder, depth = queue.pop(0)
        for p in sorted(folder.iterdir()):
            if p.is_symlink() or unsafe_path(p.relative_to(run)):
                continue
            if p.is_file() and category(p):
                yield p
            elif p.is_dir() and depth < 2 and p.name.lower() in ARTIFACT_DIRS:
                queue.append((p, depth + 1))


class Collector:
    def __init__(self, output, repo, runs, record, max_file, max_total):
        self.output, self.repo, self.runs, self.record = output, repo, runs, record
        self.max_file, self.max_total = max_file, max_total
        self.total = 0
        self.entries = []
        self.by_source = {}

    def record_bytes(self, data, original_bytes, original_stat):
        entry = {"source_path": str(self.record), "category": "llm_record",
                 "size_bytes": len(original_bytes), "sha256": hashlib.sha256(original_bytes).hexdigest(),
                 "modified_utc": dt.datetime.fromtimestamp(original_stat.st_mtime, dt.timezone.utc).isoformat()}
        issue = unsafe_json(data) or unsafe_text(original_bytes.decode("utf-8"))
        if len(original_bytes) > self.max_file or self.total + len(original_bytes) > self.max_total:
            entry.update(status="skipped_size_limit")
        elif issue:
            entry.update(status="skipped_possible_credentials", safety_reason=issue)
        else:
            destination = Path("llm_case") / "record.json"
            (self.output / destination).parent.mkdir(mode=0o700)
            with (self.output / destination).open("xb") as handle:
                handle.write(original_bytes)
            entry.update(status="copied", bundle_path=str(destination), copied_sha256=entry["sha256"])
            self.total += len(original_bytes)
        self.entries.append(entry)
        self.by_source[str(self.record)] = entry
        return entry

    def copy(self, source, destination, label):
        source = Path(source)
        key = str(source)
        if key in self.by_source:
            return self.by_source[key]
        entry = {"source_path": key, "category": label, "status": "pending"}
        self.entries.append(entry)
        self.by_source[key] = entry
        if not source.is_absolute() or not source.exists():
            entry["status"] = "missing"
            return entry
        if source.is_symlink() or unsafe_path(source):
            entry["status"] = "skipped_symlink_or_sensitive_path"
            return entry
        resolved = source.resolve()
        if source != self.record and not (within(resolved, self.repo) or any(within(resolved, r) for r in self.runs)):
            entry["status"] = "skipped_outside_authorized_roots"
            return entry
        if not source.is_file():
            entry["status"] = "skipped_not_regular_file"
            return entry
        if len(self.entries) > 600:
            entry["status"] = "skipped_file_count_limit"
            return entry
        try:
            before = source.stat()
        except OSError as exc:
            entry.update(status="read_error", error_type=type(exc).__name__)
            return entry
        entry["size_bytes"] = before.st_size
        entry["modified_utc"] = dt.datetime.fromtimestamp(before.st_mtime, dt.timezone.utc).isoformat()
        if before.st_size > self.max_file or self.total + before.st_size > self.max_total:
            entry["status"] = "skipped_size_limit"
            return entry
        dest = self.output / destination
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        digest, previous_tail, issue = hashlib.sha256(), "", None
        copied = 0
        created = False
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(str(source), flags)
            with os.fdopen(descriptor, "rb") as src, dest.open("xb") as dst:
                created = True
                start = os.fstat(src.fileno())
                if not stat.S_ISREG(start.st_mode):
                    raise ValueError("not regular")
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > self.max_file or self.total + copied > self.max_total:
                        raise ValueError("size_limit_during_read")
                    digest.update(chunk)
                    decoded = chunk.decode("utf-8", errors="ignore")
                    issue = issue or unsafe_text(previous_tail + decoded)
                    previous_tail = decoded[-4096:]
                    dst.write(chunk)
                after = os.fstat(src.fileno())
            # JSON configuration/identity files get decoded-key inspection too.
            if source.suffix.lower() == ".json" and copied <= 16 * 1024 * 1024:
                try:
                    issue = issue or unsafe_json(json.loads(dest.read_bytes()))
                except (ValueError, UnicodeError):
                    pass
            entry["sha256"] = digest.hexdigest()
            if issue:
                dest.unlink()
                entry.update(status="skipped_possible_credentials", safety_reason=issue)
            elif (start.st_size, start.st_mtime_ns, start.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or before.st_size != start.st_size:
                dest.unlink()
                entry["status"] = "skipped_source_changed_during_read"
            else:
                entry.update(status="copied", bundle_path=str(destination), copied_sha256=digest.hexdigest())
                self.total += after.st_size
        except (OSError, ValueError) as exc:
            if created and dest.exists():
                dest.unlink()
            entry.update(status="skipped_size_limit_during_read" if str(exc) == "size_limit_during_read" else "read_error", error_type=type(exc).__name__)
        return entry


def command(arguments, cwd=None, timeout=30):
    env = os.environ.copy()  # Inherited for execution only; never serialized.
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_PAGER="cat", GIT_TERMINAL_PROMPT="0")
    try:
        process = subprocess.Popen(arguments, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = bytearray()
        total = 0
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, True)
        selector.register(process.stderr, selectors.EVENT_READ, False)
        try:
            while selector.get_map():
                if time.monotonic() > deadline:
                    process.kill()
                    process.wait()
                    return {"status": "command_timeout"}, None
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > 8 * 1024 * 1024:
                        process.kill()
                        process.wait()
                        return {"status": "skipped_output_limit"}, None
                    if key.data:
                        output.extend(chunk)
            returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            if returncode:
                return {"status": "command_failed", "returncode": returncode}, None
            return {"status": "captured"}, output.decode("utf-8", errors="replace")
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__}, None


def git_snapshot(repo, output):
    info = {"scope": "CURRENT repository at collection time; not an assertion about either historical run"}
    prefix = ["git", "--literal-pathspecs", "--no-pager", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null"]
    for name, args in [("head", ["rev-parse", "HEAD"]),
                       ("status", ["status", "--porcelain=v1", "-z", "--untracked-files=normal"])]:
        state, value = command(prefix + args, repo)
        if value is not None:
            issue = unsafe_text(value)
            if issue:
                state = {"status": "skipped_possible_credentials", "safety_reason": issue}
            else:
                if name == "status":
                    value = "\n".join(line for line in value.split("\0") if line and not unsafe_path(Path(line[3:]))) + "\n"
                write_text(output / "current_git" / (name + ".txt"), value)
        info[name] = state
    # Get changed tracked paths without evaluating external diff/textconv programs.
    path_state, paths = command(prefix + ["diff", "--name-only", "-z", "--no-ext-diff", "--no-textconv"], repo)
    staged_state, staged_paths = command(prefix + ["diff", "--cached", "--name-only", "-z", "--no-ext-diff", "--no-textconv"], repo)
    allowed, excluded = [], []
    if paths is not None or staged_paths is not None:
        for item in sorted(set(((paths or "") + "\0" + (staged_paths or "")).split("\0"))):
            if not item:
                continue
            p = Path(item)
            (allowed if p.suffix.lower() in SOURCE_EXTENSIONS and not unsafe_path(p) else excluded).append(item)
    info["diff_path_selection"] = {"status": path_state["status"], "staged_status": staged_state["status"], "allowed_count": len(allowed), "excluded_count": len(excluded), "rule": "union of staged and unstaged tracked source/text paths; sensitive names excluded; all content scanned"}
    for name, options in [("unstaged.diff", []), ("staged.diff", ["--cached"])]:
        if not allowed:
            info[name] = {"status": "no_allowed_changed_paths"}
            continue
        state, value = command(prefix + ["diff"] + options + ["--no-ext-diff", "--no-textconv", "--"] + allowed, repo)
        if value is not None:
            issue = unsafe_text(value)
            if issue:
                state = {"status": "skipped_possible_credentials", "safety_reason": issue}
            else:
                write_text(output / "current_git" / name, value)
        info[name] = state
    return info


def environment_snapshot(output):
    info = {"scope": "CURRENT interpreter/environment at export time; not a tested lock and not proof of 12 October or 17 October runtime",
            "collected_utc": now(), "python_version": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(), "distributions": []}
    requirements = []
    for dist in metadata.distributions():
        try:
            name, version = dist.metadata.get("Name", ""), dist.version
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or not re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
                continue
            installer = (dist.read_text("INSTALLER") or "").strip()
            entry = {"name": name, "version": version, "installer": installer if installer in {"pip", "conda", "uv"} else "unknown"}
            direct = dist.read_text("direct_url.json")
            if direct:
                try:
                    value = json.loads(direct)
                    if not isinstance(value, dict):
                        raise ValueError("direct_url_not_object")
                    url = value.get("url", "")
                    entry["direct_source_status"] = "safe_url_recorded" if safe_url(url) else "url_omitted_for_safety"
                    if safe_url(url):
                        entry["direct_source_url"] = url
                    directory = value.get("dir_info", {})
                    entry["editable"] = bool(directory.get("editable", False)) if isinstance(directory, dict) else False
                except (ValueError, TypeError):
                    entry["direct_source_status"] = "unreadable"
            if installer == "pip":
                requirements.append("{}=={}".format(name, version))
            info["distributions"].append(entry)
        except (OSError, ValueError, TypeError):
            continue
    info["distributions"].sort(key=lambda e: (e["name"].lower(), e["version"]))
    write_text(output / "current_environment" / "pip_installer_requirements.txt",
               "# CURRENT installed distributions whose INSTALLER is pip.\n# Version inventory only; not a resolver lock or historical/tested environment.\n" + "\n".join(sorted(set(requirements), key=str.lower)) + "\n")
    executable = shutil.which("conda")
    if executable and (Path(sys.prefix) / "conda-meta").is_dir():
        state, value = command([executable, "list", "--prefix", sys.prefix, "--explicit"])
        state["prefix"] = sys.prefix
        if value is not None:
            issue = unsafe_text(value)
            # Every non-comment explicit package line must be @EXPLICIT or a safe URL.
            if not issue:
                for line in value.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and line != "@EXPLICIT" and not safe_url(line):
                        issue = "unrecognized_or_unsafe_explicit_entry"
                        break
            if issue:
                state = {"status": "skipped_possible_credentials", "safety_reason": issue, "raw_output_saved": False}
            else:
                write_text(output / "current_environment" / "conda_explicit_CURRENT.txt", value)
                state["raw_output_saved"] = True
        info["conda_explicit"] = state
    else:
        info["conda_explicit"] = {"status": "current_interpreter_not_conda" if executable else "conda_unavailable", "raw_output_saved": False}
    write_json(output / "current_environment" / "installed_distributions_CURRENT.json", info)
    return {k: v for k, v in info.items() if k != "distributions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Existing absolute BatteryML repository path")
    parser.add_argument("--output", required=True, help="New absolute output directory; its parent must exist; directory and sibling .zip must not exist")
    parser.add_argument("--llm-record", required=True, help="Existing absolute selected 4c record.json path")
    parser.add_argument("--run-dir", action="append", required=True, help="Existing absolute run directory; repeat for 12 October and 17 October")
    parser.add_argument("--max-file-mib", type=int, default=1024, help="Maximum per-file size (default 1024 MiB)")
    parser.add_argument("--max-total-mib", type=int, default=4096, help="Maximum total copied source bytes (default 4096 MiB)")
    args = parser.parse_args()
    try:
        repo = existing_absolute(args.repo, "repo")
        record = existing_absolute(args.llm_record, "llm-record")
        runs = list(dict.fromkeys(existing_absolute(p, "run-dir") for p in args.run_dir))
        output = Path(args.output)
        if not output.is_absolute() or not output.parent.is_dir() or output.exists() or Path(str(output) + ".zip").exists():
            raise ValueError("output must be a NEW absolute directory with an existing parent; sibling .zip must not exist")
        output = output.parent.resolve() / output.name
        if any(within(output, p) for p in [repo] + runs):
            raise ValueError("output must be outside the repository and run directories")
        if min(args.max_file_mib, args.max_total_mib) <= 0:
            raise ValueError("size limits must be positive")
        if record.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("LLM record exceeds 16 MiB")
        with record.open("rb") as handle:
            record_stat = os.fstat(handle.fileno())
            record_bytes = handle.read(16 * 1024 * 1024 + 1)
            record_after = os.fstat(handle.fileno())
        if len(record_bytes) > 16 * 1024 * 1024 or (record_stat.st_size, record_stat.st_mtime_ns, record_stat.st_ino) != (record_after.st_size, record_after.st_mtime_ns, record_after.st_ino):
            raise ValueError("LLM record too large or changed during read")
        data = json.loads(record_bytes.decode("utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("LLM record must be a JSON object")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    os.umask(0o077)
    output.mkdir(mode=0o700)
    collector = Collector(output, repo, runs, record, args.max_file_mib * 1024 ** 2, args.max_total_mib * 1024 ** 2)
    manifest = {"schema": SCHEMA, "collection_started_utc": now(), "purpose": "Copy existing evidence only; no numerical validation, training, LLM invocation or pickle deserialization",
                "historical_boundary": "Source bytes and record fingerprints are preserved. Current Git and environment snapshots do not establish past commit/environment or completion of proposed checks.",
                "record_id": data.get("record_id") if unsafe_json(data) is None else "omitted_for_safety", "runs": [], "fingerprint_checks": [],
                "selection_limits": {"max_file_mib": args.max_file_mib, "max_total_mib": args.max_total_mib, "max_directory_depth": 2, "max_candidate_files": 600}}
    record_entry = collector.record_bytes(data, record_bytes, record_stat)
    for index, run in enumerate(runs, 1):
        counts = {key: 0 for key in ["run_manifest", "workspace_config", "predictions", "fitted_transforms", "model_artifacts", "test_identities"]}
        selected = []
        for source in candidates(run):
            label = category(source)
            entry = collector.copy(source, Path("runs") / ("{:02d}_{}".format(index, run.name)) / source.relative_to(run), label)
            selected.append(entry)
            if entry["status"] == "copied":
                counts[label] += 1
        if not any(e["category"] == "run_manifest" for e in selected):
            collector.copy(run / "run.json", Path("runs") / ("{:02d}_{}".format(index, run.name)) / "run.json", "run_manifest")
        manifest["runs"].append({"source_directory": str(run), "copied_category_counts": counts,
                                 "category_status": {key: "copied" if count else "not_found_or_not_copied; do not infer non-use" for key, count in counts.items()}})
    fields = data.get("files", {}) if record_entry["status"] == "copied" else {}
    if not isinstance(fields, dict):
        fields = {}
    linked = [(key, fields.get(key)) for key in ["config", "run_manifest", "predictions"]]
    if isinstance(fields.get("workspace_configs"), list):
        linked.extend(("workspace_configs[{}]".format(i), v) for i, v in enumerate(fields["workspace_configs"]))
    for field, value in linked:
        check = {"record_field": "files." + field, "status": "record_field_absent"}
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            source = Path(value["path"])
            valid_kind = ((field == "config" or field.startswith("workspace_configs")) and source.suffix.lower() in {".yaml", ".yml"}) or (field == "run_manifest" and source.name == "run.json") or (field == "predictions" and source.suffix.lower() == ".pkl" and source.name.lower().startswith("prediction"))
            if not valid_kind:
                check.update(status="skipped_record_reference_wrong_file_type")
                manifest["fingerprint_checks"].append(check)
                continue
            entry = collector.copy(source, Path("record_linked") / field.replace("[", "_").replace("]", "") / source.name, "record_linked_" + field)
            check.update(source_path=str(source), recorded_sha256=value.get("sha256"), recorded_size_bytes=value.get("size_bytes"), current_file_status=entry["status"], current_sha256=entry.get("sha256"), bundle_path=entry.get("bundle_path"))
            expected = value.get("sha256")
            if entry["status"] == "copied" and entry.get("sha256") and isinstance(expected, str) and re.fullmatch(r"[a-fA-F0-9]{64}", expected):
                check.update(status="match" if entry["sha256"].lower() == expected.lower() else "MISMATCH; current file does not replace historical evidence",
                             sha256_matches=entry["sha256"].lower() == expected.lower(), size_matches=entry.get("size_bytes") == value.get("size_bytes"))
            else:
                check["status"] = "not_verified: " + entry["status"] if entry["status"] != "copied" else "recorded_sha256_missing_or_invalid"
        manifest["fingerprint_checks"].append(check)
    manifest["current_git"] = git_snapshot(repo, output)
    manifest["current_environment"] = environment_snapshot(output)
    manifest["source_files"] = collector.entries
    manifest["copied_source_bytes"] = collector.total
    manifest["collection_finished_utc"] = now()
    # Hash generated inventories too; manifest itself is intentionally excluded.
    manifest["bundle_file_inventory"] = []
    for p in sorted(output.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest["bundle_file_inventory"].append({"path": str(p.relative_to(output)), "size_bytes": p.stat().st_size, "sha256": digest.hexdigest()})
    write_json(output / "manifest.json", manifest)
    archive = Path(str(output) + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                handle.write(p, arcname=str(Path(output.name) / p.relative_to(output)))
    print("Created folder: {}\nCreated ZIP: {}\nReview manifest.json for missing, skipped and MISMATCH entries.".format(output, archive))


if __name__ == "__main__":
    main()
