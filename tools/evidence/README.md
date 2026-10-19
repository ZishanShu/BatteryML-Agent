# Evidence collection (temporary, manuscript preparation)

`collect_reference_evidence.py` is copied **byte for byte** from the
reproducibility preparation package (`Reproducibility_preparation.zip`) so it
can reach the server with `git pull`. It is not part of the platform: nothing
imports it, and it is to be removed once the manuscript evidence is collected.

| File | Size | SHA-256 |
| --- | --- | --- |
| `collect_reference_evidence.py` | 28640 bytes | `3efc6f5a2f03152b917156eed17f590543673999a508347469a07829ea4c6db1` |

Check before running:

```bash
sha256sum tools/evidence/collect_reference_evidence.py
```

What it does (from reading its source): copies existing run files
(`run.json`, configs, prediction/model/transform files), the LLM record and the
files that record points to into a **new** output folder and ZIP outside the
repository, and records the current git state and installed packages. It uses
the standard library only, runs `git` and `conda list` read-only, never
unpickles, trains, calls an LLM or writes into the repository or run
directories, and skips files that look like credentials. Its current-git and
current-environment snapshots describe the moment of collection, not the
historical runs.

Usage is in the package's `RUN_INSTRUCTIONS.txt`; for this repository on the
server:

```bash
cd /root/share/szs/BatteryML
python tools/evidence/collect_reference_evidence.py --repo $PWD \
  --output /root/share/szs/new_reference_evidence \
  --llm-record <absolute path to record.json> --run-dir <absolute run dir> [--run-dir <absolute run dir>]
```
