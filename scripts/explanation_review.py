"""List LLM explanation records and (re)build a review sheet from one (EXPL-02).

Read-only towards the record: it never calls a model, never edits
``record.json`` and never overwrites an existing review sheet, which may
already carry an author's judgements.

    # Every record of a project, newest last
    python -m scripts.explanation_review --list runs/web_projects/matr_protocol

    # Print a record's identity, models, settings and outcome
    python -m scripts.explanation_review --show runs/web_projects/<project>/llm_explanations/<run>/<record>/record.json

    # Build a fresh review sheet into a NEW folder (e.g. after the checks improved)
    python -m scripts.explanation_review --record <record.json> --out runs/explanation_review/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from batteryml_web.agent import explanation_checklist as ec
from batteryml_web.agent import explanation_record as er


def _summary(record: dict) -> dict:
    call = record.get("call") or {}
    result = record.get("result") or {}
    files = record.get("files") or {}
    return {
        "record_id": record.get("record_id"),
        "outcome": record.get("outcome"),
        "project": result.get("project"),
        "run_id": result.get("run_id"),
        "task": result.get("task"),
        "model_key": result.get("model_key"),
        "code_commit": (record.get("code") or {}).get("commit"),
        "run_git_commit": result.get("run_git_commit"),
        "config": {k: (files.get("config") or {}).get(k) for k in ("path", "sha256")},
        "predictions": {k: (files.get("predictions") or {}).get(k) for k in ("path", "sha256")},
        "sent_at": call.get("sent_at"),
        "requested_model": call.get("requested_model"),
        "response_model": call.get("response_model"),
        "generation_settings_sent": call.get("generation_settings_sent"),
        "stage": call.get("stage"),
        "error": (record.get("error") or {}).get("message"),
        "human_review": record.get("human_review"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", metavar="PROJECT_DIR", help="list the records under a project folder")
    group.add_argument("--show", metavar="RECORD_JSON", help="print a record's summary as JSON")
    group.add_argument("--record", metavar="RECORD_JSON", help="build a review sheet from this record")
    parser.add_argument("--out", metavar="DIR", help="with --record: a folder that must not already hold a sheet")
    args = parser.parse_args(argv)

    if args.list:
        records = er.list_records(args.list)
        if not records:
            print(f"No explanation records under {Path(args.list) / er.RECORDS_DIRNAME}.")
            return 1
        for path in records:
            record = er.load_record(path)
            print(f"{path}  outcome={record.get('outcome')}  run={record['result'].get('run_id')}  "
                  f"model={record['call'].get('response_model') or record['call'].get('requested_model')}")
        return 0

    if args.show:
        print(json.dumps(_summary(er.load_record(args.show)), ensure_ascii=False, indent=2))
        return 0

    if not args.out:
        print("--record needs --out: a new folder for the sheet (an existing sheet is never overwritten).",
              file=sys.stderr)
        return 2
    record = er.load_record(args.record)
    try:
        paths = ec.write_checklist(record, args.out)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sheet = ec.build_checklist(record)
    counts = {label: sum(1 for row in sheet["rows"] if row["automatic_result"] == label)
              for label in (ec.FLAG, ec.NOTE, ec.NONE_FOUND)}
    print(f"{len(sheet['rows'])} claim(s): {counts}. Human review: NOT STARTED.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
