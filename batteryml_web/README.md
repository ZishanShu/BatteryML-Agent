# BatteryML Localhost Protocol Platform

`batteryml_web` is a local-first BatteryML platform for teaching, demos, and reproducible protocol-style workflows.

## Design goals

- BatteryML performs all real computation: preprocessing, feature extraction, training, and evaluation
- The web app manages local interaction, status display, file paths, plots, and report export
- LLM helpers are optional and only assist with explanation, mapping suggestions, troubleshooting, and result summaries
- No API key is required for the Quick Start workflow
- LLM output must always be reviewed by the user or validated before it is used

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run batteryml_web/app.py
```

The platform stores project artifacts locally under `runs/web_projects/<project_name>/`.

## Quick Start

This is the recommended demo path and it works without any API key.

### Fixed example

- Dataset: `MATR`
- Device: `CPU`
- Task: `RUL`
- Model: `XGBoostRULPredictor`
- Config template: `configs/baselines/sklearn/xgb/matr_1.yaml`

### What to do

1. Open the app.
2. Enter a project name.
3. Choose `Quick-start without LLM`.
4. Leave LLM fields empty.
5. Click `Run quick-start`.

### What you will see

- raw data path
- processed data path
- config path
- workspace path
- output folder
- training log
- metrics
- prediction artifacts
- feature information

## LLM helper buttons

The app exposes LLM assistance as step-level buttons instead of a separate page:

- Use LLM to explain columns
- Use LLM to suggest mapping
- Use LLM to explain error
- Use LLM to summarize results

### Safety rules

- LLM never auto-modifies raw data
- LLM suggestions must pass validation or be confirmed by the user
- Validators gate progression between workflow steps
- If no API key is configured, Quick Start still runs normally

## Output folder

Each project run is stored under:

`runs/web_projects/<project_name>/`

Common files include:

- `project_state.json`
- `quickstart_config.yaml`
- `generated_config.yaml`
- `schema_mapping.json`
- `schema_validation_report.json`
- `config_validation_report.json`
- `training_log.txt`
- `metrics.csv`
- `predictions_seed_0_*.pkl`
- `report.html`
- `error_diagnosis.json`
- `agent_fix_suggestion.md`
- `validation_after_fix.json`
- `rerun_log.txt`

## How to verify

### Quick Start verification

- Create a project
- Use the MATR quick-start path
- Confirm the project state changes to a successful run
- Open `training_log.txt`
- Check `metrics.csv` or prediction artifacts if BatteryML emits them

### LLM helper verification

- Leave API key blank
- Confirm Quick Start still runs
- Upload a tabular file
- Inspect the columns and create a mapping
- Run validation before saving the mapping

### Troubleshooting verification

- Use a failed run
- Ask the LLM to explain the error
- Review the suggestion manually
- Run the validator after your manual fix

## Known limitations

- The web UI is still a local wrapper around the existing BatteryML CLI
- Artifact discovery depends on what BatteryML writes to the workspace
- Plot rendering is limited to whatever figures are produced by the underlying run
