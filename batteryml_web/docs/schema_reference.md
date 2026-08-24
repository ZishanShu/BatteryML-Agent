# BatteryML Schema Reference

This document describes the canonical BatteryML-facing fields used by the local onboarding workflow.

## cycle_index

- Description: Sequential cycle number for each record or measurement group.
- Required or optional: Required
- Expected dtype: integer
- Expected unit: cycle count
- Example raw column names: `cycle`, `cycle_index`, `cycle_id`, `step_cycle`
- Validation rules:
  - Must be numeric and non-null for mapped records.
  - Prefer monotonically increasing values within a cell.

## timestamp

- Description: Time of measurement or cycle event.
- Required or optional: Optional
- Expected dtype: datetime, string, or integer epoch
- Expected unit: seconds, milliseconds, ISO datetime
- Example raw column names: `timestamp`, `time`, `date_time`, `datetime`
- Validation rules:
  - If present, values should be parseable as time-like data.

## voltage

- Description: Measured voltage.
- Required or optional: Required
- Expected dtype: float
- Expected unit: V
- Example raw column names: `voltage`, `V`, `volt`
- Validation rules:
  - Must be numeric.
  - Recommended to be non-negative.

## current

- Description: Measured current.
- Required or optional: Required
- Expected dtype: float
- Expected unit: A
- Example raw column names: `current`, `I`, `ampere`
- Validation rules:
  - Must be numeric.
  - Signed values are allowed when charge and discharge polarity is encoded.

## charge_capacity

- Description: Charge capacity for a cycle or step.
- Required or optional: Optional
- Expected dtype: float
- Expected unit: Ah or mAh
- Example raw column names: `charge_capacity`, `chg_cap`, `Q_charge`
- Validation rules:
  - Must be numeric if mapped.
  - Unit should be documented if known.

## discharge_capacity

- Description: Discharge capacity for a cycle or step.
- Required or optional: Required
- Expected dtype: float
- Expected unit: Ah or mAh
- Example raw column names: `discharge_capacity`, `dchg_cap`, `Q_discharge`
- Validation rules:
  - Must be numeric.
  - Required for most BatteryML workflows.

## temperature

- Description: Temperature of the cell or environment.
- Required or optional: Optional
- Expected dtype: float
- Expected unit: °C
- Example raw column names: `temperature`, `temp`, `T`
- Validation rules:
  - Must be numeric if mapped.

## cell_id

- Description: Unique identifier for a cell.
- Required or optional: Optional
- Expected dtype: string, integer, or categorical
- Expected unit: identifier
- Example raw column names: `cell_id`, `sample_id`, `battery_id`, `cell`
- Validation rules:
  - Should uniquely identify a cell when used for grouping.

## dataset_name

- Description: Dataset or cohort name.
- Required or optional: Optional
- Expected dtype: string
- Expected unit: dataset label
- Example raw column names: `dataset`, `dataset_name`, `source`
- Validation rules:
  - Useful for provenance and reporting.

## step_index

- Description: Step number within a cycle if the dataset is step-resolved.
- Required or optional: Optional
- Expected dtype: integer
- Expected unit: step count
- Example raw column names: `step`, `step_index`, `substep`
- Validation rules:
  - Must be numeric when present.
  - Use only when the raw data contains step-level measurements.

## General validation notes

- Raw files are never modified in place.
- A mapping must be validated before a standardized dataset is exported.
- The standardized dataset should preserve raw-column provenance.
- Some BatteryML models may require additional adapters beyond the standardized CSV or Parquet export.
