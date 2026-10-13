"""The prompt behind Run Model's "Explain this config".

It used to send only the config's *path*, so the assistant had to guess the
contents from a file name. With ``configs/external/batlinet/matr_1.yaml`` it
guessed the dataset was "BatLiNet" and the model "MATR" — the reverse — and
invented sections (``data:``, ``train:``, ``eval:``) that a BatteryML config
does not have.

The prompt now carries what the page already knows for certain — dataset,
task and model, resolved from the config registry — and the file's own text,
and it tells the assistant how a BatteryML config is laid out and to explain
only what the file says.

Pure: no Streamlit, no file reads, so it is testable on its own.
"""

from __future__ import annotations

from typing import Optional

#: Upper bound on the YAML sent. The repository's largest config is ~2 KB.
MAX_CONFIG_CHARS = 6000

CONFIG_EXPLANATION_SYSTEM_PROMPT = (
    "You are an assistant for a BatteryML web protocol platform. Explain the "
    "given BatteryML config using only its own text and the facts stated in "
    "the request. Provide suggestions only. Do not modify raw data, metrics, "
    "predictions, or config files, and never state a metric or benchmark value."
)


def build_config_explanation_prompt(
    *,
    dataset: str,
    task: str,
    model_key: str,
    model_display: Optional[str],
    config_path: str,
    config_text: Optional[str],
) -> str:
    """The user prompt for "Explain this config".

    ``config_text`` is the file's content, or None when it could not be read;
    the prompt then says so instead of inviting a guess.
    """
    model_label = f"{model_display} (`{model_key}`)" if model_display else f"`{model_key}`"
    if config_text is None:
        content = (
            "The config file could not be read. Say that you cannot explain its "
            "contents, and do not describe what it might contain."
        )
    else:
        text = config_text
        truncated = len(text) > MAX_CONFIG_CHARS
        if truncated:
            text = text[:MAX_CONFIG_CHARS]
        content = f"```yaml\n{text}\n```"
        if truncated:
            content += f"\n(Truncated to the first {MAX_CONFIG_CHARS} characters.)"

    return f"""Explain this BatteryML run configuration to a battery scientist.

Facts established by the platform (use them exactly; do not swap or rename them):
- Dataset: {dataset}
- Prediction task: {task}
- Model: {model_label}
- Config file: {config_path}

Config file content:
{content}

How a BatteryML config is organised:
- `model`: the model class (`model.name`) and its hyperparameters.
- `train_test_split`: how cells are split into train/test (`name`), and
  `cell_data_path`, the processed dataset the run reads.
- `feature`: the feature extractor (`name`) and its settings.
- `label`: the label annotator, which defines the prediction target.
- `feature_transformation` / `label_transformation`: normalisation steps.

Rules:
- Explain only the sections and keys that appear in the file above. Do not
  add sections, keys or values that are not in it, and do not show a
  reconstructed or "typical" config.
- The dataset is the one named above (and in `cell_data_path`); the model is
  the one in `model.name`. Never present a model name as a dataset or the
  reverse.
- If a setting's meaning is not clear from the file, say so rather than guess.
- Do not state metric values, expected accuracy or benchmark numbers.

Answer with short sections: what the run does; model and key hyperparameters;
data split; features and label; anything a user should check before running.
"""
