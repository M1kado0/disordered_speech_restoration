from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split_strategy: str
    train_split: str
    validation_split: str
    test_split: str
    audio_column: str
    text_column: str
    speaker_column: str
    target_sampling_rate: int


@dataclass(frozen=True)
class ModelConfig:
    base_id: str
    language: str
    task: str
    max_label_length: int
    torch_dtype: str = "float32"
    attn_implementation: str = "eager"


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    num_train_epochs: int
    weight_decay: float
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    logging_steps: int
    fp16: bool
    gradient_checkpointing: bool
    max_train_steps: int | None


@dataclass(frozen=True)
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: list[str]
    prediction_output: str


@dataclass(frozen=True)
class AppConfig:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    lora: LoraConfig
    evaluation: EvaluationConfig


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{key}' section in config")
    return value


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parents[1] / config_path
    raw = yaml.safe_load(config_path.read_text())

    return AppConfig(
        seed=int(raw["seed"]),
        dataset=DatasetConfig(**_section(raw, "dataset")),
        model=ModelConfig(**_section(raw, "model")),
        training=TrainingConfig(**_section(raw, "training")),
        lora=LoraConfig(**_section(raw, "lora")),
        evaluation=EvaluationConfig(**_section(raw, "evaluation")),
    )
