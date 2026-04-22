from __future__ import annotations

import re
from typing import Any

from transformers import WhisperProcessor

from src.config import AppConfig


_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", "", text)
    return _whitespace_re.sub(" ", text)


def build_preprocess_fn(processor: WhisperProcessor, config: AppConfig):
    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[config.dataset.audio_column]
        transcript = batch[config.dataset.text_column]

        features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        labels = processor.tokenizer(
            normalize_text(str(transcript)),
            max_length=config.model.max_label_length,
            truncation=True,
        ).input_ids

        batch["input_features"] = features["input_features"][0]
        batch["labels"] = labels
        batch["normalized_text"] = normalize_text(str(transcript))
        return batch

    return preprocess
