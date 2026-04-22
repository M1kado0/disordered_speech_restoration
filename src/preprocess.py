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
        audio_info = batch[config.dataset.audio_column]
        transcript = batch[config.dataset.text_column]

        # Manually decode audio from bytes since we bypass torchcodec
        if audio_info.get("bytes"):
            import io
            import soundfile as sf
            array, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            import soundfile as sf
            array, sr = sf.read(audio_info["path"])

        features = processor.feature_extractor(
            array,
            sampling_rate=sr,
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
