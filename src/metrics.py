from __future__ import annotations

from dataclasses import dataclass

import evaluate
import numpy as np
from pesq import pesq as pesq_score
from pystoi import stoi as stoi_score


_wer = evaluate.load("wer")
_cer = evaluate.load("cer")


@dataclass(frozen=True)
class MetricBundle:
    wer: float
    cer: float
    pesq: float | None = None
    stoi: float | None = None


def compute_asr_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {
        "wer": float(_wer.compute(predictions=predictions, references=references)),
        "cer": float(_cer.compute(predictions=predictions, references=references)),
    }


def compute_audio_metrics(
    reference_audio: np.ndarray,
    generated_audio: np.ndarray,
    sampling_rate: int,
) -> dict[str, float]:
    reference_audio = np.asarray(reference_audio, dtype=np.float32)
    generated_audio = np.asarray(generated_audio, dtype=np.float32)

    if reference_audio.ndim > 1:
        reference_audio = np.mean(reference_audio, axis=0)
    if generated_audio.ndim > 1:
        generated_audio = np.mean(generated_audio, axis=0)

    limit = min(len(reference_audio), len(generated_audio))
    if limit == 0:
        raise ValueError("Audio arrays must not be empty")

    reference_audio = reference_audio[:limit]
    generated_audio = generated_audio[:limit]

    mode = "wb" if sampling_rate >= 16000 else "nb"
    return {
        "pesq": float(pesq_score(sampling_rate, reference_audio, generated_audio, mode)),
        "stoi": float(stoi_score(reference_audio, generated_audio, sampling_rate, extended=False)),
    }
