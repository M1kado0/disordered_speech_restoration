from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.data import get_split, load_private_dataset
from src.metrics import compute_asr_metrics
from src.modeling import load_whisper_processor_and_model
from src.preprocess import normalize_text


def run_baseline(token: str | None = None, split_name: str | None = None):
    config = load_config()
    bundle = load_private_dataset(config.dataset, token=token)
    processor, model = load_whisper_processor_and_model(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    split_name = split_name or bundle.test_split
    split = get_split(bundle, split_name)

    predictions: list[str] = []
    references: list[str] = []

    for example in split:
        audio = example[config.dataset.audio_column]
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_features)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(normalize_text(text))
        references.append(normalize_text(str(example[config.dataset.text_column])))

    metrics = compute_asr_metrics(predictions, references)
    return {
        "split": split_name,
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot Whisper baseline")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for private dataset access")
    parser.add_argument("--split", default=None, help="Dataset split to evaluate")
    parser.add_argument("--output", default="outputs/baseline_metrics.json")
    args = parser.parse_args()

    result = run_baseline(token=args.hf_token, split_name=args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
