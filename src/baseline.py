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


def run_baseline(token: str | None = None, split_name: str | None = None, adapter_path: str | None = None):
    config = load_config()
    bundle = load_private_dataset(config.dataset, token=token)
    processor, model = load_whisper_processor_and_model(config)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    split_name = split_name or bundle.test_split
    split = get_split(bundle, split_name)

    predictions: list[str] = []
    references: list[str] = []

    # Configure DataLoader and Batch Processing for speed!
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    def collate_fn(batch):
        audio_arrays = []
        texts = []
        for item in batch:
            # Manually decode audio from bytes since we bypass torchcodec
            audio_info = item[config.dataset.audio_column]
            if audio_info.get("bytes"):
                import io
                import soundfile as sf
                array, sr = sf.read(io.BytesIO(audio_info["bytes"]))
                audio_arrays.append({"array": array, "sampling_rate": sr})
            else:
                import soundfile as sf
                array, sr = sf.read(audio_info["path"])
                audio_arrays.append({"array": array, "sampling_rate": sr})
            
            texts.append(item[config.dataset.text_column])

        return {
            "audio": audio_arrays,
            "text": texts
        }
        
    eval_dataloader = DataLoader(
        split,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    feature_dtype = dtype_mapping.get(config.model.torch_dtype, torch.float32)

    for batch in tqdm(eval_dataloader, desc="Running baseline evaluation"):
        # Combine batches of audio
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        sampling_rates = batch["audio"][0]["sampling_rate"]

        inputs = processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rates,
            return_attention_mask=False,
            return_tensors="pt",
        )
        
        # Cast inputs explicitly to the same precision type your model is configured as
        input_features = inputs.input_features.to(device, dtype=feature_dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features, 
                language="en", 
                task="transcribe" 
            )
            
        decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        predictions.extend([normalize_text(text) for text in decoded_texts])
        references.extend([normalize_text(str(text)) for text in batch["text"]])

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
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter to load on top of the base model")
    args = parser.parse_args()

    result = run_baseline(token=args.hf_token, split_name=args.split, adapter_path=args.adapter_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
