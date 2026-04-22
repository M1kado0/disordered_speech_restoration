# Disordered Speech Restoration

Whisper medium baseline and LoRA fine-tuning for private UASpeech restoration experiments.

## Setup

Install dependencies, then authenticate to Hugging Face before loading the private dataset:

```bash
pip install -e .
```

```bash
huggingface-cli login
```

## Run

Zero-shot Whisper baseline:

```bash
python -m src.baseline --hf-token "$HF_TOKEN"
```

LoRA fine-tuning:

```bash
python -m src.train_lora --hf-token "$HF_TOKEN"
```

Comparison run:

```bash
python -m src.evaluate --hf-token "$HF_TOKEN"
```

## Metrics

Primary metrics are WER and CER.

PESQ and STOI are included as audio-quality comparison helpers for original versus generated waveforms.

## Notes

- The dataset is private, so Hugging Face authentication is required.
- The first implementation excludes any LLM-based post-processing.
- If the dataset does not expose official train/validation/test splits, add a deterministic split step next.
# Disordered Speech Restoration
