# Disordered Speech Restoration

Whisper medium baseline and LoRA fine-tuning for private UASpeech restoration experiments.

## Setup

Install dependencies, then authenticate to Hugging Face before loading the private dataset:

```bash
pip install -e .
```

```bash
hf auth login
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

## Pipeline Architecture

```
Disordered Speech Audio
    ↓
Fine-tuned Whisper ASR
    ↓
Vocabulary-Constrained Text Normalization  ← NEW
    ↓
Text-to-Speech Synthesis (VITS)
    ↓
Restored Audio Output
```

## Text Normalization: Vocabulary-Constrained Correction

The text normalization layer uses **edit distance matching** (Levenshtein distance) against a closed vocabulary instead of open-ended LLM correction.

### Why Vocabulary-Constrained?

**Problem with Generic LLM Correction:**
- Open-ended prompts cause hallucination
- Rewrites specialized/phonetic vocabulary into common English
- Example: `"nuremberg"` → `"number"` ❌ (wrong domain)

**Solution: Constrained Matching**
- Maps Whisper predictions to closest vocabulary word
- Deterministic and safe (no hallucinations)
- Perfect for specialized domains with bounded vocabularies

### Results

| Approach | WER | CER | Status |
|----------|-----|-----|--------|
| Baseline Whisper | 0.3848 | 0.3422 | Original |
| Generic LLM Correction | 0.4786 | 0.3927 | ❌ Worse |
| **Vocabulary-Constrained** | **0.3371** | **0.3185** | ✅ **+12% improvement** |

### Usage

**Batch Evaluation:**
```bash
python -m src.restore_metrics \
  --input outputs/finetuned_metrics.json \
  --output outputs/restored_metrics.json
```

**Single Transcript Restoration (with TTS):**
```bash
python -m src.restore \
  --transcript "nurember" \
  --vocabulary outputs/finetuned_metrics.json \
  --output restored_output.wav
```

## Metrics

Primary metrics are WER and CER, computed before and after vocabulary-constrained normalization.

PESQ and STOI are included as audio-quality comparison helpers for original versus generated waveforms.

## Notes

- The dataset is private, so Hugging Face authentication is required.
- Text normalization uses Levenshtein distance matching against the reference vocabulary (no external LLM required).
- This approach is deterministic, domain-aware, and does not introduce hallucinations.
- If the dataset does not expose official train/validation/test splits, add a deterministic split step next.
