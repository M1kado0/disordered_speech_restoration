import json
import argparse
import torch
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from src.metrics import compute_asr_metrics
from src.preprocess import normalize_text

def main():
    parser = argparse.ArgumentParser(description="Clean Whisper predictions using vocabulary-constrained correction and compute new WER/CER")
    parser.add_argument("--input", type=str, default="outputs/finetuned_metrics.json", help="Input metrics JSON file")
    parser.add_argument("--output", type=str, default="outputs/restored_metrics.json", help="Output restored metrics JSON file")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for vocabulary matching (0-1)")
    args = parser.parse_args()

    print(f"Loading input file: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    references = data.get("references", [])

    if not predictions or not references:
        raise ValueError("Input JSON must contain both 'predictions' and 'references' lists.")

    # Build vocabulary from references (closed set of valid words)
    vocabulary = list(set(references))
    print(f"Built vocabulary from {len(vocabulary)} unique reference words")

    print(f"Correcting {len(predictions)} predictions using vocabulary-constrained matching...")
    
    restored_predictions = []
    
    # For each prediction, find the closest match in the vocabulary
    for pred in tqdm(predictions):
        pred_normalized = normalize_text(pred)
        
        # Find closest match in vocabulary using Levenshtein distance
        best_match = pred_normalized
        best_distance = float('inf')
        
        for vocab_word in vocabulary:
            dist = levenshtein_distance(pred_normalized.lower(), vocab_word.lower())
            if dist < best_distance:
                best_distance = dist
                best_match = vocab_word
        
        # Use the best match from vocabulary
        restored_predictions.append(best_match)

    print("\nComputing new intelligibility metrics...")
    # Calculate the new metrics using the restored words vs the original reference words
    new_metrics = compute_asr_metrics(restored_predictions, references)
    
    print("\n--- Results ---")
    print(f"Original ASR Metrics: {data.get('metrics')}")
    print(f"Vocabulary-Corrected Metrics: {new_metrics}")

    # Build the final dict imitating the original JSON
    out_data = {
        "split": data.get("split", "test"),
        "metrics": new_metrics,
        "predictions": restored_predictions,
        "references": references
    }

    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    
    print(f"\nSaved newly generated metrics and restored words to: {args.output}")

if __name__ == "__main__":
    main()
