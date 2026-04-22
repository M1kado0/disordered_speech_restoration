from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline import run_baseline


def compare_runs(baseline_result: dict, finetuned_result: dict) -> dict:
    return {
        "baseline": baseline_result["metrics"],
        "finetuned": finetuned_result["metrics"],
        "delta": {
            key: finetuned_result["metrics"][key] - baseline_result["metrics"][key]
            for key in baseline_result["metrics"]
        },
    }


def _load_metrics(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline Whisper with fine-tuned results")
    parser.add_argument("--baseline-metrics", default="outputs/baseline_metrics.json")
    parser.add_argument("--finetuned-metrics", default="outputs/finetuned_metrics.json")
    parser.add_argument("--output", default="outputs/comparison.json")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_metrics)
    baseline_result = _load_metrics(baseline_path) or run_baseline(token=args.hf_token)

    finetuned_path = Path(args.finetuned_metrics)
    finetuned_result = _load_metrics(finetuned_path)
    if finetuned_result is None:
        raise FileNotFoundError(
            "Fine-tuned metrics not found. Run training and save the metrics to "
            f"{finetuned_path} before comparison."
        )

    comparison = compare_runs(baseline_result, finetuned_result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
