from __future__ import annotations

import argparse
import torch
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor

from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.config import load_config, AppConfig
from src.data import get_split, load_private_dataset
from src.metrics import compute_asr_metrics
from src.modeling import attach_lora, load_whisper_processor_and_model
from src.preprocess import build_preprocess_fn, normalize_text


class AudioTextDataset(Dataset):
    def __init__(self, hf_dataset, processor: WhisperProcessor, config: AppConfig):
        self.data = hf_dataset
        self.processor = processor
        self.config = config
        self._preprocess_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._preprocess_fn is None:
            # Build lazily so each DataLoader worker initializes its own copy
            # after forking, avoiding shared-state issues with the processor.
            self._preprocess_fn = build_preprocess_fn(self.processor, self.config)
        return self._preprocess_fn(self.data[idx])


def run_training(token: str | None = None):
    config = load_config()
    bundle = load_private_dataset(config.dataset, token=token)
    processor, model = load_whisper_processor_and_model(config)
    model = attach_lora(model, config)

    train_split = get_split(bundle, bundle.train_split)
    eval_split = get_split(bundle, bundle.validation_split)

    train_dataset = AudioTextDataset(train_split, processor, config)
    eval_dataset = AudioTextDataset(eval_split, processor, config)

    pad_token_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = pad_token_id
        preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        preds = [normalize_text(p) for p in preds]
        refs = [normalize_text(r) for r in refs]
        return compute_asr_metrics(preds, refs)

    model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        num_train_epochs=config.training.num_train_epochs,
        weight_decay=config.training.weight_decay,
        eval_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=config.model.max_label_length,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(config.training.output_dir)
    processor.save_pretrained(config.training.output_dir)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper medium with LoRA")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for private dataset access")
    args = parser.parse_args()

    run_training(token=args.hf_token)


if __name__ == "__main__":
    main()
