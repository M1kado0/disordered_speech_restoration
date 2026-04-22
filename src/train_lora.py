from __future__ import annotations

import argparse

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.config import load_config
from src.data import get_split, load_private_dataset
from src.modeling import attach_lora, load_whisper_processor_and_model
from src.preprocess import build_preprocess_fn


def run_training(token: str | None = None):
    config = load_config()
    bundle = load_private_dataset(config.dataset, token=token)
    processor, model = load_whisper_processor_and_model(config)
    model = attach_lora(model, config)

    train_split = get_split(bundle, bundle.train_split)
    eval_split = get_split(bundle, bundle.validation_split)

    preprocess_fn = build_preprocess_fn(processor, config)
    train_split = train_split.map(preprocess_fn, remove_columns=train_split.column_names)
    eval_split = eval_split.map(preprocess_fn, remove_columns=eval_split.column_names)

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
        evaluation_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=config.model.max_label_length,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        tokenizer=processor.tokenizer,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
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
