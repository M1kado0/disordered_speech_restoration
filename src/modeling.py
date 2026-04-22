from __future__ import annotations

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import AppConfig


def load_whisper_processor_and_model(config: AppConfig):
    processor = WhisperProcessor.from_pretrained(config.model.base_id)
    model = WhisperForConditionalGeneration.from_pretrained(config.model.base_id)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.model.language,
        task=config.model.task,
    )
    model.config.suppress_tokens = []
    return processor, model


def attach_lora(model, config: AppConfig):
    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    return get_peft_model(model, peft_config)
