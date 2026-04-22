from __future__ import annotations

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import AppConfig

import torch

def load_whisper_processor_and_model(config: AppConfig):
    processor = WhisperProcessor.from_pretrained(config.model.base_id)
    
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    # Load model using configurations
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model.base_id,
        torch_dtype=dtype_mapping.get(config.model.torch_dtype, torch.float32),
        attn_implementation=config.model.attn_implementation
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = config.model.language
    model.generation_config.task = config.model.task
    return processor, model


def attach_lora(model, config: AppConfig):
    # Omitting task_type avoids PeftModelForSeq2SeqLM which leaks input_ids=None
    # into Whisper's **kwargs, causing "multiple values for input_ids" at the decoder.
    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_config)

    # Whisper's conv encoder breaks the gradient graph under gradient checkpointing.
    # This hook forces the conv1 output to require gradients, keeping backprop alive
    # through the encoder. Required by all official PEFT+Whisper examples.
    peft_model.base_model.model.model.encoder.conv1.register_forward_hook(
        lambda _module, _input, output: output.requires_grad_(True)
    )

    return peft_model
