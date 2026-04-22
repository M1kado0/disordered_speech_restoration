from __future__ import annotations

from dataclasses import dataclass

from datasets import Audio, DatasetDict, load_dataset

from src.config import DatasetConfig


@dataclass(frozen=True)
class DatasetBundle:
    dataset: DatasetDict
    train_split: str
    validation_split: str
    test_split: str


def load_private_dataset(config: DatasetConfig, token: str | None = None) -> DatasetBundle:
    dataset = load_dataset(config.name, token=token)

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict from Hugging Face datasets")

    split_name = next(iter(dataset.keys()))
    columns = dataset[split_name].column_names
    if config.audio_column not in columns:
        raise KeyError(f"Audio column '{config.audio_column}' not found in dataset")
    if config.text_column not in columns:
        raise KeyError(f"Text column '{config.text_column}' not found in dataset")

    # Use decode=False to bypass torchcodec C++ bugs, allowing us to decode audio manually
    dataset = dataset.cast_column(config.audio_column, Audio(decode=False))

    # Create a validation split from the train split if it doesn't exist
    if config.validation_split not in dataset and config.train_split in dataset:
        print(f"Validation split '{config.validation_split}' not found. Creating a random 10% split from '{config.train_split}'.")
        splits = dataset[config.train_split].train_test_split(test_size=0.1, seed=42)
        dataset[config.train_split] = splits["train"]
        dataset[config.validation_split] = splits["test"]

    return DatasetBundle(
        dataset=dataset,
        train_split=config.train_split,
        validation_split=config.validation_split,
        test_split=config.test_split,
    )


def get_split(bundle: DatasetBundle, split_name: str):
    if split_name not in bundle.dataset:
        raise KeyError(f"Split '{split_name}' not found in dataset")
    return bundle.dataset[split_name]
