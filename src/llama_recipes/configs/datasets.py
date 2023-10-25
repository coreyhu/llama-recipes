# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class DatasetConfig:
    dataset: str
    train_split: str
    test_split: str

@dataclass
class FinetuneDatasetConfig(DatasetConfig):
    pass

@dataclass
class PretrainDatasetConfig(DatasetConfig):
    subset: str = None
    text_column: str = None

@dataclass
class MergedDatasetConfig(DatasetConfig):
    configs: list[DatasetConfig] = None
    dataset: str = "merged_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class SamsumDataset(FinetuneDatasetConfig):
    dataset: str =  "samsum"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class GrammarDataset(FinetuneDatasetConfig):
    dataset: str = "grammar"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class AlpacaDataset(FinetuneDatasetConfig):
    dataset: str = "alpaca"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class CustomDataset(FinetuneDatasetConfig):
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class C4PretrainDataset(PretrainDatasetConfig):
    dataset: str =  "c4"
    subset: str = "en.noblocklist"
    text_column: str = "text"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


DATASET_CONFIGS = {
    "samsum": SamsumDataset,
    "grammar": GrammarDataset,
    "alpaca": AlpacaDataset,
    "c4": C4PretrainDataset,
    "custom_dataset": CustomDataset
}