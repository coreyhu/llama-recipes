# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from transformers import LlamaTokenizer
from llama_recipes.configs.datasets import PretrainDatasetConfig

def get_pretrain_dataset(
    dataset_config: PretrainDatasetConfig, 
    tokenizer: LlamaTokenizer, 
    split: str, 
    streaming: bool = False
):
    dataset = datasets.load_dataset(
        dataset_config.dataset, 
        dataset_config.subset, 
        split=split, 
        streaming=streaming
    )
    preprocess_col = "__fmt_text__"

    def format_text(sample) -> dict[str, str]:
        return {preprocess_col: sample[dataset_config.text_column] + tokenizer.eos_token}
    
    dataset = dataset.map(format_text, remove_columns=list(dataset.features))
    return dataset.map(
        lambda sample: tokenizer(sample[preprocess_col]),
        batched=True,
        remove_columns=[preprocess_col]
    )
