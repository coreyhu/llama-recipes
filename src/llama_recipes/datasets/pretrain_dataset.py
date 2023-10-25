# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from transformers import LlamaTokenizer
from llama_recipes.datasets.utils import Concatenator
from llama_recipes.configs.datasets import PretrainDatasetConfig

def get_pretrain_dataset(dataset_config: PretrainDatasetConfig, tokenizer: LlamaTokenizer, split: str):
    dataset = datasets.load_dataset(dataset_config.dataset, dataset_config.subset, split=split)
    dataset = dataset.remove_columns(["timestamp", "url"])

    remove_features = list(dataset.features).remove(dataset_config.text_column)
    
    return dataset.map(
        lambda sample: tokenizer(sample["text"] + tokenizer.eos_token),
        batched=True,
        remove_columns=remove_features,
    ).map(Concatenator(), batched=True)
