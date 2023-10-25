# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
from dataclasses import asdict
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from llama_recipes.configs.datasets import DATASET_CONFIGS
from llama_recipes.configs import LoraConfig, LlamaAdapterConfig, PrefixConfig, TrainConfig
from llama_recipes.utils.dataset_utils import DATASET_PREPROC


def update_config(config: object, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, TrainConfig):
                print(f"Warning: unknown parameter {k}")
                        
                        
def generate_peft_config(train_config: TrainConfig, kwargs):
    config_cls_map = {
        "lora": (LoraConfig, LoraConfig),
        "llama_adapter": (LlamaAdapterConfig, AdaptionPromptConfig),
        "prefix": (PrefixConfig, PrefixTuningConfig)
    }
    
    assert train_config.peft_method in config_cls_map, f"Peft config not found: {train_config.peft_method}"
    config_cls, peft_config_cls = config_cls_map[train_config.peft_method]

    config = config_cls()
    
    update_config(config, **kwargs)
    params = asdict(config)
    return peft_config_cls(**params)


def generate_dataset_config(train_config: TrainConfig, kwargs):
    names = tuple(DATASET_PREPROC.keys())
    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    
    dataset_config = DATASET_CONFIGS[train_config.dataset]()
    update_config(dataset_config, **kwargs)
    
    return dataset_config