import json
from dataclasses import dataclass

_k_BLOCK_SIZE = "block_size"
_k_N_EMBD = "n_embd"
_k_N_HEAD = "n_head"
_k_N_LAYER = "n_layer"
_k_VOCAB_SIZE = "vocab_size"


@dataclass
class ModelConfig:
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    vocab_size: int


def _validate_config(config):
    if _k_BLOCK_SIZE not in config:
        raise ValueError(f"{_k_BLOCK_SIZE} not defined in config.")
    if _k_N_EMBD not in config:
        raise ValueError(f"{_k_N_EMBD} not defined in config.")
    if _k_N_HEAD not in config:
        raise ValueError(f"{_k_N_HEAD} not defined in config.")
    if _k_N_LAYER not in config:
        raise ValueError(f"{_k_N_LAYER} not defined in config.")
    if _k_VOCAB_SIZE not in config:
        raise ValueError(f"{_k_VOCAB_SIZE} not defined in config.")


def load_from_file(path: str) -> ModelConfig:
    with open(path, "r") as f:
        config = json.load(f)
        _validate_config(config)
    return ModelConfig(**config)
