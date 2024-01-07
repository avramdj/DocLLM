import json
from dataclasses import dataclass


class Config:
    """Base class for configuration objects."""

    @classmethod
    def from_json(cls, json_path):
        """Load a configuration from a JSON file."""
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

    def to_json(self, json_path):
        """Save a configuration to a JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f)


@dataclass
class DocLLMConfig(Config):
    """Configuration for DocLLM."""

    vocab_size: int
    num_layers: int
    num_heads: int
    hidden_size: int
    max_seq_len: int
    ffn_multiplier: int
    dropout: float
    batch_size: int

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def ffn_expanded_size(self):
        return self.hidden_size * self.ffn_multiplier
