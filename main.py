from model.docllm import Decoder
from model.config import DocLLMConfig
import torch

config = DocLLMConfig.from_json("config.json")

batch_size = 2
seq_len = 50
vocab_size = config.vocab_size

x = torch.randint(0, vocab_size, (batch_size, seq_len))

model = Decoder(config=config)

y = model(x)

print(y.shape)