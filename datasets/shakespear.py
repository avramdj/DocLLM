import os
import requests
import tempfile

import sentencepiece as spm
import torch
import torch.utils.data


def download_text(url):
    response = requests.get(url)
    return response.text.lower()  # return the text in lowercase


class Tokenizer:
    def __init__(self, vocab_size=32000, model_type="bpe"):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.model_prefix = "bpe/m"
        os.makedirs("bpe", exist_ok=True)

    def train(self, text):
        if not os.path.isfile(f"{self.model_prefix}.model"):
            print("No checkpoint found. Training tokenizer...")
            tmptxt = tempfile.mktemp(suffix=".txt")
            with open(tmptxt, "w", encoding="utf-8") as f:
                f.write(text)
            special_tokens = ["<s>", "<pad>"]
            spm.SentencePieceTrainer.train(
                f"--input={tmptxt} --model_prefix={self.model_prefix} "
                f"--control_symbols={','.join(special_tokens)} "
                f"--vocab_size={self.vocab_size} --model_type={self.model_type}"
            )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{self.model_prefix}.model")
        return sp

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)
    
    def vocab_size(self):
        return self.sp.get_piece_size()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_len=512, overlap=128):
        self.max_len = max_len
        self.overlap = overlap
        self.tokenizer = tokenizer
        self.id_chunks = self.chunked_corpus(text)

    def chunked_corpus(self, text):
        tokenized_corpus = self.tokenizer.encode_as_ids(text)
        id_chunks = []
        for i in range(0, len(tokenized_corpus), self.max_len - self.overlap):
            id_chunks.append(tokenized_corpus[i : i + self.max_len])
        return id_chunks

    def __len__(self):
        return len(self.id_chunks)

    def __getitem__(self, index):
        return torch.as_tensor(self.id_chunks[index])

    def vocab_size(self):
        return self.tokenizer.get_piece_size()


def get_shakespear_dataload_and_tokenizer(
    batch_size: int, max_len: int, overlap: int
) -> tuple[torch.utils.data.DataLoader, Tokenizer]:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = download_text(url)
    tokenizer = Tokenizer()
    sp = tokenizer.train(text)
    dataset = Dataset(text, sp, max_len=max_len, overlap=overlap)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader, tokenizer
