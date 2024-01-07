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
    def __init__(self, text, vocab_size=2000, model_type="bpe"):
        self._vocab_size = vocab_size
        self.model_type = model_type
        self.model_prefix = "bpe/m"
        os.makedirs("bpe", exist_ok=True)
        if not os.path.isfile(f"{self.model_prefix}.model"):
            print("No checkpoint found. Training tokenizer...")
            tmptxt = tempfile.mktemp(suffix=".txt")
            with open(tmptxt, "w", encoding="utf-8") as f:
                f.write(text)
            special_tokens = ["<bos>", "<eos>", "<pad>"]
            spm.SentencePieceTrainer.train(
                f"--input={tmptxt} --model_prefix={self.model_prefix} "
                f"--control_symbols={','.join(special_tokens)} "
                f"--vocab_size={self._vocab_size} --model_type={self.model_type} "
                "--minloglevel=5"
            )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{self.model_prefix}.model")
        self._sp = sp
        self.box_token = "<box>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_idx = sp.piece_to_id("<bos>")
        self.eos_idx = sp.piece_to_id("<eos>")
        self.pad_idx = sp.piece_to_id("<pad>")

    def encode(self, text):
        if not hasattr(self, "_sp"):
            raise Exception("Tokenizer not trained yet.")
        return self._sp.EncodeAsIds(text)

    def decode(self, ids):
        if not hasattr(self, "_sp"):
            raise Exception("Tokenizer not trained yet.")
        return self._sp.DecodeIds(ids)

    def vocab_size(self):
        if not hasattr(self, "_sp"):
            raise Exception("Tokenizer not trained yet.")
        return self._sp.piece_size()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer: Tokenizer, max_length=512, overlap=128):
        self.max_length = max_length
        self.overlap = overlap
        self.tokenizer = tokenizer
        self.id_chunks = self.chunked_corpus(text)

    def chunked_corpus(self, text):
        tokenized_corpus = self.tokenizer.encode(text)
        id_chunks = []
        for i in range(0, len(tokenized_corpus), self.max_length - self.overlap):
            chunk = tokenized_corpus[i : i + self.max_length]
            if len(chunk) < self.max_length:
                chunk += [self.tokenizer.pad_idx] * (self.max_length - len(chunk))
            id_chunks.append(chunk)
        return id_chunks

    def __len__(self):
        return len(self.id_chunks)

    def __getitem__(self, index):
        return torch.as_tensor(self.id_chunks[index])


def get_shakespear_dataload_and_tokenizer(
    vocab_size: int, batch_size: int, max_length: int, overlap: int
) -> tuple[torch.utils.data.DataLoader, Tokenizer]:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    corpus = download_text(url)
    tokenizer = Tokenizer(corpus, vocab_size=vocab_size)
    dataset = Dataset(corpus, tokenizer, max_length=max_length, overlap=overlap)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader, tokenizer
