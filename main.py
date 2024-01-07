import os
import torch
import argparse
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from model.docllm import TransformerDecoder
from model.config import DocLLMConfig
from datasets.shakespear import get_shakespear_dataload_and_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--use-checkpoint", action="store_true")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device
print(f"Using {device}")


checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

config = DocLLMConfig.from_json("config.json")
dataloader, tokenizer = get_shakespear_dataload_and_tokenizer(
    batch_size=config.batch_size,
    max_length=config.max_seq_len,
    overlap=config.max_seq_len // 4,
)

config.vocab_size = tokenizer.vocab_size()
if args.use_checkpoint:
    checkpoint_path = os.path.join(
        checkpoint_dir, sorted(os.listdir(checkpoint_dir))[-1]
    )
    model = torch.load(checkpoint_path)
else:
    model = TransformerDecoder(config=config)

epochs = 1000
optimizer = AdamW(model.parameters(), lr=1e-3)

warmup_steps = 100
total_steps = len(dataloader) * epochs
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(
        (step + 1) / warmup_steps, 1.0 - (step + 1) / total_steps
    ),
)

model.to(device)

for epoch in range(epochs):
    model.train()

    with torch.inference_mode():
        model.eval()
        context = "romeo:"
        context_ids = tokenizer.encode(context)
        context_ids = torch.as_tensor(context_ids).to(device)
        output = (
            model.generate(context_ids, max_length=100, eos_id=tokenizer.eos_idx)
            .cpu()
            .numpy()
            .tolist()
        )
        print(tokenizer.decode(output))

    pbar = tqdm(dataloader)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch[:, :-1])
        loss = cross_entropy(
            output.reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(
            f"Loss: {loss.item():.3f} LR: {scheduler.get_last_lr()[0]:.4e}"
        )

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        },
        checkpoint_path,
    )
    previous_checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_{epoch - 1}.pt"
    )
    if os.path.exists(previous_checkpoint_path):
        os.remove(previous_checkpoint_path)
