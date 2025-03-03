"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import torch.distributed as dist  # 分散処理用

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Helper function to download a file from a given URL."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(fname)}",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split: str) -> None:
    """
    Downloads HellaSwag data into DATA_CACHE_DIR.
    分散処理（DDP）の場合、rank 0 のみがダウンロードを実施し、他プロセスは barrier で待機する。
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Rank {rank}/{world_size}] Starting download() for split '{split}'.")
        if rank == 0:
            if not os.path.exists(data_filename):
                print(f"[MASTER] Downloading {data_url} to {data_filename}...")
                download_file(data_url, data_filename)
                print(f"[MASTER] Download completed: {data_filename}")
            else:
                print(f"[MASTER] {data_filename} already exists (downloaded earlier).")
        else:
            print(f"[Rank {rank}] Waiting for master to download {data_filename}...")
        dist.barrier()  # 全プロセスがここで待機
        if rank != 0:
            print(f"[Rank {rank}] Download confirmed, resuming execution.")
    else:
        # 分散環境でない場合
        if not os.path.exists(data_filename):
            print(f"Downloading {data_url} to {data_filename}...")
            download_file(data_url, data_filename)
            print("Download completed.")
        else:
            print(f"{data_filename} already exists.")

def render_example(example: dict):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        # Prepend a space so that GPT-2 tokenizer treats it properly
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split: str):
    """
    Generator that yields examples from the HellaSwag JSONL file.
    ファイルは download() により、必要に応じてダウンロードされる。
    """
    download(split)
    filepath = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    print(f"Reading examples from {filepath} ...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type: str, device: str):
    """
    Evaluates a given GPT-2 model on HellaSwag.
    """
    torch.set_float32_matmul_precision('high')  # use tf32
    print(f"Loading model '{model_type}' on device {device}...")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # Optionally: model = torch.compile(model)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"[Example {num_total}] acc_norm: {num_correct_norm}/{num_total} = {num_correct_norm/num_total:.4f}")

        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"Predicted: {pred_norm}, Actual: {label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()

    # 分散環境の場合、各プロセスの情報をログ出力
    if dist.is_available() and dist.is_initialized():
        print(f"[Rank {dist.get_rank()}] LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}")
    evaluate(args.model_type, args.device)
