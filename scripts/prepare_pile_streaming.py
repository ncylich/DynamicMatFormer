"""
Stream Pile data from HuggingFace, tokenize with gpt-neox-20b, and write .npy memmap files.

Usage:
    python scripts/prepare_pile_streaming.py \
        --output /mnt/data/noahcylich/data/pile \
        --num-tokens 200000000 \
        --val-tokens 10000000
"""

import argparse
import logging
import os
import sys

import numpy as np

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Output directory for .npy files")
    parser.add_argument("--num-tokens", type=int, default=200_000_000, help="Train tokens to collect")
    parser.add_argument("--val-tokens", type=int, default=10_000_000, help="Validation tokens to collect")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max-tokens-per-file", type=int, default=100_000_000, help="Max tokens per .npy file")
    parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "val"), exist_ok=True)

    # Load tokenizer
    from olmo.tokenizer import Tokenizer
    tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    log.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, eos_id={eos_id}")

    # Stream dataset
    from datasets import load_dataset
    log.info(f"Streaming from {args.dataset}...")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    total_needed = args.num_tokens + args.val_tokens

    # Collect tokens
    all_tokens = []
    collected = 0
    for i, example in enumerate(ds):
        text = example["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(ids)
        collected = len(all_tokens)

        if i % 10000 == 0 and i > 0:
            log.info(f"  Processed {i:,} docs, {collected:,}/{total_needed:,} tokens ({100*collected/total_needed:.1f}%)")

        if collected >= total_needed:
            break

    log.info(f"Collected {collected:,} tokens from {i+1:,} documents")

    # Split into train and val
    val_tokens = all_tokens[args.num_tokens:args.num_tokens + args.val_tokens]
    train_tokens = all_tokens[:args.num_tokens]

    # Write train .npy files
    log.info("Writing train files...")
    write_npy_files(train_tokens, os.path.join(args.output, "train"), "train", args.max_tokens_per_file)

    # Write val .npy files
    log.info("Writing val files...")
    write_npy_files(val_tokens, os.path.join(args.output, "val"), "val", args.max_tokens_per_file)

    log.info("Done!")


def write_npy_files(tokens, output_dir, prefix, max_tokens_per_file):
    """Write tokens to one or more .npy memmap files."""
    total = len(tokens)
    file_idx = 0
    offset = 0

    while offset < total:
        chunk_size = min(max_tokens_per_file, total - offset)
        path = os.path.join(output_dir, f"{prefix}_{file_idx:05d}.npy")

        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(chunk_size,))
        arr[:] = np.array(tokens[offset:offset + chunk_size], dtype=np.uint16)
        arr.flush()
        del arr

        log.info(f"  Wrote {path}: {chunk_size:,} tokens ({os.path.getsize(path):,} bytes)")
        offset += chunk_size
        file_idx += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
