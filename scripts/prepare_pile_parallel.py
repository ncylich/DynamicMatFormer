"""
Parallel Pile data preparation: downloads multiple shards concurrently,
tokenizes in parallel workers, writes .npy memmap files.

Usage:
    python scripts/prepare_pile_parallel.py \
        --output /mnt/data/noahcylich/data/pile-700M \
        --num-tokens 750000000 \
        --val-tokens 20000000 \
        --workers 8
"""

import argparse
import logging
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

log = logging.getLogger(__name__)


def tokenize_shard(args):
    """Tokenize a portion of the dataset. Each worker skips to its start offset."""
    shard_idx, num_shards, tokenizer_id, target_tokens, docs_per_shard = args
    from datasets import load_dataset
    from olmo.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    ds = load_dataset(
        "EleutherAI/the_pile_deduplicated",
        split="train",
        streaming=True,
    )

    skip_docs = shard_idx * docs_per_shard
    tokens = []
    doc_count = 0
    for i, example in enumerate(ds):
        if i < skip_docs:
            continue
        text = example["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        tokens.extend(ids)
        doc_count += 1
        if len(tokens) >= target_tokens:
            break
        if doc_count % 5000 == 0 and doc_count > 0:
            print(f"  [worker {shard_idx}] {doc_count:,} docs, {len(tokens):,}/{target_tokens:,} tokens", flush=True)

    return tokens[:target_tokens]


def write_npy(tokens, path):
    """Write tokens to a single .npy memmap file."""
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(tokens),))
    arr[:] = np.array(tokens, dtype=np.uint16)
    arr.flush()
    del arr
    print(f"  Wrote {path}: {len(tokens):,} tokens ({os.path.getsize(path):,} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-tokens", type=int, default=750_000_000)
    parser.add_argument("--val-tokens", type=int, default=20_000_000)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "val"), exist_ok=True)

    total_needed = args.num_tokens + args.val_tokens
    per_worker = total_needed // args.workers + 1

    print(f"Downloading {total_needed:,} tokens using {args.workers} parallel workers")
    print(f"Each worker targets ~{per_worker:,} tokens")

    # Estimate docs per shard: ~1400 tokens/doc average from prior runs
    docs_per_shard = per_worker // 1400 + 1000

    # Launch parallel tokenization
    worker_args = [
        (i, args.workers, args.tokenizer, per_worker, docs_per_shard)
        for i in range(args.workers)
    ]

    all_tokens = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(tokenize_shard, wa): i for i, wa in enumerate(worker_args)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                shard_tokens = future.result()
                print(f"Shard {idx} done: {len(shard_tokens):,} tokens")
                all_tokens.extend(shard_tokens)
            except Exception as e:
                print(f"Shard {idx} failed: {e}")

    print(f"\nTotal collected: {len(all_tokens):,} tokens")

    # Trim to exact sizes
    if len(all_tokens) < args.num_tokens + args.val_tokens:
        print(f"WARNING: only got {len(all_tokens):,} tokens, needed {args.num_tokens + args.val_tokens:,}")
        val_count = min(args.val_tokens, len(all_tokens) // 10)
        train_count = len(all_tokens) - val_count
    else:
        train_count = args.num_tokens
        val_count = args.val_tokens

    train_tokens = all_tokens[:train_count]
    val_tokens = all_tokens[train_count:train_count + val_count]

    # Write train files (split into ~200M token chunks)
    print(f"\nWriting {len(train_tokens):,} train tokens...")
    chunk_size = 200_000_000
    for i in range(0, len(train_tokens), chunk_size):
        chunk = train_tokens[i:i + chunk_size]
        write_npy(chunk, os.path.join(args.output, "train", f"train_{i // chunk_size:05d}.npy"))

    # Write val file
    print(f"Writing {len(val_tokens):,} val tokens...")
    write_npy(val_tokens, os.path.join(args.output, "val", "val_00000.npy"))

    print("\nDone!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO)
    main()
