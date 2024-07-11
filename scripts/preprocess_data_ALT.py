import argparse
import os
import random
import sys

import pandas as pd
import torch

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_to_save", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--padding_value", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--articles_for_test", type=int, default=1_000)
    parser.add_argument("--articles_for_train", type=int, default=1_000_000)
    return parser.parse_args()


def preprocess(data, idxs, tokenizer, padding_value, context_length, chunk_size, saving_dir):
    os.makedirs(saving_dir, exist_ok=True)

    i = 0
    chunk = dict()
    for idx in tqdm(idxs):
        text = data.iloc[idx]["text"]  # type: ignore
        tokens = tokenizer.encode(text)

        for j in range(0, len(tokens) - 1, context_length):
            t = tokens[j : j + context_length + 1]

            if len(t) < 2:
                continue

            pad_mask = torch.zeros(context_length).bool()
            pad_mask[:len(t)] = True

            t.extend([padding_value] * (context_length + 1 - len(t)))
            t = torch.tensor(t)

            chunk[i] = dict(xy=t, pad_mask=pad_mask)

            i += 1
            if len(chunk) == chunk_size:
                torch.save(chunk, os.path.join(saving_dir, f"chunk_{i-len(chunk)}_{i}.pt"))
                chunk = dict()

    if len(chunk) > 0:
        torch.save(chunk, os.path.join(saving_dir, f"chunk_{i-len(chunk)}_{i}.pt"))
def get_data():
    return pd.read_parquet("/data/d2/m.koltyugin/TEST_TASK_LLM/data/a.parquet")

def main(args):
    # data = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train")
    data = get_data()
    tokenizer = Tokenizer.init_and_load(args.tokenizer_path)
    # idxs = random.sample(range(len(data)), len(data))  # type: ignore
    idxs = list(range(len(data)))

    preprocess(
        data,
        idxs[: args.articles_for_test],
        tokenizer,
        args.padding_value,
        args.context_length,
        args.chunk_size,
        os.path.join(args.dir_to_save, "test"),
    )

    preprocess(
        data,
        idxs[args.articles_for_test : args.articles_for_test + args.articles_for_train],
        tokenizer,
        args.padding_value,
        args.context_length,
        args.chunk_size,
        os.path.join(args.dir_to_save, "train"),
    )


if __name__ == "__main__":
    main(parse_args())
