import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data import WikipediaTokenizedDataset
from modules.transformer import Transformer, TransformerConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    return parser.parse_args()


def main(args):
    train_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "train"))
    test_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "test"))

    print(len(train_dataset))
    print(len(test_dataset))


if __name__ == "__main__":
    main(parse_args())
