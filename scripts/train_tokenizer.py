import argparse
import os
import sys

from random import sample

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.tokenizer import Tokenizer

TEST_EXAMPLE = """
What is a piece of text?
A text is a passage of words that conveys a set of meanings to the person who is reading it.
It’s a body of written work, in various forms and structures, that can be words, phrases and sentences that piece together a passage of written work.
To put it as simply as possible, it is a group of words. But it can come in many different forms.
A text can be written materials, such as books, magazines, newspapers, or online content. 
But it can also be other things, those that we may not associate with standard text!
Text could be movies, scripts, paintings, songs, political cartoons, advertisements and maps. 
If we can look at something with words and sentences, explore it, find layers of meaning in it, and draw information and conclusions from it, you’re looking at a text.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_merges", type=int, required=True)
    parser.add_argument("--n_articles", type=int, required=True, help="Max is around 6M...the more, the better, but takes linearly more time")
    parser.add_argument("--path_to_save", type=str, required=True)
    parser.add_argument("--path_to_save_vocab", type=str)
    parser.add_argument("--cased", action="store_true", default=False)
    return parser.parse_args()


def get_tokenizer_train_data(n_articles):
    data = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train")
    idxs = sample(range(len(data)), n_articles)  # type: ignore
    list_of_articles = [data[idx]["text"] for idx in idxs]  # type: ignore
    train_data = " ".join(list_of_articles)  # type: ignore
    return train_data


def main(args):
    # get data
    train_data = get_tokenizer_train_data(args.n_articles)

    # train
    tokenizer = Tokenizer(cased=args.cased)
    tokenizer.train(train_data, args.n_merges)
    tokenizer.save(args.path_to_save)

    # save vocab
    if args.path_to_save_vocab is not None:
        tokenizer.save_vocab(args.path_to_save_vocab)

    # test
    if args.cased:
        assert tokenizer.decode(tokenizer.encode(TEST_EXAMPLE)) == TEST_EXAMPLE
        print("Encode -> decode test passed!")
    else:
        assert tokenizer.decode(tokenizer.encode(TEST_EXAMPLE)) == TEST_EXAMPLE.lower()
        print("Encode -> decode test passed! (Case is not preserved)")

    tokenizer.visualize(TEST_EXAMPLE)


if __name__ == "__main__":
    main(parse_args())
