import pickle

import torch

from tqdm import trange


class Tokenizer:
    def __init__(self, cased=True, pad_id=0):
        self.cased = cased
        self.pad_id = pad_id

        self.merges = dict()
        self.vocab = {token: bytes([token]) for token in range(256)}
        self.vocab[self.pad_id] = b"<pad>"

    def __call__(self, list_of_texts, max_length=512):
        """Returns tensor with padded and maybe truncated sequences of tokens and padding mask (1s for real tokens, 0s for padding)"""
        list_of_token_sequences = [self.encode(text) for text in list_of_texts]
        max_sequence_length = max(len(x) for x in list_of_token_sequences)
        max_sequence_length = min(max_sequence_length, max_length)

        output_tensor = torch.full((len(list_of_token_sequences), max_sequence_length), self.pad_id)
        attention_mask = torch.zeros((len(list_of_token_sequences), max_sequence_length))

        for i, tokens in enumerate(list_of_token_sequences):
            output_tensor[i, : len(tokens)] = torch.tensor(tokens[:max_sequence_length])
            attention_mask[i, : len(tokens)] = 1

        return output_tensor, attention_mask

    def encode(self, text):
        """Tokenizes text and returns list of tokens"""

        if not self.cased:
            text = text.lower()

        tokens = []
        parts = [" " + w if i > 0 else w for i, w in enumerate(text.split(" "))]
        for p in parts:
            tokens.extend(self._tokenize(p))
        return tokens

    def decode(self, tokens):
        tokens_bytes = b"".join(self.vocab[token] for token in tokens)
        return tokens_bytes.decode("utf-8", errors="replace")

    def train(self, text, n_merges):
        if not self.cased:
            text = text.lower()

        byte_sequence = list(text.encode("utf-8"))
        for current_token in trange(256, 256 + n_merges):
            stats = self._get_stats(byte_sequence)

            try:
                pair_to_merge = max(stats, key=lambda pair: stats[pair] * self._is_pair_valid(pair))
            except ValueError:
                print(f"The input dataset is too small. Training terminated after {current_token - 256} merges.")
                break

            self.merges[pair_to_merge] = current_token
            byte_sequence = self._merge(byte_sequence, pair_to_merge)
            self.vocab[current_token] = self.vocab[pair_to_merge[0]] + self.vocab[pair_to_merge[1]]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(merges=self.merges, vocab=self.vocab, cased=self.cased), f)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for k, v in self.vocab.items():
                f.write(f"{k} |{v.decode('utf-8', errors='replace')}|\n")

    @classmethod
    def init_and_load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            tokenizer = cls(cased=data["cased"])
            tokenizer.merges = data["merges"]
            tokenizer.vocab = data["vocab"]
        return tokenizer

    def visualize(self, text):
        tokens = self.encode(text)
        parts = [self.vocab[token].decode("utf-8", errors="replace") for token in tokens]
        for i, p in enumerate(parts):
            if i % 2 == 0:
                print("\033[41m" + p + "\033[0m", end="")
            else:
                print("\033[44m" + p + "\033[0m", end="")

    def _tokenize(self, text):
        byte_sequence = list(text.encode("utf-8"))
        while len(byte_sequence) >= 2:
            stats = self._get_stats(byte_sequence)
            pair_to_merge = min(stats, key=lambda pair: self.merges.get(pair, float("inf")))
            if pair_to_merge not in self.merges:
                break
            byte_sequence = self._merge(byte_sequence, pair_to_merge)
        return byte_sequence

    def _get_stats(self, byte_sequence):
        stats = dict()
        for a, b in zip(byte_sequence, byte_sequence[1:]):
            stats[(a, b)] = stats.get((a, b), 0) + 1
        return stats

    def _merge(self, byte_sequence, pair):
        output = []
        i = 0
        while i < len(byte_sequence):
            if byte_sequence[i] == pair[0] and i + 1 < len(byte_sequence) and byte_sequence[i + 1] == pair[1]:
                output.append(self.merges[pair])
                i += 2
            else:
                output.append(byte_sequence[i])
                i += 1
        return output

    def _is_token_alphabetical(self, token_str):
        """Performs like str.isalpha() but also considers spaces and words with leading/trailing spaces to be alphabetical"""
        token_str = token_str.strip()
        if token_str == "":
            return True
        return token_str.isalpha()

    def _is_pair_valid(self, pair):
        a_str = self.vocab[pair[0]].decode("utf-8", errors="replace")
        b_str = self.vocab[pair[1]].decode("utf-8", errors="replace")

        #  this condition is used to stop words concatenation into one token (long spaces tokens are allowed)
        if a_str[-1] != " " and b_str[0] == " ":
            return False

        #  this condition is used to stop concatenation of alphabetical tokens with others
        if self._is_token_alphabetical(a_str) != self._is_token_alphabetical(b_str):
            return False

        #  this condition is used to stop concatenation of numeric tokens with others
        if a_str.isnumeric() != b_str.isnumeric():
            return False

        # to exclude any tokens like '\nExternal' or \n\nReferences\n\n'
        if a_str == "\n" or b_str == "\n":
            return False

        return True
