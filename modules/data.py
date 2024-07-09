import os

import torch


class WikipediaTokenizedDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.chunk_names = os.listdir(self.data_dir)
        self.size = max([int(name.split(".pt")[0].split("_")[1:][1]) for name in self.chunk_names])
        self.cache = dict()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx not in self.cache:
            chunk_name = self._get_chunk_name(idx)
            chunk_path = os.path.join(self.data_dir, chunk_name)
            self.cache = torch.load(chunk_path)

        d = self.cache[idx]

        return dict(
            x=d["xy"][:-1],
            y=d["xy"][1:],
            pad_mask=d["pad_mask"],
        )

    def _get_chunk_name(self, idx):
        for name in self.chunk_names:
            start, end = name.split(".pt")[0].split("_")[1:]
            if int(start) <= idx < int(end):
                return f"chunk_{start}_{end}.pt"
        raise IndexError
