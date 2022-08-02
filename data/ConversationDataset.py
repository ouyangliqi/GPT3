import json
import mmap
import threading
from pathlib import Path

import numpy as np
import torch
from colossalai.logging import get_dist_logger

from .SPECIAL_TOKENS import END_OF_TEXT_TOKEN, EOD_ID, PAD_ID, PAD_TOKEN

logger = get_dist_logger()


def pretrain_tokenize(tokenizer, text):
    token_dict = tokenizer.encode(text)
    return {
        "input_ids": token_dict.ids,
        "token_type_ids": token_dict.type_ids,
        "attention_mask": token_dict.attention_mask,
    }



class ConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self, path, sequence_length, data_type, tokenizer, recache=False
    ) -> None:
        super().__init__()
        self.path = path
        self.max_len = sequence_length
        self.tokenizer = tokenizer
        self.data_type = data_type

        self.pad_id = PAD_ID
        self.threadlocal = threading.local()

        self.cache = Path(f"{path}.fairseq.idx.npy")
        self.length_cache = Path(f"{path}.fairseq.length.npy")
        if self.cache.exists() and not recache:
            self.offsets, self.length = np.load(self.cache), np.load(self.length_cache)
        else:
            self.offsets, self.length = self._build_index(path)
            np.save(self.cache, self.offsets)
            np.save(self.length_cache, self.length)

    def __len__(self):
        return len(self.offsets)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            f = open(self.path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.threadlocal.handles = [f, mm]
            if (
                self.path.endswith(".gz")
                or self.path.endswith(".bz")
                or self.path.endswith(".bz2")
            ):
                raise NotImplementedError(
                    "Compressed files are not supported because .seek() would require "
                    "rereading the entire file, making performance too slow."
                )
        return self.threadlocal.handles[-1]

    def _build_index(self, path: str):
        """Build index of start positions of each line."""
        logger.info(f"Building index for file: {path}")
        f = self._get_mmap()
        f.seek(0)
        offsets = []
        length = []
        cur = 0
        while True:
            line = f.readline()
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
            length.append(len(" ".join(json.loads(line.decode("utf-8").strip())["texts"])))
        return offsets, length

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        f = self._get_mmap()
        f.seek(self.offsets[idx])
        item = f.readline().decode("utf-8")
        item = json.loads(item)

        item = END_OF_TEXT_TOKEN.join(item) + END_OF_TEXT_TOKEN

        if self.tokenizer is not None:
            item = pretrain_tokenize(self.tokenizer, item)

        if self.max_len is not None and len(item['input_ids']) > self.max_len:
            item['input_ids'] = item['input_ids'][: (self.max_len - 1)] + [EOD_ID]
            item['attention_mask'] = item['attention_mask'][: (self.max_len - 1)] + [0]

        item["input_ids"] = torch.tensor(
            item["input_ids"], dtype=self.data_type
        )
        item["attention_mask"] = torch.tensor(
            item["attention_mask"], dtype=self.data_type
        )
        item["labels"] = item["input_ids"].clone()

        assert (
            item["input_ids"].shape[0]
            == item["attention_mask"].shape[0]
            == item["labels"].shape[0]
        )
        return item

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)
