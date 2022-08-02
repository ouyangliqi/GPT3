import glob
import json
import mmap
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from colossalai.logging import get_dist_logger
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .DistributedSampler import get_dataloader
from .SPECIAL_TOKENS import END_OF_TEXT_TOKEN, EOD_ID, PAD_ID, PAD_TOKEN

tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "./tokenizer/gpt_bpe.json"))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_dist_logger()


def pretrain_tokenize(tokenizer, text):
    token_dict = tokenizer.encode(text)
    return {
        "input_ids": token_dict.ids,
        "token_type_ids": token_dict.type_ids,
        "attention_mask": token_dict.attention_mask,
    }


def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


class IdentitySplitter(object):
     @staticmethod
     def tokenize(*text):
         return text


class Encoder(object):
    def __init__(self, field, length, sentence_splitter):
        self.field = field
        self.length = length
        self.sentence_splitter = sentence_splitter
        self.tokenizer = Tokenizer.from_file(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "./tokenizer/gpt_bpe.json")
        )
        self.splitter = IdentitySplitter()

    def initializer(self):
        # Use Encoder class as a container for global data
        pass

    def encode(self, line):
        # end with <eod>
        line = json.loads(line)[self.field]
        if len(line) > 20000:
            return None, 0
        if len(line) < 10:
            return None, 0
        data = line.strip()
        doc_ids = self.tokenizer.encode(data)
        doc_ids.append(EOD_ID)
        # group by length and return list of encoded ids

        return doc_ids, len(sum())


class CommonCrawlDataset(torch.utils.data.Dataset):
    """
    For loading JSONL data and encoding on-the-fly with a given tokenizer.

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "text": "text goes here, with newlines",
        "meta": {"pile_set_name": "name of corpus", "other": "metadata"}
    }
    ```

    Note that only the "text" key is used.
    """

    def __init__(self, path: str, sequence_length, field, data_type, tokenizer, recache=False):
        self.path = path
        self.max_len = sequence_length
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.field = field

        self.threadlocal = threading.local()
        # TODO(susan): Fix this fairseq reference. _build_index fails otherwise.
        self.cache = Path(f"{path}.fairseq.idx.npy")
        self.length_cache = Path(f"{path}.fairseq.length.npy")

        if self.cache.exists() and not recache:
            self.offsets, self.length = np.load(self.cache), np.load(self.length_cache)
        else:
            self.offsets, self.length = self._build_index(path)
            np.save(self.cache, self.offsets)
            np.save(self.length_cache, self.length)
        # print(f'n offsets: {len(self.offsets)}')

        # self.length = [self.offsets[i + 1] - self.offsets[i] for i in range(0, len(self.offsets))]

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

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        f = self._get_mmap()
        f.seek(self.offsets[idx])
        item = f.readline().decode("utf-8")
        item = json.loads(item)[self.field]
        # TODO(chloe): change into encoder
        item = "".join(item) + END_OF_TEXT_TOKEN
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

    def __len__(self):
        return len(self.offsets)

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
            length.append(len("".join(json.loads(line.decode("utf-8").strip())[self.field])))
        return offsets, length

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


def collate(features):
    # batch
    input_ids = pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )

    attention_mask = pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_data_loader(
    batch_size,
    sequence_length,
    device,
    data_type,
):
    # TODO(chloe): add cache
    logger.info("loading data")
    start = time.time()
    datasets = []
    dataset_length = []

    # commoncrawl = "/mnt/cfs/commoncrawl-202*-**-filter/under_all-data.txt"
    # for dir in glob.glob(commoncrawl):
    #     datasets.append(CommonCrawlDataset(
    #         path=dir, sequence_length=sequence_length, field="cont", data_type=data_type, tokenizer=tokenizer, recache=True
    #     ))
    #     dataset_length.extend(datasets[-1].length)

    zh_wiki = "/mnt/cfs/zh_wiki/zh_wiki_all.txt"
    datasets.append(CommonCrawlDataset(
        path=zh_wiki, sequence_length=sequence_length, field="text", data_type=data_type, tokenizer=tokenizer, recache=False
    ))
    dataset_length.extend(datasets[-1].length)


    # weibo = "/mnt/cfs/weibo_comments/weibocomments_all.txt"
    # datasets.append(
    #     ConversationDataset(
    #         path=weibo, sequence_length=sequence_length, data_type=data_type, tokenizer=tokenizer, recache=False
    #     ))
    # dataset_length.extend(datasets[-1].length)


    # lccc = "/mnt/cfs/LCCC/lccc_all.txt"
    # datasets.append(
    #     ConversationDataset(
    #         path=lccc, sequence_length=sequence_length, data_type=data_type, tokenizer=tokenizer, recache=False
    #     ))
    # dataset_length.extend(datasets[-1].length)


    step_elapse = time.time() - start
    logger.info(f"LOADING DATA SPLEND {step_elapse}s")

    train_dataset = torch.utils.data.ConcatDataset(datasets)
    batches = np.array(dataset_length).argsort()
    train_loader = get_dataloader(
        train_dataset,
        batches,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collate,
    )
    return train_loader

if __name__ == "__main__":
    dataloader = get_data_loader(
        batch_size=32, sequence_length=128, device="cuda", data_type=torch.long)
