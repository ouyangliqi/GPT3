# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob
import itertools
import json
import math
import os
import random
import re
import time

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ChainDataset, Sampler, SequentialSampler
from torch.utils.data.dataloader import default_collate
from transformers import BertTokenizer, GPT2Tokenizer

from .SPECIAL_TOKENS import END_OF_TEXT_TOKEN, PAD_TOKEN

tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "./tokenizer/gpt_bpe.json"))


def pretrain_tokenize(tokenizer, text):
    token_dict = tokenizer.encode(text)
    return {
        "input_ids": token_dict.ids,
        "token_type_ids": token_dict.type_ids,
        "attention_mask": token_dict.attention_mask,
    }


def bert_tokenize(text):
    return tokenizer(text)


class JsonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, sequence_length, device, data_type, tokenizer, path, field
    ) -> None:
        super().__init__()
        self.max_len = sequence_length
        self.device = device
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.field = field
        self.path = path

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.epoch = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunk_id = self.rank
        total_chunks = self.world_size

        file_iter = open(self.path)

        # Map each element using the line_mapper
        mapped_itr = map(self.__getitem__, file_iter)

        if worker_info is not None:
            # Add multiworker functionality
            mapped_itr = itertools.islice(
                mapped_itr,
                worker_info.num_workers * chunk_id,
                worker_info.num_workers * (chunk_id + 1),
            )
        else:
            mapped_itr = itertools.islice(mapped_itr, chunk_id, None, total_chunks)
        return mapped_itr

    def __getitem__(self, line):
        input_sample = [
            cont for cont in json.loads(line)[self.field]
        ]
        input_sample = END_OF_TEXT_TOKEN.join([i for i in input_sample if len(i) > 1]) + END_OF_TEXT_TOKEN

        if self.tokenizer is not None:
            item = pretrain_tokenize(self.tokenizer, input_sample)

        if self.max_len is not None and len(item['input_ids']) > self.max_len:
            item['input_ids'] = item['input_ids'][: (self.max_len - 1)] + tokenizer.encode(END_OF_TEXT_TOKEN).ids
            item['attention_mask'] = item['attention_mask'][: (self.max_len - 1)] + [1]
            item['token_type_ids'] = item['token_type_ids'][: (self.max_len - 1)] + [0]

        assert len(item['input_ids']) <= self.max_len

        item["position_ids"] = torch.tensor(
            list(range(len(item["input_ids"]))),
            dtype=self.data_type,
            device=self.device,
        )
        item["input_ids"] = torch.tensor(
            item["input_ids"], dtype=self.data_type, device=self.device
        )
        item["token_type_ids"] = torch.tensor(
            item["token_type_ids"], dtype=self.data_type, device=self.device
        )
        item["attention_mask"] = torch.tensor(
            item["attention_mask"], dtype=self.data_type, device=self.device
        )

        item["labels"] = item["input_ids"].clone()
        # item["labels"] = torch.tensor(
        #     item["labels"], dtype=self.data_type, device=self.device
        # )

        assert (
            item["input_ids"].shape[0]
            == item["token_type_ids"].shape[0]
            == item["attention_mask"].shape[0]
            == item["position_ids"].shape[0]
            == item["labels"].shape[0]
        )
        return item


class ConversationDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, sequence_length, device, data_type, tokenizer, path, field=None
    ) -> None:
        super().__init__()
        self.max_len = sequence_length
        self.device = device
        self.tokenizer = tokenizer
        self.path = path
        self.field = field
        self.data_type = data_type

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.epoch = 0


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunk_id = self.rank
        total_chunks = self.world_size

        if "weibo" in self.path:
            file_iter = json.load(open(self.path))[0]
        else:
            file_iter = json.load(open(self.path))

        # Map each element using the line_mapper
        mapped_itr = map(self.__getitem__, file_iter)

        if worker_info is not None:
            # Add multiworker functionality
            mapped_itr = itertools.islice(
                mapped_itr,
                worker_info.num_workers * chunk_id,
                worker_info.num_workers * (chunk_id + 1),
            )
        else:
            mapped_itr = itertools.islice(mapped_itr, chunk_id, None, total_chunks)
        return mapped_itr

    def __getitem__(self, input_sample):
        if self.field is not None:
            input_sample = END_OF_TEXT_TOKEN.join(input_sample[self.field])
        else:
            input_sample = END_OF_TEXT_TOKEN.join(input_sample)

        if self.tokenizer is not None:
            item = pretrain_tokenize(self.tokenizer, input_sample)

        if self.max_len is not None and len(item['input_ids']) > self.max_len:
            item['input_ids'] = item['input_ids'][: (self.max_len - 1)] + tokenizer.encode(END_OF_TEXT_TOKEN)['input_ids']
            item['attention_mask'] = item['attention_mask'][: (self.max_len - 1)] + [1]
            item['token_type_ids'] = item['token_type_ids'][: (self.max_len - 1)] + [0]

        item["position_ids"] = torch.tensor(
            list(range(len(item["input_ids"]))),
            dtype=self.data_type,
            device=self.device,
        )
        item["input_ids"] = torch.tensor(
            item["input_ids"], dtype=self.data_type, device=self.device
        )
        item["token_type_ids"] = torch.tensor(
            item["token_type_ids"], dtype=self.data_type, device=self.device
        )
        item["attention_mask"] = torch.tensor(
            item["attention_mask"], dtype=self.data_type, device=self.device
        )
        item["labels"] = item["input_ids"].clone()

        assert (
            item["input_ids"].shape[0]
            == item["token_type_ids"].shape[0]
            == item["attention_mask"].shape[0]
            == item["position_ids"].shape[0]
            == item["labels"].shape[0]
        )
        return item


def collate(features):
    # batch
    input_ids = pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    position_ids = pad_sequence(
        [f["position_ids"] for f in features], batch_first=True, padding_value=0
    )
    token_type_ids = pad_sequence(
        [f["token_type_ids"] for f in features], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        # "position_ids": position_ids,
        # "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def read_file(args, sequence_length, device, data_type):
    datasets = []

    # common crawl
    commoncrawl = "/mnt/cfs/commoncrawl-202*-**-filter/"
    for dir in glob.glob(commoncrawl):
        file = dir + "under_all-data.txt"
        datasets.append(
            JsonDataset(sequence_length, device, data_type, tokenizer, file, "cont")
        )

    # LCCC
    LCCC = "/mnt/cfs/LCCC"
    for file in glob.glob(LCCC + "/**"):
        datasets.append(
            ConversationDataset(sequence_length, device, data_type, tokenizer, file)
        )

    # weibo
    weibo = "/mnt/cfs/weibo_comments/processed"
    for file in glob.glob(weibo + "/**"):
        datasets.append(
            ConversationDataset(
                sequence_length, device, data_type, tokenizer, file, "texts"
            )
        )

    dataset = torch.utils.data.ChainDataset(datasets)
    # logger.info("dataset size: {}".format(len(dataset)))

    # TODO(chloe): shuffle order across epochs
    return dataset


def get_data_loader(
    batch_size, sequence_length, device, data_type,
):
    # TODO(chloe): add cache
    # logger.info("loading data")
    start = time.time()
    train_dataset = read_file(None, sequence_length, device, data_type)
    step_elapse = time.time() - start
    # logger.info(f"LOADING DATA SPLEND {step_elapse}s")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate,
    )
    return train_loader

