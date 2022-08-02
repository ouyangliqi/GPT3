#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import math
import os
import random
import time
from itertools import chain

import colossalai
import numpy as np
import torch
import transformers
from accelerate.utils import set_seed
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import (get_current_device, get_dataloader,
                              load_checkpoint, save_checkpoint)
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from titans.utils import barrier_context
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (CONFIG_MAPPING, MODEL_MAPPING, AutoConfig,
                          GPT2Tokenizer, OPTForCausalLM,
                          PreTrainedTokenizerFast, SchedulerType,
                          default_data_collator, get_scheduler)
from transformers.utils.versions import require_version

from data.dataset_loader import get_data_loader
from utils import colo_memory_cap

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument("--mem_cap", type=int, default=0, help="use mem cap")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    seed_everything(123)

    colossalai.launch_from_torch(config='./colossalai_zero.py')

    logger = get_dist_logger()
    is_main_process = gpc.get_local_rank(ParallelMode.DATA) == 0

    if args.mem_cap > 0:
        colo_memory_cap(args.mem_cap)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Seed: {args.seed}")

    # Handle the repository creation
    with barrier_context():
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process:
        tensorboard_logdir = os.path.join("tb_logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        os.makedirs(tensorboard_logdir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tensorboard_logdir)


    logger.info("Prepare dataset")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/gpt_bpe.json")

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, vocab_size=len(tokenizer), n_ctx=1024, eos_token_id=1)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, vocab_size=len(tokenizer), n_ctx=1024, eos_token_id=1)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    logger.info("Model config is created")

    # build model
    shard_strategy = TensorShardStrategy()
    with ZeroInitContext(target_device=get_current_device(), shard_strategy=shard_strategy, shard_param=True):
        model = OPTForCausalLM(config=config)
        model.config.pad_token_id = 0
        model.config.eos_token_id = 1

        if args.resume_from_checkpoint is not None:
            load_checkpoint(os.path.join(args.resume_from_checkpoint, "model_weights.pt"), model)
            logger.info("Loaded model weights from checkpoint")

        # enable graident checkpointing
        model.gradient_checkpointing_enable()
        # model.resize_token_embeddings(len(tokenizer))

    logger.info(f'{model.__class__.__name__} is created')

    # Preprocessing the datasets.

    # DataLoaders creation:
    train_dataloader = get_data_loader(
        batch_size=args.per_device_train_batch_size,
        sequence_length=1024,
        device=get_current_device(),
        data_type=torch.long,
    )

    logger.info("Dataloaders are created")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = 100000
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare everything with our `accelerator`.
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(
        model, optimizer, None, train_dataloader, None, lr_scheduler
    )

    # if resume_from_checkpoint is not None, load optimizer and scheduler state_dict from the checkpoint.
    if args.resume_from_checkpoint is not None:
        optimizer_checkpoint = torch.load(
            os.path.join(args.resume_from_checkpoint, f"optimizer_weights_{gpc.get_local_rank(ParallelMode.DATA)}.pt"),
            map_location=None
        )

        engine.optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        lr_scheduler.load_state_dict(optimizer_checkpoint["lr_scheduler"])

        logger.info(f"Loaded optimizer state_dict from checkpoint")
        del optimizer_checkpoint

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = 100000

    # Train!
    total_batch_size = args.per_device_train_batch_size * gpc.get_world_size(ParallelMode.DATA)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    # Only show the progress bar once on each machine.

    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        progress = tqdm(train_dataloader, disable=not is_main_process)

        if args.resume_from_checkpoint is not None:
            starting_step = int(args.resume_from_checkpoint.split("_")[-1])
            logger.info("Startd from step {}".format(starting_step))

        for batch in progress:
            if epoch == 0 and args.resume_from_checkpoint is not None and completed_steps < starting_step:
                completed_steps += 1
                continue
            if epoch == 0 and args.resume_from_checkpoint is not None and completed_steps == starting_step:
                logger.info(f"Resumed from step {starting_step}")

            torch.cuda.empty_cache()

            batch = {k: v.cuda() for k, v in batch.items()}

            outputs = engine(**batch)
            loss = outputs['loss']
            engine.backward(loss)
            engine.step()
            lr_scheduler.step()
            engine.zero_grad()
            completed_steps += 1

            if is_main_process:
                tb_writer.add_scalar("train/loss", loss.item(), completed_steps)
                for _, lr in enumerate(lr_scheduler.get_lr()):
                    tb_writer.add_scalar(f"train/lr_{_}", lr, completed_steps)

            logger.info(f"step {completed_steps}: train_loss: {loss.item()}", ranks=[0])

            if completed_steps != 0 and completed_steps % 30 == 0:
                ckpt_dir = os.path.join(args.output_dir, f'weights_{completed_steps}')
                os.makedirs(ckpt_dir, exist_ok=True)

                ckpt_file_path = os.path.join(ckpt_dir, f'model_weights.pt')

                save_checkpoint(ckpt_file_path, completed_steps, engine.model)

                others = {'optimizer': engine.optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}
                torch.save(
                    others, os.path.join(ckpt_dir, f'optimizer_weights_{gpc.get_local_rank(ParallelMode.DATA)}.pt')
                )

        logger.info(f"epoch: {epoch} finished")

    if args.output_dir is not None:
        ckpt_dir = os.path.join(args.output_dir, f'weights_{completed_steps}')
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_file_path = os.path.join(ckpt_dir, f'model_weights.pt')

        save_checkpoint(ckpt_file_path, completed_steps, engine.model)


if __name__ == "__main__":
    main()
