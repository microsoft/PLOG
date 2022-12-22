#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The Microsoft. All rights reserved.
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
Evaluate the fidelity of LogicNLG dataset with TAPEX.
Adapted from script: https://github.com/huggingface/transformers/blob/main/examples/research_projects/tapex/run_tabfact_with_tapex.py
"""

import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import json

import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    TapexTokenizer,
    BartForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="tab_fact", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    affix: Optional[str] = field(default='test', metadata={"help": "The ID of an experiment."})
    split_name: Optional[str] = field(default="test",
                                      metadata={"help": "val or test, the evaluation dataset"})
    test_name: Optional[str] = field(default=None,
                                     metadata={"help": "A .json file containing the model outputs to evaluate."})
    data_dir: Optional[str] = field(default='data/logicnlg',
                                    metadata={"help": "The data dir for storing original table files"})






@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.




    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed before initializing model.
    set_seed(training_args.seed)

    def preprocess_logicnlg_function(examples):
        # Tokenize the texts
        def _construct_table(table_id):
            _pd_table = pd.read_csv(os.path.join(data_args.data_dir, 'all_csv/' + table_id), '#')
            _pd_table = _pd_table.astype(str)
            return _pd_table
        questions = examples["sent"]
        tables = list(map(_construct_table, examples["table_id"]))

        result = tokenizer(tables, questions,
                           padding="max_length", max_length=1024, truncation=True)
        result["label"] = examples["label"]
        return result

    def prepare_single_data(args, output_dir, split, test_name):

        with open(test_name) as fsent, \
                open(os.path.join(output_dir, args.affix, f'for_{split}.json'), 'w') as fout:
            data = json.load(fsent)
            for table_id, entry in data.items():
                for e in entry:
                    d_new = {}
                    d_new["table_id"] = table_id
                    d_new["sent"] = e
                    d_new['label'] = 1
                    fout.write(json.dumps(d_new))
                    fout.write('\n')

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.

        data_files = {}
        if training_args.do_predict and data_args.test_name is not None:
            if not os.path.exists(os.path.join(training_args.output_dir, data_args.affix)):
                os.makedirs(os.path.join(training_args.output_dir, data_args.affix))
            prepare_single_data(data_args, training_args.output_dir, data_args.split_name, data_args.test_name)
            test_file_name = os.path.join(training_args.output_dir, data_args.affix, f'for_{data_args.split_name}.json')
            data_files[data_args.split_name] = test_file_name
        else:
           raise NotImplementedError
        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")


        # Loading a dataset from local json files
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Labels
    label_list = ['Refused', 'Entailed']
    # label_list = raw_datasets["train"].features["label"].names
    # num_labels = len(label_list)
    num_labels = 2

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # load tapex tokenizer
    tokenizer = TapexTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True
    )
    model = BartForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    model.config.label2id = {"Refused": 0, "Entailed": 1}
    model.config.id2label = {0: "Refused", 1: "Entailed"}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)



    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_logicnlg_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        raise NotImplementedError

    if training_args.do_eval:
       raise NotImplementedError

    if training_args.do_predict or data_args.test_name is not None:
        predict_dataset = raw_datasets[data_args.split_name]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )


    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions[0], axis=1)
        #
        output_predict_file = os.path.join(training_args.output_dir, f"{data_args.affix}_predict_results.txt")
        # if trainer.is_world_process_zero():
        num_all = 0
        num_correct = 0
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict Results *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                num_all += 1
                item = label_list[item]
                if item == 'Entailed':
                    num_correct += 1
                writer.write(f"{index}\t{item}\n")
        print("acc: ", num_correct * 1.0 / num_all)

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    #
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()