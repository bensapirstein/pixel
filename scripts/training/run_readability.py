#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""

import argparse
import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from PIL import Image
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Modality,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    BARECDataset,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
    resize_model_embeddings,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    set_seed, default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


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
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()


def get_processor(model_args: argparse.Namespace, modality: Modality):
    if modality == Modality.TEXT:
        processor = AutoTokenizer.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            use_fast=True,
            add_prefix_space=True if model_args.model_name_or_path == "roberta-base" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
        )
    elif modality == Modality.IMAGE:
        renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
        processor = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
        )
    else:
        raise ValueError("Modality not supported.")
    return processor


def get_model_and_config(model_args: argparse.Namespace, num_labels: int, task_name: str):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        attention_probs_dropout_prob=model_args.dropout_prob,
        hidden_dropout_prob=model_args.dropout_prob,
        **config_kwargs,
    )

    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if config.model_type in ["bert", "roberta"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    elif config.model_type in ["vit_mae", "pixel"]:
        model = PIXELForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            pooling_mode=model_args.pooling_mode,
            add_layer_norm=model_args.pooler_add_layer_norm,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


def get_collator(
    training_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    is_regression: bool = False
):
    def image_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        if "label" in examples[0]:
            if is_regression:
                labels = torch.FloatTensor([example["label"] for example in examples])
            else:
                labels = torch.LongTensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "attention_mask": attention_mask, "labels": labels}
        return {"pixel_values": pixel_values, "attention_mask": attention_mask}

    if modality == Modality.IMAGE:
        collator = image_collate_fn
    elif modality == Modality.TEXT:
        collator = DataCollatorWithPadding(processor, pad_to_multiple_of=8) if training_args.fp16 else None
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return collator

def get_dataset(
    config: PretrainedConfig,
    data_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    split: str,
):
    kwargs = {}
    if modality == Modality.IMAGE:
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
    else:
        transforms = None
        kwargs.update({
            "sep_token_extra": bool(config.model_type in ["roberta"]),
            "cls_token": processor.cls_token,
            "sep_token": processor.sep_token,
            "pad_token": processor.convert_tokens_to_ids([processor.pad_token])[0]
        })

    return BARECDataset(
        data_args.dataset_name,
        processor=processor,
        modality=modality,
        max_seq_length=data_args.max_seq_length,
        split=split,
        transforms=transforms,
    )

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Labels
    num_labels = 19
    modality = Modality.IMAGE

    processor = get_processor(model_args, modality)
    if modality == Modality.IMAGE and hasattr(processor, "max_seq_length"):
        processor.max_seq_length = data_args.max_seq_length

    model = PIXELForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        pooling_mode=model_args.pooling_mode,
        hidden_dropout_prob=model_args.dropout_prob,
        attention_probs_dropout_prob=model_args.dropout_prob,
    )

    if modality == Modality.IMAGE:
        resize_model_embeddings(model, data_args.max_seq_length)

    data_collator = default_data_collator
    # data_collator = get_collator(training_args, processor, modality)

    # Load datasets
    train_dataset = get_dataset(model.config, data_args, processor, modality, "train") if training_args.do_train else None
    eval_dataset = get_dataset(model.config, data_args, processor, modality, "validation") if training_args.do_eval else None
    test_dataset = get_dataset(model.config, data_args, processor, modality, "test") if data_args.test_file is not None else None

    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
        if training_args.early_stopping
        else None,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and test_dataset:
        predictions = trainer.predict(test_dataset)
        trainer.log_metrics("test", predictions.metrics)
        trainer.save_metrics("test", predictions.metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
