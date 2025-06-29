import os
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
from datasets import load_dataset

import torch
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, is_torch_available

from ...utils import Modality, get_attention_mask
from ..rendering import PyGameTextRenderer, PangoCairoTextRenderer

logger = logging.getLogger(__name__)

@dataclass
class BARECInputExample:
    sentence: str
    label: int  # 0-based readability level

def _get_examples_to_features_fn(modality: Modality):
    if modality == Modality.IMAGE:
        return convert_examples_to_image_features
    if modality == Modality.TEXT:
        return convert_examples_to_text_features
    else:
        raise ValueError("Modality not supported.")

def convert_examples_to_image_features(
    examples: List[BARECInputExample],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    **kwargs
) -> List[Dict[str, Union[int, torch.Tensor]]]:
    features = []
    for ex_index, example in enumerate(examples):
        encoding = processor(example.sentence)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        features.append({
            "pixel_values": pixel_values,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": example.label,
        })

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {example.sentence}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"label: {example.label}")

    return features

def convert_examples_to_text_features(
    examples: List[BARECInputExample],
    max_seq_length: int,
    processor,
    **kwargs
) -> List[Dict[str, Union[int, torch.Tensor]]]:
    features = []
    for ex_index, example in enumerate(examples):
        encoding = processor(
            example.sentence,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        feature = {k: v.squeeze(0) for k, v in encoding.items()}
        feature["label"] = example.label
        features.append(feature)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {example.sentence}")
            logger.info(f"input_ids: {feature['input_ids']}")
            logger.info(f"attention_mask: {feature['attention_mask']}")
            logger.info(f"label: {example.label}")

    return features

if is_torch_available():
    from torch.utils.data import Dataset

    class BARECDataset(Dataset):
        """
        PyTorch Dataset for BAREC readability classification, wrapping a HuggingFace datasets.Dataset object.
        """

        def __init__(
            self,
            dataset_name,  # HuggingFace datasets.Dataset object
            processor: Union[PyGameTextRenderer, PangoCairoTextRenderer, Callable],
            modality: Modality,
            max_seq_length: int,
            split: str = "train",
            transforms: Optional[Callable] = None,
        ):
            logger.info(f"Creating features from HuggingFace dataset (no cache)")

            hf_dataset = load_dataset(dataset_name, split=split)

            self.examples = [
                BARECInputExample(
                    sentence=ex["Sentence"],
                    label=int(ex["Readability_Level_19"]) - 1
                )
                for ex in hf_dataset
            ]
            examples_to_features_fn = _get_examples_to_features_fn(modality)
            self.features = examples_to_features_fn(
                self.examples,
                max_seq_length=max_seq_length,
                processor=processor,
                transforms=transforms,
            )

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]