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

@dataclass
class BARECDocumentInputExample:
    sentences: str  # Multiple sentences separated by newlines
    label: int  # 0-based readability level

def _split_text_into_blocks(
    text: str,
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    offset: int = 2
) -> List[str]:
    """
    Split text into blocks that fit within max_seq_length pixels.
    """
    lines = text.split("\n")
    blocks = []
    current_block = []
    current_width = 0
    
    max_pixels = processor.pixels_per_patch * max_seq_length - 2 * processor.pixels_per_patch
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_width = processor.get_text_width(line)
        
        if current_width + line_width >= max_pixels:
            # Start new block
            if current_block:
                blocks.append(" ".join(current_block))
            current_block = [line]
            current_width = line_width
        else:
            current_block.append(line)
            current_width += line_width + offset
    
    # Add remaining block
    if current_block:
        blocks.append(" ".join(current_block))
    
    return blocks

def convert_document_examples_to_image_features(
    examples: List[BARECDocumentInputExample],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    **kwargs
) -> List[Dict[str, Union[int, torch.Tensor, List[torch.Tensor]]]]:
    """
    Convert document examples to image features, splitting long texts into blocks.
    Returns a list of features where each feature contains multiple images for one document.
    """
    features = []
    for ex_index, example in enumerate(examples):
        blocks = _split_text_into_blocks(
            example.sentences, max_seq_length, processor
        )
        
        pixel_values_list = []
        attention_masks_list = []
        
        for block in blocks:
            encoding = processor(block)
            image = encoding.pixel_values
            num_patches = encoding.num_text_patches
            
            pixel_values = transforms(Image.fromarray(image))
            attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)
            
            pixel_values_list.append(pixel_values)
            attention_masks_list.append(torch.tensor(attention_mask, dtype=torch.long))
        
        features.append({
            "pixel_values": pixel_values_list,  # List of images for this document
            "attention_mask": attention_masks_list,  # List of attention masks
            "label": example.label,
            "num_blocks": len(blocks),
        })
        
        if ex_index < 5:
            logger.info("*** Document Example ***")
            logger.info(f"sentences (first 100 chars): {example.sentences[:100]}...")
            logger.info(f"num_blocks: {len(blocks)}")
            logger.info(f"label: {example.label}")
    
    return features

def _get_examples_to_features_fn(modality: Modality, is_document: bool = False):
    if modality == Modality.IMAGE:
        if is_document:
            return convert_document_examples_to_image_features
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
        Supports both sentence-level and document-level data.
        """

        def __init__(
            self,
            dataset_name,
            processor: Union[PyGameTextRenderer, PangoCairoTextRenderer, Callable],
            modality: Modality,
            max_seq_length: int,
            split: str = "train",
            transforms: Optional[Callable] = None,
            is_document: bool = False,  # New parameter to handle document vs sentence data
        ):
            logger.info(f"Creating features from HuggingFace dataset (no cache)")

            hf_dataset = load_dataset(dataset_name, split=split)
            
            if is_document:
                self.examples = [
                    BARECDocumentInputExample(
                        sentences=ex["Sentences"],
                        label=int(ex["Readability_Level_19"]) - 1
                    )
                    for ex in hf_dataset
                ]
            else:
                self.examples = [
                    BARECInputExample(
                        sentence=ex["Sentence"],
                        label=int(ex["Readability_Level_19"]) - 1
                    )
                    for ex in hf_dataset
                ]
            
            examples_to_features_fn = _get_examples_to_features_fn(modality, is_document)
            self.features = examples_to_features_fn(
                self.examples,
                max_seq_length=max_seq_length,
                processor=processor,
                transforms=transforms,
            )
            
            self.is_document = is_document

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]