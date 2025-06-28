"""
Script used to prerender BAREC from https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2025-doc
Processes the dataset doc-by-doc and uploads the rendered examples in chunks to HuggingFace.
Examples are stored and compressed in parquet files.
Relies on a modified version of the datasets library installed through git submodule.
"""

import argparse
import logging
import sys

from datasets import load_dataset
from PIL import Image
from pixel import PyGameTextRenderer, PangoCairoTextRenderer, log_example_while_rendering, push_rendered_chunk_to_hub

logger = logging.getLogger(__name__)

def prerender_barec(
    args,
    dataset_names: str,
    split: str,
    output_format: str = "minimal",  # "minimal" or "full"
):
    # Load renderer
    text_renderer = PangoCairoTextRenderer.from_pretrained(
        args.renderer_name_or_path, use_auth_token=args.auth_token
    )


    # Prepare stats and data
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    def get_empty_data():
        if output_format == "minimal":
            return {"pixel_values": [], "num_patches": []}
        else:
            # Add all features from the dataset, plus pixel_values and num_patches
            features = barec.features.keys() if hasattr(barec, "features") else []
            data = {k: [] for k in features}
            data["pixel_values"] = []
            data["num_patches"] = []
            return data

    data = get_empty_data()

    max_pixels = text_renderer.pixels_per_patch * text_renderer.max_seq_length - 2 * text_renderer.pixels_per_patch
    target_seq_length = max_pixels
    idx = 0
    offset = 2

    for dataset_name in dataset_names:

        # Load dataset
        barec = load_dataset(dataset_name, split=split, streaming=True)

        for doc_id, doc in enumerate(barec):
            # Handle both doc and sent structure
            if "Sentences" in doc:
                lines = doc["Sentences"].split("\n")
            elif "Sentence" in doc:
                lines = [doc["Sentence"]]
            else:
                logger.warning(f"Unknown structure in doc: {doc}")
                continue

            logger.info(f"{doc_id}: {doc.get('Document', doc.get('Sentence', ''))}, {target_seq_length=}px, {idx=}, {dataset_stats['total_num_words']=}")

            width = 0
            block = []
            for line in lines:
                line = line.strip()
                if line:
                    dataset_stats["total_num_words"] += len(line.split(" "))

                    line_width = text_renderer.get_text_width(line)
                    if width + line_width >= target_seq_length:
                        idx += 1
                        sequence = " ".join(block)
                        encoding = text_renderer(text=sequence)

                        data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
                        data["num_patches"].append(encoding.num_text_patches)
                        if output_format == "full":
                            for k in doc:
                                data[k].append(doc[k])

                        if idx % args.chunk_size == 0:
                            log_example_while_rendering(idx, sequence, encoding.num_text_patches)
                            dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
                            data = get_empty_data()

                        width = line_width
                        block = [line]
                    else:
                        block.append(line)
                        width += line_width + offset

            if len(block) > 0:
                idx += 1
                sequence = " ".join(block)
                encoding = text_renderer(text=sequence)
                data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
                data["num_patches"].append(encoding.num_text_patches)
                if output_format == "full":
                    for k in doc:
                        data[k].append(doc[k])

                if idx % args.chunk_size == 0:
                    log_example_while_rendering(idx, sequence, encoding.num_text_patches)
                    dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
                    data = get_empty_data()

    # Upload remaining data if any
    if len(data["pixel_values"]) > 0:
        dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)

    logger.info(f"Total num words in {dataset_name} ({split}): {dataset_stats['total_num_words']}")

def main(args: argparse.Namespace):
    # Support multiple datasets and splits
    dataset_names = args.dataset_names.split(",")
    splits = args.splits.split(",")
    for split in splits:
        args.split = split
        prerender_barec(
            args,
            dataset_names=dataset_names,
            split=split.strip(),
            output_format=args.output_format,
        )

if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer_name_or_path",
        type=str,
        help="Path or Huggingface identifier of the text renderer",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Push data to hub in chunks of N lines",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=-1,
        help="Only look at the first N non-empty lines",
    )
    parser.add_argument("--repo_id", type=str, help="Name of dataset to upload")
    parser.add_argument("--dataset_names", type=str, default="CAMeL-Lab/BAREC-Shared-Task-2025-doc,CAMeL-Lab/BAREC-Shared-Task-2025-sent", help="Comma-separated list of dataset names (e.g. CAMeL-Lab/BAREC-Shared-Task-2025-doc,CAMeL-Lab/BAREC-Shared-Task-2025-sent)")
    parser.add_argument("--splits", type=str, default="train,validation,test", help="Comma-separated list of splits to process")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["minimal", "full"],
        default="minimal",
        help="Output format: minimal (pixel_values, num_patches) or full (all features)",
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Huggingface auth token with write access to the repo id",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
