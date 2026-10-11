#!/usr/bin/env python3

# script for preprocessing datasets for CLIP training
# WARNING: this script is extremely inefficient for large datasets due to not batching tokenization
# input format: plain directory of images and txt file with same basename as image files
# output format: jsonl file with {"file_name": image_path, "text": caption}, images are hardlinked/copied/reflinked to output directory
#
# processing includes:
# - checking if captions fit CLIP tokenization requirements (75 tokens max)
# - chopping captions that are too long into multiple entries (and don't break comma-separated "tags")
# - skipping images without captions

import os
import shutil
from multiprocessing import Pool, cpu_count

import click
import reflink
import tqdm
from jsonlines import jsonlines
from transformers import CLIPProcessor  # for trial tokenization

from picurate.common.images import walk_directory_relative

default_clip_model = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
max_clip_tokens = 75

file_op_modes = ["copy", "hardlink", "reflink", "symlink"]


def output_relative_image_path(image_path: str) -> str:
    """Return the output path relative to the dataset root.

    Preserve the original relative path underneath ``images/``. Flattening
    paths with ``replace(os.path.sep, "_")`` collides for inputs such as
    ``a/b.jpg`` and ``a_b.jpg`` and silently overwrites training samples.
    """
    return os.path.join("images", image_path)


def process_image_caption(args):
    image_path, caption, processor = args
    # Tokenize caption
    tokens = processor.tokenizer(caption, truncation=False, padding=False, return_tensors="pt")["input_ids"][0]

    dataset_image_path = output_relative_image_path(image_path)

    if len(tokens) <= max_clip_tokens:
        return [{"file_name": dataset_image_path, "text": caption}]
    else:
        # Split caption into smaller parts
        parts = []
        current_part = []
        current_length = 0

        for tags in caption.split(", "):
            tag_tokens = processor.tokenizer(tags, truncation=False, padding=False, return_tensors="pt")["input_ids"][0]
            tag_length = len(tag_tokens)

            if current_length + tag_length + 1 <= max_clip_tokens:
                current_part.append(tags)
                current_length += tag_length + 1  # +1 for space and comma
            else:
                parts.append(", ".join(current_part))
                current_part = [tags]
                current_length = tag_length

        if current_part:
            parts.append(", ".join(current_part))

        return [{"file_name": dataset_image_path, "text": part} for part in parts]


@click.command()
@click.argument("input_dataset_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--clip-model", type=str, default=default_clip_model, help="CLIP model to use for tokenization", show_default=True)
@click.option(
    "--mode", type=click.Choice(file_op_modes), default="copy", help="File operation mode for handling images", show_default=True
)
def main(input_dataset_dir, output_dir, clip_model, mode):
    metadata_output_path = os.path.join(output_dir, "metadata.jsonl")
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    processor = CLIPProcessor.from_pretrained(clip_model)
    args_list = []
    for image_rel_path in walk_directory_relative(input_dataset_dir):
        image_base, file_ext = os.path.splitext(image_rel_path)
        if file_ext.lower() == ".txt":
            continue
        caption_path = os.path.join(input_dataset_dir, image_base + ".txt")

        if os.path.isfile(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            args_list.append((image_rel_path, caption, processor))

    results = []
    print(f"Processing {len(args_list)} image captions...")
    with Pool(cpu_count()) as pool:
        for res in tqdm.tqdm(pool.imap_unordered(process_image_caption, args_list), total=len(args_list)):
            results.extend(res)

    with jsonlines.open(metadata_output_path, mode="w") as writer:
        for item in results:
            writer.write(item)

    print(f"Processed {len(results)} caption entries, output written to {metadata_output_path}")
    print(f"Images should be placed in {image_output_dir} using '{mode}' mode.")
    for a in tqdm.tqdm(args_list):
        image_rel_path = a[0]
        src_path = os.path.join(input_dataset_dir, image_rel_path)
        dest_image_path = os.path.join(output_dir, output_relative_image_path(image_rel_path))
        os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
        if mode == "copy":
            shutil.copy2(src_path, dest_image_path)
        elif mode == "hardlink":
            os.link(src_path, dest_image_path)
        elif mode == "reflink":
            reflink.reflink(src_path, dest_image_path)
        elif mode == "symlink":
            os.symlink(src_path, dest_image_path)


if __name__ == "__main__":
    main()
