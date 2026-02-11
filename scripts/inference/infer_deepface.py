#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

from deepface import DeepFace


def list_images(data_dir):
    images = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            images.append(os.path.join(root, f))
    return sorted(images)


def parse_args():
    parser = argparse.ArgumentParser("DeepFace batch inference")

    parser.add_argument("--data", type=str, required=True,
                        help="Directory with images")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL file")

    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Resume from index")

    return parser.parse_args()


def main():
    args = parse_args()

    images = list_images(args.data)

    if args.max_images is not None:
        images = images[:args.max_images]

    images = images[args.start_index:]

    print(f"[INFO] Images to process: {len(images)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as fout:
        for img_path in tqdm(images):
            try:
                result = DeepFace.analyze(
                    img_path=img_path,
                    actions=["age", "gender", "race"],
                    enforce_detection=False
                )

                record = {
                    "image_path": img_path,
                    "model": "DeepFace",
                    "raw_output": result,
                    "status": "ok"
                }

            except Exception as e:
                record = {
                    "image_path": img_path,
                    "model": "DeepFace",
                    "status": "error",
                    "error": str(e)
                }

            fout.write(json.dumps(record, default=str) + "\n")
            fout.flush()  # CRUCIAL on FRIDA

    print("[INFO] DeepFace inference completed.")


if __name__ == "__main__":
    main()
