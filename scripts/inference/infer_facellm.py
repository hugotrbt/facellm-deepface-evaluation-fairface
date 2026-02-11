#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path
from tqdm import tqdm


def list_images(data_dir):
    images = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            images.append(os.path.join(root, f))
    return sorted(images)


def parse_args():
    parser = argparse.ArgumentParser("FaceLLM batch inference (HPC wrapper)")

    parser.add_argument("--data", type=str, required=True,
                        help="Directory with images")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to prompt text file")
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

    with open(args.prompt_file, "r") as f:
        prompt = f.read().strip()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as fout:
        for img_path in tqdm(images):

            cmd = [
                "python3",
                "inference.py",
                "--path_image", img_path,
                "--prompt", prompt
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                raw_output = result.stdout.strip()

                record = {
                    "image_path": img_path,
                    "model": "Facellm",
                    "raw_output": raw_output,
                    "status": "ok"
                }

            except subprocess.CalledProcessError as e:
                record = {
                    "image_path": img_path,
                    "model": "Facellm",
                    "status": "error",
                    "stderr": e.stderr
                }

            fout.write(json.dumps(record) + "\n")
            fout.flush()  # CRUCIAL on FRIDA

    print("[INFO] Facellm inference completed.")


if __name__ == "__main__":
    main()
