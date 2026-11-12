"""Create a tiny protocol demonstration. These are not benchmark predictions."""
import argparse
import json
from pathlib import Path

import numpy as np


def create_example(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Example directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    categories = [{"id": 0, "name": "target"}]
    samples, split_meta, probability_splits = [], {}, {}
    for split in ("val", "test"):
        images, entries = [], []
        for image_id in (1, 2):
            sample_id = f"synthetic:{split}-{image_id}"
            images.append({"id": image_id, "file_name": f"{split}-{image_id}.png",
                           "height": 4, "width": 4})
            samples.append({"sample_id": sample_id, "split": split,
                            "image_id": image_id, "height": 4, "width": 4})
            prob = np.zeros((4, 4), dtype=np.float32)
            if image_id == 1:
                prob[1, 1] = .8 if split == "val" else .4
                prob[3, 3] = .2 if split == "val" else .25
            elif split == "test":
                prob[3, 3] = .7
            filename = f"{split}-{image_id}.npy"
            np.save(directory / filename, prob, allow_pickle=False)
            entries.append({"sample_id": sample_id, "path": filename})
        gt = {"images": images, "categories": categories,
              "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                               "bbox": [1, 1, 1, 1], "iscrowd": 0}]}
        gt_name = f"gt_{split}.json"
        (directory / gt_name).write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
        split_meta[split] = {"gt_json": gt_name}
        probability_splits[split] = entries
    manifest = {"version": 1, "categories": categories, "samples": samples,
                "mask_box_xy_offset": [0, 0], "splits": split_meta}
    probability_index = {"prediction_set_id": "synthetic-demo-not-paper-results",
                         "probability_space": "unpadded", "splits": probability_splits}
    for filename, value in (("manifest.json", manifest), ("probabilities.json", probability_index)):
        (directory / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_example(args.output)
    print("Synthetic fixture created; no paper scores are reproduced.")


if __name__ == "__main__":
    main()
