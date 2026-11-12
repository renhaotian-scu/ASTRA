"""Unchanged geometric/GT helpers extracted from the recorded evaluation code.

See SOURCE_PROVENANCE.json for exact source and per-function fingerprints.
"""
from collections import defaultdict
import math

SPLITS = ("train", "val", "test")


def _np():
    import numpy as np
    return np


def category_id(gt):
    categories = gt.get("categories", [])
    if len(categories) != 1:
        raise ValueError("This protocol requires one target category; read its id from GT")
    return categories[0]["id"]


def _validate_gt(gt):
    cat = category_id(gt)
    images = {image["id"]: image for image in gt["images"]}
    if len(images) != len(gt["images"]):
        raise ValueError("Duplicate image_id inside one COCO split")
    annotations = defaultdict(list)
    seen_annotations = set()
    for annotation in gt["annotations"]:
        if annotation["id"] in seen_annotations:
            raise ValueError("Duplicate annotation id inside one COCO split")
        seen_annotations.add(annotation["id"])
        if annotation["image_id"] not in images or annotation["category_id"] != cat:
            raise ValueError("Unknown image/category id in GT annotation")
        box = annotation["bbox"]
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box) or min(box[2:]) <= 0:
            raise ValueError("Invalid GT bounding box")
        if not annotation.get("iscrowd", 0):
            annotations[annotation["image_id"]].append(annotation)
    return images, annotations


def samples_for_split(manifest, split):
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")
    return [s for s in manifest["samples"] if s["split"] == split]


def restore_probability(probability, original_size, pad_meta=None):
    from PIL import Image
    np = _np()
    prob = np.asarray(probability, dtype="float32")
    # Accept one singleton channel/batch only; never squeeze a one-pixel image axis.
    while prob.ndim > 2 and prob.shape[0] == 1:
        prob = prob[0]
    if prob.ndim == 3 and prob.shape[-1] == 1:
        prob = prob[..., 0]
    if prob.ndim != 2 or not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("Expected a finite [H,W] probability map in [0,1], not logits")
    h, w = map(int, original_size)
    if min(h, w) <= 0:
        raise ValueError("Original image dimensions must be positive")
    if pad_meta is not None:
        if (pad_meta["original_height"], pad_meta["original_width"]) != (h, w):
            raise ValueError("Padding metadata belongs to another image shape")
        th, tw = pad_meta["input_height"], pad_meta["input_width"]
        if prob.shape != (th, tw):
            prob = np.asarray(Image.fromarray(prob).resize((tw, th), Image.Resampling.BILINEAR))
        top, left, rh, rw = (pad_meta[k] for k in ("pad_top", "pad_left", "resized_height", "resized_width"))
        if min(top, left) < 0 or min(rh, rw) < 1 or top + rh > th or left + rw > tw:
            raise ValueError("Invalid letterbox crop")
        prob = prob[top:top + rh, left:left + rw]
    if prob.shape != (h, w):
        prob = np.asarray(Image.fromarray(prob).resize((w, h), Image.Resampling.BILINEAR))
    return np.array(prob, dtype="float32", copy=True)


def _box_iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
