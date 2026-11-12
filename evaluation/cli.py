
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import geometry
from .protocol import MASK_GRID, PROTOCOL, better, convert, match_all

def json_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def relative_asset(root, value):

    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("Asset paths must be relative to the input JSON")
    root = Path(root).resolve()
    target = (root / value).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Asset path escapes the input JSON directory")
    return target

def load_manifest(path):
    path = Path(path)
    manifest = read_json(path)
    geometry.category_id(manifest)
    samples = manifest.get("samples", [])
    seen_ids, seen_image_ids = set(), set()
    for sample in samples:
        if sample["split"] not in ("val", "test"):
            raise ValueError("The portable evaluation manifest contains val/test only")
        if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
            raise ValueError("Each sample needs a nonempty sample_id")
        if sample["sample_id"] in seen_ids:
            raise ValueError("sample_id must be unique across val/test")
        seen_ids.add(sample["sample_id"])
        pair = (sample["split"], sample["image_id"])
        if pair in seen_image_ids:
            raise ValueError("Duplicate image_id within one split")
        seen_image_ids.add(pair)
        if any(not isinstance(sample[k], int) or sample[k] <= 0
               for k in ("height", "width")):
            raise ValueError("Original image height/width must be positive integers")
    offset = manifest.get("mask_box_xy_offset", [0, 0])
    if (not isinstance(offset, list) or len(offset) != 2
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in offset)):
        raise ValueError("Declare a finite, fixed [dx,dy] annotation offset")
    gt_paths = {split: relative_asset(path.parent, manifest["splits"][split]["gt_json"])
                for split in ("val", "test")}

    identity = {"manifest_sha256": file_hash(path),
                "GT_sha256": {split: file_hash(p) for split, p in gt_paths.items()}}
    return manifest, gt_paths, identity

def load_gt(manifest, path, split):
    gt = read_json(path)
    images, _ = geometry._validate_gt(gt)
    if gt["categories"] != manifest["categories"]:
        raise ValueError("Manifest and GT category definitions differ")
    samples = geometry.samples_for_split(manifest, split)
    if {s["image_id"] for s in samples} != set(images):
        raise ValueError("Manifest must cover every GT image, including empty-GT images")
    for sample in samples:
        image = images[sample["image_id"]]
        if (sample["height"], sample["width"]) != (image["height"], image["width"]):
            raise ValueError("Manifest and GT original image dimensions differ")
    return gt

def load_maps(path, manifest, split):
    path = Path(path)
    index = read_json(path)
    if index.get("probability_space") != "unpadded":
        raise ValueError("Declare probability_space=unpadded; undo letterbox padding first")
    prediction_id = index.get("prediction_set_id")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise ValueError("Declare a prediction_set_id shared by val and test")
    checkpoint = index.get("checkpoint_sha256")
    if checkpoint is not None and (not isinstance(checkpoint, str)
            or len(checkpoint) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint)):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256 digest when given")
    entries = index["splits"][split]
    expected = {s["sample_id"] for s in geometry.samples_for_split(manifest, split)}
    ids = [entry["sample_id"] for entry in entries]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Probability index must exactly cover the requested split")
    maps, hashes = {}, {}
    for entry in entries:
        sample_id = entry["sample_id"]
        asset = relative_asset(path.parent, entry["path"])
        if asset.suffix.lower() != ".npy":
            raise ValueError("Use unquantized numeric .npy probability maps")
        digest = file_hash(asset)
        if entry.get("sha256", digest) != digest:
            raise ValueError("Probability-map SHA256 differs from its index")
        maps[sample_id] = np.load(asset, allow_pickle=False)
        hashes[sample_id] = digest
    return maps, {"prediction_set_id": prediction_id,
                  "checkpoint_sha256": checkpoint, "map_SHA256": hashes}

def select_validation(manifest, gt, maps):

    selected, candidates = None, []
    for threshold in MASK_GRID:
        boxes = convert(manifest, "val", maps, threshold)
        metrics, _ = match_all(gt, boxes)
        row = {"mask_threshold": threshold, "metrics": metrics}
        candidates.append(row)
        if better(row, selected):
            selected = row
    return selected, candidates

def freeze_selection(selected, identity, map_identity):
    frozen = {"protocol": PROTOCOL["version"], "protocol_sha256": json_hash(PROTOCOL),
              "selected_on": "val", "mask_threshold": selected["mask_threshold"],
              "mask_grid": list(MASK_GRID), "confidence_threshold": None,
              "confidence_policy": "all mean-CC instances; no confidence/topK/area cut",
              "VAL_metrics": selected["metrics"], **identity, **map_identity}
    frozen["selection_sha256"] = json_hash(frozen)
    return frozen

def verify_selection(frozen, identity):
    unsigned = {k: v for k, v in frozen.items() if k != "selection_sha256"}
    if frozen.get("selection_sha256") != json_hash(unsigned):
        raise ValueError("Frozen validation selection digest mismatch")
    if (frozen.get("selected_on") != "val"
            or frozen.get("protocol") != PROTOCOL["version"]
            or frozen.get("protocol_sha256") != json_hash(PROTOCOL)
            or frozen.get("mask_grid") != list(MASK_GRID)
            or frozen.get("mask_threshold") not in MASK_GRID
            or frozen.get("confidence_threshold") is not None):
        raise ValueError("Selection is not a frozen WP2-v2 validation working point")
    if any(frozen.get(k) != v for k, v in identity.items()):
        raise ValueError("Manifest/GT identity differs from validation selection")

def write_outputs(directory, values):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory must be new or empty; records are not overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        with (directory / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("describe", help="Print the frozen numerical protocol")
    for name in ("select", "test"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--manifest", required=True, type=Path)
        cmd.add_argument("--probabilities", required=True, type=Path)
        cmd.add_argument("--output", required=True, type=Path)
        if name == "test":
            cmd.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "describe":
        print(json.dumps(PROTOCOL, indent=2))
        return 0
    manifest, gt_paths, identity = load_manifest(args.manifest)
    if args.command == "select":
        gt = load_gt(manifest, gt_paths["val"], "val")
        maps, map_identity = load_maps(args.probabilities, manifest, "val")
        selected, candidates = select_validation(manifest, gt, maps)
        frozen = freeze_selection(selected, identity, map_identity)
        write_outputs(args.output, {"selection.json": frozen,
                                   "validation_candidates.json": candidates,
                                   "protocol.json": PROTOCOL})
        print(json.dumps({"selected_on": "val", "mask_threshold": frozen["mask_threshold"],
                          "VAL_metrics": frozen["VAL_metrics"]}))
    else:
        frozen = read_json(args.selection)
        verify_selection(frozen, identity)
        maps, map_identity = load_maps(args.probabilities, manifest, "test")
        for key in ("prediction_set_id", "checkpoint_sha256"):
            if frozen.get(key) != map_identity[key]:
                raise ValueError("Val/test predictions declare different model identities")
        gt = load_gt(manifest, gt_paths["test"], "test")
        boxes = convert(manifest, "test", maps, frozen["mask_threshold"])
        metrics, details = match_all(gt, boxes)
        result = {"protocol": PROTOCOL["version"], "split": "test",
                  "mask_threshold": frozen["mask_threshold"],
                  "selection_sha256": frozen["selection_sha256"],
                  "metrics": metrics, "Pd_equals_R": True,
                  **identity, **map_identity}
        write_outputs(args.output, {"metrics.json": result, "boxes.json": boxes,
                                   "matching_details.json": details,
                                   "protocol.json": PROTOCOL})
        print(json.dumps(metrics))
    return 0
