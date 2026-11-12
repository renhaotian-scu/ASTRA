
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import unittest
import tempfile

from evaluation import geometry, protocol
from evaluation.cli import (
    freeze_selection, load_gt, load_manifest, load_maps, main,
    select_validation, verify_selection,
)
from evaluation.make_synthetic_example import create_example

def gt(boxes, n_images=1):
    return {"images": [{"id": i + 1, "width": 64, "height": 64}
                       for i in range(n_images)],
            "categories": [{"id": 0, "name": "target"}],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": 0,
                             "bbox": box} for i, box in enumerate(boxes)]}

def det(box, score, image_id=1):
    return {"image_id": image_id, "category_id": 0, "bbox": box, "score": score}

class TestSegmentationEvaluation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_function_bodies_match_recorded_source_fingerprints(self):
        provenance = json.loads((Path(protocol.__file__).parent / "SOURCE_PROVENANCE.json").read_text())
        for filename, module in (("geometry.py", geometry), ("protocol.py", protocol)):
            for origin in provenance["extracted_functions"][filename]:
                source = inspect.getsource(getattr(module, origin["function"]))
                digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                assert digest == origin["extracted_function_LF_SHA256"]

    def test_eight_connectivity_tight_box_and_foreground_mean(self):
        prob = np.zeros((3, 3), dtype=np.float32)
        prob[0, 0], prob[1, 1] = .8, .6
        boxes = protocol.component_boxes(prob, 1, (3, 3), .5, 0)
        assert len(boxes) == 1
        assert boxes[0]["bbox"] == [0., 0., 2., 2.]
        assert boxes[0]["component_area_pixels"] == 2
        self.assertAlmostEqual(boxes[0]["score"], .7, places=6)

        self.assertAlmostEqual(boxes[0]["component_max_probability"], .8, places=6)

    def test_single_pixel_no_area_filter_inclusive_threshold_and_frozen_offset(self):
        prob = np.zeros((2, 3), dtype=np.float32)
        prob[1, 2] = .5
        boxes = protocol.component_boxes(prob, 7, (2, 3), .5, 0, xy_offset=(1, -1))
        assert len(boxes) == 1
        assert boxes[0]["bbox"] == [3., 0., 1., 1.]
        assert boxes[0]["component_area_pixels"] == 1

    def test_all_components_count_without_topk(self):
        prob = np.zeros((37, 37), dtype=np.float32)
        prob[::2, ::2] = .6
        boxes = protocol.component_boxes(prob, 1, prob.shape, .5, 0)
        assert len(boxes) == 361
        metrics, _ = protocol.match_all(gt([], n_images=2), boxes)
        assert (metrics["TP"], metrics["FP"], metrics["FN"]) == (0, 361, 0)
        assert metrics["Fa_per_image"] == 180.5

    def test_score_ordered_greedy_is_not_replaced_with_optimal_assignment(self):
        truth = gt([[0, 0, 10, 10], [4, 0, 10, 10]])

        boxes = [det([1, 0, 10, 10], .9), det([0, 0, 8, 10], .8)]
        metrics, details = protocol.match_all(truth, boxes)
        assert (metrics["TP"], metrics["FP"], metrics["FN"]) == (1, 1, 1)
        assert [p["matched_gt_id"] for p in details[0]["predictions"]] == [1, None]

        boxes[0]["score"], boxes[1]["score"] = .8, .9
        metrics, _ = protocol.match_all(truth, boxes)
        assert (metrics["TP"], metrics["FP"], metrics["FN"]) == (2, 0, 0)

    def test_prediction_score_ties_are_stable_and_iou_ties_choose_last_gt(self):
        truth = gt([[0, 0, 10, 10], [4, 0, 10, 10]])
        boxes = [det([1, 0, 10, 10], .8), det([0, 0, 8, 10], .8)]
        metrics, _ = protocol.match_all(truth, boxes)
        assert metrics["TP"] == 1
        metrics, _ = protocol.match_all(truth, list(reversed(boxes)))
        assert metrics["TP"] == 2
        _, details = protocol.match_all(gt([[0, 0, 2, 2], [0, 0, 2, 2]]),
                                      [det([0, 0, 2, 2], .8)])
        assert details[0]["predictions"][0]["matched_gt_id"] == 2

    def test_iou_point_five_inclusive_and_empty_images_in_fa_denominator(self):
        metrics, _ = protocol.match_all(gt([[0, 0, 2, 2]], n_images=2),
                                      [det([0, 0, 1, 2], .9), det([0, 0, 1, 1], .8, 2)])
        assert metrics["TP"] == 1
        assert metrics["FP"] == 1
        assert metrics["R"] == 100
        assert metrics["P"] == 50
        self.assertAlmostEqual(metrics["F1"], 200 / 3, places=10)
        assert metrics["Fa_per_image"] == .5

    def test_empty_detections_and_empty_gt_are_defined(self):
        metrics, _ = protocol.match_all(gt([], n_images=2), [])
        assert all(metrics[key] == 0 for key in ("TP", "FP", "FN", "P", "R", "F1", "Fa_per_image"))

    def test_restore_one_pixel_axes_and_reject_logits(self):
        prob = geometry.restore_probability(np.array([[[.7, .8]]]), (1, 2))
        assert prob.shape == (1, 2)
        assert prob.dtype == np.float32
        with self.assertRaises(ValueError):
            geometry.restore_probability(np.array([[1.1]]), (1, 1))
        with self.assertRaises(ValueError):
            geometry.restore_probability(np.array([[np.nan]]), (1, 1))

    def test_better_preserves_f1_then_fa_then_lower_mask_with_tolerance(self):
        current = {"mask_threshold": .5, "metrics": {"F1": 80., "Fa_per_image": .2}}
        better_f1 = {"mask_threshold": .7, "metrics": {"F1": 81., "Fa_per_image": .4}}
        lower_fa = {"mask_threshold": .7, "metrics": {"F1": 80., "Fa_per_image": .1}}
        lower_mask = {"mask_threshold": .3, "metrics": {"F1": 80., "Fa_per_image": .2}}
        assert protocol.better(better_f1, current)
        assert protocol.better(lower_fa, current)
        assert protocol.better(lower_mask, current)
        assert not protocol.better(current, current)
        tiny_delta = copy.deepcopy(lower_mask)
        tiny_delta["metrics"]["F1"] -= 5e-13
        assert protocol.better(tiny_delta, current)

    def test_exact_split_coverage_and_validation_only_freeze(self):
        tmp_path = self.tmp_path
        directory = create_example(tmp_path / "demo")
        manifest, gt_paths, identity = load_manifest(directory / "manifest.json")
        val_maps, map_identity = load_maps(directory / "probabilities.json", manifest, "val")
        selected, candidates = select_validation(manifest, load_gt(manifest, gt_paths["val"], "val"), val_maps)
        assert tuple(row["mask_threshold"] for row in candidates) == protocol.MASK_GRID
        assert selected["mask_threshold"] == .3
        frozen = freeze_selection(selected, identity, map_identity)
        verify_selection(frozen, identity)
        with self.assertRaises(ValueError):
            protocol.convert(manifest, "test", val_maps, .3)
        tampered = copy.deepcopy(frozen)
        tampered["mask_threshold"] = .9
        with self.assertRaises(ValueError):
            verify_selection(tampered, identity)
        changed_gt_identity = copy.deepcopy(identity)
        changed_gt_identity["GT_sha256"]["test"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_selection(frozen, changed_gt_identity)

    def test_cli_select_then_frozen_test_and_no_overwrite(self):
        tmp_path = self.tmp_path
        directory = create_example(tmp_path / "demo")
        index_path = directory / "probabilities.json"
        index = json.loads(index_path.read_text())
        val_only = copy.deepcopy(index)
        del val_only["splits"]["test"]
        (directory / "val_only.json").write_text(json.dumps(val_only))
        select_args = ["select", "--manifest", str(directory / "manifest.json"),
                       "--probabilities", str(directory / "val_only.json"),
                       "--output", str(tmp_path / "selected")]
        assert main(select_args) == 0
        selection = tmp_path / "selected/selection.json"
        frozen = json.loads(selection.read_text())
        assert frozen["mask_threshold"] == .3
        assert frozen["confidence_threshold"] is None
        assert main(["test", "--manifest", str(directory / "manifest.json"),
                     "--probabilities", str(index_path), "--selection", str(selection),
                     "--output", str(tmp_path / "tested")]) == 0
        result = json.loads((tmp_path / "tested/metrics.json").read_text())
        metrics = result["metrics"]
        assert (metrics["TP"], metrics["FP"], metrics["FN"]) == (1, 1, 0)
        assert metrics["P"] == 50
        assert metrics["R"] == 100
        self.assertAlmostEqual(metrics["F1"], 200 / 3, places=10)
        assert metrics["Fa_per_image"] == .5
        assert "AP" not in result
        with self.assertRaises(ValueError):
            main(select_args)

    def test_cli_rejects_cross_model_prediction_identity(self):
        tmp_path = self.tmp_path
        directory = create_example(tmp_path / "demo")
        manifest = directory / "manifest.json"
        index = directory / "probabilities.json"
        main(["select", "--manifest", str(manifest), "--probabilities", str(index),
              "--output", str(tmp_path / "selected")])
        changed = json.loads(index.read_text())
        changed["prediction_set_id"] = "different-model"
        index.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different model"):
            main(["test", "--manifest", str(manifest), "--probabilities", str(index),
                  "--selection", str(tmp_path / "selected/selection.json"),
                  "--output", str(tmp_path / "tested")])

if __name__ == "__main__":
    unittest.main()
