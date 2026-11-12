# ASTRA

Core implementation of **ASTRA: Adapting Real-Time Detection Transformers to Infrared Small Targets in Remote Sensing Imagery**.

ASTRA adapts RT-DETR for infrared small-target detection using a spatial–frequency context encoder (SFCE), target-aware Gaussian mask supervision (TGMS), and a micro-scale assignment and regression strategy (MARS).

## Initial release

The first release provides the core architecture and modules, inference, deployment conversion, and unified target-level evaluation. Trained weights, full training losses, denoising integration and complete benchmark training recipes will follow after paper acceptance. See the [release plan](docs/release_plan.md).

## Architecture

```text
Image → ResNet-18 with ASC in both terminal blocks
      → input projection → LGCI → lateral projection Y5
B3, B4, Y5 → CCFM → F3, F4, F5
           → stride-aligned encoder priors → Top-Q selection
           → decoder → boxes and scores

Training: B3, B4, Y5 → independent TGMS pathway
                     → P3/P4/P5 response heads → auxiliary loss
```

SFCE combines frequency selection, modulated sampling and local/global context. TGMS supervises shared pre-decoder features and is completely removed by explicit deployment conversion. MARS supplies reference priors and the geometric distance primitives. Complete matching/regression training integration is scheduled for the second release. The final-model ASC sampling-scale behavior and exact MARS coordinate convention are documented in [implementation notes](docs/compatibility.md).

## Installation

Python 3.8 or newer is required. Install a PyTorch build suitable for your device, then:

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

The explicit `pytorch` sampling backend runs on CPU. Optional torchvision and MMCV backends require compatible compiled operators.

## Quick start

```sh
python -m astra inspect --backend pytorch
python -m astra smoke --height 128 --width 128 --backend pytorch
python -m unittest discover -s tests -v
```

The randomly initialized model outputs normalized center-format boxes and class probabilities. See [model usage](docs/model_usage.md), [runtime validation](docs/validation.md), and `examples/model_smoke.py`.

## Segmentation-to-detection evaluation

```sh
python -m evaluation describe
python -m evaluation select --manifest data/manifest.json --probabilities data/probabilities.json --output results/validation
python -m evaluation test --manifest data/manifest.json --probabilities data/probabilities.json --selection results/validation/selection.json --output results/test
```

The evaluator converts probability maps into eight-connected components and tight bounding boxes. One-to-one IoU matching at 0.5 produces **P, Pd/Recall, F1 and Fa per image**. The mask threshold is selected by validation F1 and frozen for test evaluation. All components are retained, without morphology, area filtering, or Top-K truncation. The complete input format and matching conventions are in the [evaluation guide](evaluation/README.md).

## Citation and licensing

Citation metadata is provided in [CITATION.cff](CITATION.cff). This release retains the AGPL terms of the Ultralytics-derived integration and the applicable Apache-2.0/MIT notices of other components. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [third-party notices](THIRD_PARTY_NOTICES.md).
