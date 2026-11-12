# Target-level segmentation evaluation

The standalone `evaluation` package consumes original-resolution probability
maps and the supplied bounding-box annotations. It does not regenerate GT boxes
from masks or load a segmentation network. See [the full protocol and JSON input
format](../evaluation/README.md).

```sh
python -m evaluation describe
python -m evaluation select --manifest data/manifest.json --probabilities data/probabilities.json --output results/validation
python -m evaluation test --manifest data/manifest.json --probabilities data/probabilities.json --selection results/validation/selection.json --output results/test
python -m unittest discover -s tests -p test_segmentation_evaluation.py -v
```

Threshold selection uses validation F1, then lower Fa and lower threshold for
ties. The selected mask threshold is frozen before the test operation.
Eight-connected components become tight boxes; all components are retained.
The original foreground-mean score ordering and greedy one-to-one IoU >= 0.5
matching are preserved. P, Pd/Recall, F1 and false-positive instances per image
are reported; empty-GT images remain in the Fa denominator.

The package includes a hand-checkable synthetic example. Synthetic checks verify
implementation behavior and do not reproduce the paper's benchmark scores.
Source-function fingerprints are in `evaluation/SOURCE_PROVENANCE.json`.
