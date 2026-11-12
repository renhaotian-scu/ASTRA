# Runtime validation

The local validation environment is Windows, Python 3.8.20 and PyTorch 2.4.1+cpu. The explicit pure-PyTorch sampling backend supports CPU execution.

```sh
python -m unittest discover -s tests -v
python -m astra smoke --height 640 --width 640 --backend pytorch
```

The first-release tests cover deformable sampling and its gradients, complete model inference, exact equality before/after TGMS pruning, rectangular input, and the segmentation-to-box evaluation protocol. Training-only tests are delivered together with the deferred training integration.

The deployed model has 20,252,530 parameters; the graph retaining auxiliary fusion has 22,028,725 parameters. Random-weight execution checks establish the interface and numerical behavior, rather than benchmark detection accuracy. Native MMCV/CUDA parity and hardware timing are separate from the portable CPU checks.

## Fixed-checkpoint CPU compatibility

The released graph was also checked privately against nine final-model checkpoints across three datasets. All state dictionaries loaded strictly, and the three architecture fields matched. Two probe inputs were used per checkpoint: a zero-valued 640 × 640 image and a seeded random 128 × 192 image. Prediction sets matched the original source graph with maximum absolute error 7.45e-07; ordinary evaluation and branch-pruned predictions were exactly equal in all eighteen comparisons. This CPU graph-compatibility check does not measure dataset accuracy. Checkpoint tensors and private file locations are excluded from the release.
