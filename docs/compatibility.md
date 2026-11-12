# Implementation notes

## Architecture

The core retains both terminal-backbone ASC blocks, full-input HiLo branches, decoder priors and the registered architecture parameters. The configurations use `reg_max=0`. `CORE_SOURCE_MANIFEST.json` identifies the selected research-source symbols. Unrelated networks from the original working repository are excluded.

## ASC sampling

ASC combines frequency selection with nine-position modulation. The preserved scale predictor uses zero initialization followed by an unshifted ReLU. In the final trained artifacts, its weights and biases remain zero; the resulting scalar field is zero and sampling uses the regular dilation-one grid. Frequency gates and modulation remain input dependent. This release preserves that behavior and does not substitute a differently initialized adaptive-scale variant.

The backend is chosen explicitly. The pure-PyTorch implementation supports portable execution; the research implementation used MMCV modulated deformable convolution. Native MMCV/CUDA cross-backend equivalence and throughput are separate from the CPU execution checks.

## MARS geometry

For normalized center-format boxes `(cx, cy, w, h)`, the retained helper first halves the extents, forms left/top edges, and recovers its effective center using the already-halved extents:

```text
u = w / 2; v = h / 2
left = cx - u; top = cy - v
mu_x = left + u / 2 = cx - w / 4
mu_y = top  + v / 2 = cy - h / 4

D_squared = (delta_cx - delta_w / 4)^2
          + (delta_cy - delta_h / 4)^2
          + (delta_w^2 + delta_h^2) / 16
similarity = exp(-sqrt(D_squared + 1e-7) / 0.02)
```

The final research training integration uses the explicit constant `0.02`; that complete integration is scheduled for the second release. Pass `constant=0.02` when using the distance primitive with normalized boxes. This is an implementation-specific parameterization; it differs from the standard centered-box Wasserstein formula. `astra.mars.legacy_exact` retains its arithmetic.

## Deployment and checkpoints

Ordinary evaluation skips response projections but retains auxiliary fusion. `model.to_deploy()` removes the complete TGMS auxiliary subgraph. The preserved BatchNorm settings are `eps=0.001` and `momentum=0.03`; they are initialized in the public model builder because a state dictionary does not contain them. The deployed architecture has 20,252,530 parameters. The non-pruned architecture has 22,028,725 parameters.

The current examples run with random initialization. Weights, dataset-specific training recipes and the verified checkpoint-to-configuration map are scheduled for the post-acceptance release. Native pickled research-model objects are not automatically loaded by this package.
