# Model interfaces

The core includes an explicit architecture builder, an independent TGMS auxiliary pathway and MARS matching/regression utilities. The model is randomly initialized unless a compatible state dictionary is loaded.

```python
import torch
from astra import build_astra

torch.set_num_threads(2)
model = build_astra(num_classes=1, auxiliary=True, backend="pytorch")
model.eval()
x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    predictions, raw_outputs = model(x)
deploy = model.to_deploy()
```

Check `examples/model_smoke.py` for the current output contract and comparison of the ordinary evaluation graph with the branch-pruned graph. The prediction tensor has shape `[B, 300, 4 + num_classes]`: normalized center-format boxes followed by class probabilities.

This initial wrapper provides inference and deployment conversion. Full training losses, target generation, assignment and denoising integration are scheduled for the post-acceptance release. Calling the full-model forward in training mode raises an explicit error.

Archived pickled model objects are not automatically deserialized. Verify the architecture and state-dictionary contract before loading any checkpoint. A successful shape check is not an accuracy reproduction.
