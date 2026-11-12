# Initial core release and planned additions

The initial source release focuses on the model architecture, core inference modules and the complete target-level evaluator.

## Available now

- ASC frequency selection and modulated sampling; LGCI local/global attention.
- TGMS auxiliary fusion and response-head architecture.
- MARS stride-aligned reference priors and geometric distance helpers.
- Complete model inference and explicit TGMS deployment pruning.
- Architecture YAML files, installation instructions, examples and inference tests.
- Segmentation-to-box conversion, validation threshold selection, and P/Pd/F1/Fa evaluation.
- Citation metadata, source attribution and applicable license notices.

## After paper acceptance

- Verified checkpoints for the three datasets and checksum/configuration mappings.
- Complete Gaussian target/loss implementation and joint detection objectives.
- Hungarian cost, regression and quality-target training integration.
- Denoising training integration and the end-to-end training launcher.
- Dataset-specific optimization recipes and model-specific prediction export adapters.

The initial release is intentionally partial. Its scope is defined by these functional components, rather than by deleting an arbitrary percentage of source lines. Dataset files, private experiments, server administration and account credentials are excluded.
