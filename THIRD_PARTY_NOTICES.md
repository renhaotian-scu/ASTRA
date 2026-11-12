# Third-party source and license notices

The detector integration retains the AGPL-3.0 terms of its Ultralytics-derived
source. The full text is in [LICENSE](LICENSE). The source distribution also
contains adaptations of the separately identified components below; their
existing notices and license texts are preserved. No blanket MIT relicensing
is applied.

| Upstream | Source relationship | License text |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Backbone and detector layers, attention/decoder helpers, box geometry and assignment/loss integration | [AGPL-3.0](LICENSES/Ultralytics-AGPL-3.0.txt) |
| [FADC](https://github.com/Linwei-Chen/FADC) | Frequency selection and adaptive dilated convolution used by ASC; modified scalar sampling path and integration | [Apache-2.0](LICENSES/FADC-Apache-2.0.txt) |
| [LITv2 / HiLo](https://github.com/ziplab/LITv2) | Local/global split attention used in the LGCI wrapper; local source retains its architecture-specific adaptations | [Apache-2.0](LICENSES/HiLo-LITv2-Apache-2.0.txt) |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | Dual-layout LayerNorm; local default data layout differs | [MIT, Meta](LICENSES/ConvNeXt-MIT.txt) |
| [DCMPNet](https://github.com/zhoushen1/DCMPNet) | MFM fusion adapted with local projection and resizing behavior | [MIT, IDKiro](LICENSES/DCMPNet-MIT.txt) |
| [MMCV](https://github.com/open-mmlab/mmcv) | Optional external modulated deformable-convolution operator interface; binary not bundled | [Apache-2.0](LICENSES/MMCV-Apache-2.0.txt) |

The source headers and [NOTICE](NOTICE) retain the known original attributions.
The isolated code is modified for this package; the provenance manifests identify
the selected source bodies. Exact historical upstream transplant commits were
not recoverable from the stored source and are not invented here. Official
license URLs, complete downloaded line coverage and local text hashes are in
[LICENSES/SOURCES.json](LICENSES/SOURCES.json).

PyTorch, torchvision, NumPy, SciPy and Pillow are external dependencies installed
by the user; their source/binaries are not redistributed in this archive and
their own licenses remain applicable. The portable operator backend is identified
explicitly in the architecture documentation. Attribution does not establish
checkpoint-level numerical equivalence between operator backends.
