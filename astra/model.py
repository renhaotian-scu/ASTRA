"""Focused static ASTRA graph, preserving archived layer and state-dict indices.

The graph is the archived ResNet-18 / reg_max=0 variant. It is a core architecture
release, not a training runner or a claim to reproduce unpublished checkpoints.
"""
from copy import deepcopy
import ast
import torch
from torch import nn
from .architecture import TRAIN_ARCHITECTURE, DEPLOY_ARCHITECTURE
from .layers import Conv, ConvNormLayer, RepC3, Blocks, BasicBlock, Concat
from .backbone import BasicBlock_FADC
from .attention import TransformerEncoderLayer_HiLo
from .transformer import AIFI
from .decoder import RTDETRDecoder, RTDETRPriorDecoder
from .tgms import MFM, RTDETRDenseAuxHead
from .deform_backend import use_deform_backend, resolve_deform_backend

_MODULES = {c.__name__: c for c in (Conv, ConvNormLayer, RepC3, Blocks, BasicBlock,
             BasicBlock_FADC, Concat, TransformerEncoderLayer_HiLo, AIFI,
             RTDETRPriorDecoder, MFM, RTDETRDenseAuxHead)}
_MODULES.update({'nn.MaxPool2d': nn.MaxPool2d, 'nn.Upsample': nn.Upsample})


class ASTRA(nn.Module):
    """ASTRA core model with an explicit training-auxiliary/deployment boundary.

    This initial public wrapper exposes inference. The complete training
    integration is reserved for the post-acceptance release.
    ``eval()`` returns ``(predictions, raw_decoder_outputs)`` where
    predictions have shape [B,300,4+num_classes], normalized xywh boxes followed
    by sigmoid scores. Ordinary eval still executes auxiliary fusion nodes; call
    ``to_deploy()`` to remove the entire auxiliary component.
    """
    def __init__(self, num_classes=1, auxiliary=True, backend='pytorch'):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError('num_classes must be a positive integer')
        self.num_classes = num_classes
        self.num_queries = 300
        self.auxiliary = bool(auxiliary)
        self.backend = resolve_deform_backend(backend)
        self.architecture = deepcopy(TRAIN_ARCHITECTURE if auxiliary else DEPLOY_ARCHITECTURE)
        self.architecture['nc'] = num_classes
        self.model = nn.ModuleList()
        self.froms = []
        channels = []
        for index, (f, repeats, name, arguments) in enumerate(self.architecture['backbone'] + self.architecture['head']):
            cls = _MODULES[name]
            args = deepcopy(arguments)
            for j, value in enumerate(args):
                if isinstance(value, str):
                    if value == 'nc':
                        args[j] = num_classes
                    else:
                        try:
                            args[j] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            pass
            c1 = (3 if index == 0 else channels[-1]) if f == -1 else (channels[f] if isinstance(f, int) else None)
            if cls in (Conv, ConvNormLayer, RepC3):
                c2 = args[0]
                args = [c1, c2] + args[1:]
                if cls is RepC3:
                    args.insert(2, repeats)
                    repeats = 1
            elif cls is Blocks:
                c2 = args[0]
                args[1] = _MODULES[args[1]]
                args = [c1] + args
            elif cls in (AIFI, TransformerEncoderLayer_HiLo):
                c2 = c1
                args = [c1] + args
            elif cls is Concat:
                c2 = sum(channels[j] for j in f)
            elif cls is MFM:
                c2 = args[0]
                args = [[channels[j] for j in f]] + args
            elif cls is RTDETRDenseAuxHead:
                c2 = None
                args[0] = num_classes
                args.append([channels[j] for j in f])
            elif issubclass(cls, RTDETRDecoder):
                args[0] = num_classes
                args.insert(1, [channels[j] for j in f])
                c2 = None
            else:
                c2 = c1
            layer = nn.Sequential(*(cls(*args) for _ in range(repeats))) if repeats > 1 else cls(*args)
            # Preserve useful archived graph metadata without a parser dependency.
            layer.i = index
            layer.f = f
            self.model.append(layer)
            self.froms.append(f)
            channels.append(c2)

        # Match the initialization attributes serialized by the research framework.
        # These BatchNorm settings are not stored in a state_dict.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Hardswish, nn.SiLU)):
                module.inplace = True
        self.eval()

    @property
    def auxiliary_logits(self):
        return self.model[34].aux_mask_preds if self.auxiliary else None

    def forward(self, images, batch=None):
        if self.training:
            raise RuntimeError('The initial release provides inference; complete training integration follows after acceptance.')
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected images shaped [B,3,H,W]')
        # The frozen HiLo window topology needs even stride-32 spatial dimensions.
        if any(v % 64 for v in images.shape[-2:]) or min(images.shape[-2:]) < 128:
            raise ValueError('This archived graph requires H,W divisible by 64 and at least 128 (e.g. 640x640)')
        values = []
        with use_deform_backend(self.backend):
            for f, layer in zip(self.froms, self.model):
                x = (images if not values else values[-1]) if f == -1 else values[f] if isinstance(f, int) else [values[j] for j in f]
                values.append(layer(x, batch=batch) if isinstance(layer, RTDETRDecoder) else layer(x))
        return values[-1]

    def to_deploy(self, inplace=False):
        """Remove nodes26–34 and remap decoder35→26; preserve detector tensors.

        The result is placed in evaluation mode. The default leaves the input
        graph untouched. This is exact pruning within this package; testing
        arbitrary historical checkpoint pickles is a separate responsibility.
        """
        model = self if inplace else deepcopy(self)
        if model.auxiliary:
            model.model = nn.ModuleList(list(model.model[:26]) + [model.model[35]])
            model.froms = list(model.froms[:26]) + [[19, 22, 25]]
            model.model[26].i = 26
            model.auxiliary = False
            model.architecture = deepcopy(DEPLOY_ARCHITECTURE)
            model.architecture['nc'] = model.num_classes
        return model.eval()


def build_astra(num_classes=1, auxiliary=True, backend='pytorch'):
    """Build the focused, untrained reg_max=0 ASTRA architecture."""
    return ASTRA(num_classes=num_classes, auxiliary=auxiliary, backend=backend)
