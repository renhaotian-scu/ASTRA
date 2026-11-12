

from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

_BACKEND = ContextVar('astra_deform_backend', default='pytorch')

def resolve_deform_backend(name):

    if name not in ('auto', 'pytorch', 'torchvision', 'mmcv'):
        raise ValueError("backend must be 'auto', 'pytorch', 'torchvision' or 'mmcv'")
    if name != 'auto':
        return name
    try:
        from torchvision.ops import deform_conv2d
        if callable(deform_conv2d) and hasattr(torch.ops.torchvision, 'deform_conv2d'):
            return 'torchvision'
    except (ImportError, OSError, AttributeError, RuntimeError):
        pass
    return 'pytorch'

@contextmanager
def use_deform_backend(name):

    token = _BACKEND.set(resolve_deform_backend(name))
    try:
        yield
    finally:
        _BACKEND.reset(token)

class ModulatedDeformConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError('Convolution groups must divide input and output channels')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.init_weights()

    def init_weights(self):
        bound = 1.0 / math.sqrt((self.in_channels // self.groups) * math.prod(self.kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

def modulated_deform_conv2d(x, offset, mask, weight, bias, stride, padding,
                          dilation, groups, deform_groups):

    if weight.shape[1] * groups != x.shape[1]:
        raise ValueError('Weight/input convolution group mismatch')
    if offset.shape[1] != deform_groups * 2 * weight.shape[2] * weight.shape[3]:
        raise ValueError('Offset/deformable group mismatch')
    name = _BACKEND.get()
    if name == 'pytorch':
        return modulated_deform_conv2d_reference(x, offset, mask, weight, bias,
                                               stride, padding, dilation, groups, deform_groups)
    if name == 'mmcv':
        try:
            from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d as op
        except (ImportError, OSError) as exc:
            raise RuntimeError('The requested MMCV backend requires a working mmcv.ops extension') from exc
        return op(x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups)
    try:
        from torchvision.ops import deform_conv2d
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError('The torchvision backend requires matching torch/torchvision builds with deform_conv2d') from exc
    return deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)

def modulated_deform_conv2d_reference(x, offset, mask, weight, bias, stride,
                                     padding, dilation, groups=1, deform_groups=1):

    if x.ndim != 4 or weight.ndim != 4 or offset.ndim != 4 or mask.ndim != 4:
        raise ValueError('DCNv2 reference expects four-dimensional tensors')
    batch, channels, height, width = x.shape
    out_channels, kernel_channels, kh, kw = weight.shape
    ho, wo = offset.shape[-2:]
    if channels % deform_groups or channels % groups or out_channels % groups:
        raise ValueError('Invalid deformable or convolution groups')
    if kernel_channels * groups != channels:
        raise ValueError('Weight shape does not match convolution groups')
    k = kh * kw
    if tuple(offset.shape[:2]) != (batch, deform_groups * 2 * k):
        raise ValueError('Offset shape does not match deformable groups/kernel')
    if tuple(mask.shape) != (batch, deform_groups * k, ho, wo):
        raise ValueError('Mask shape does not match offsets')
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    expected = ((height + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
                (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1)
    if (ho, wo) != expected:
        raise ValueError('Offsets do not match convolution output shape')
    oy, ox = torch.meshgrid(torch.arange(ho, device=x.device, dtype=x.dtype),
                            torch.arange(wo, device=x.device, dtype=x.dtype), indexing='ij')
    ky, kx = torch.meshgrid(torch.arange(kh, device=x.device, dtype=x.dtype),
                            torch.arange(kw, device=x.device, dtype=x.dtype), indexing='ij')
    offsets = offset.reshape(batch, deform_groups, k, 2, ho, wo)
    ys = oy * sh - ph + ky.flatten().view(k, 1, 1) * dh + offsets[:, :, :, 0]
    xs = ox * sw - pw + kx.flatten().view(k, 1, 1) * dw + offsets[:, :, :, 1]
    grid = torch.stack(((2 * xs + 1) / width - 1, (2 * ys + 1) / height - 1), dim=-1)
    sampled = F.grid_sample(x.reshape(batch * deform_groups, channels // deform_groups, height, width),
                            grid.reshape(batch * deform_groups, k * ho, wo, 2),
                            mode='bilinear', padding_mode='zeros', align_corners=False)
    sampled = sampled.reshape(batch, deform_groups, channels // deform_groups, k, ho, wo)
    sampled = sampled * mask.reshape(batch, deform_groups, 1, k, ho, wo)
    sampled = sampled.reshape(batch, groups, channels // groups * k, ho * wo)
    kernels = weight.reshape(groups, out_channels // groups, channels // groups * k)
    y = torch.einsum('goc,bgcn->bgon', kernels, sampled).reshape(batch, out_channels, ho, wo)
    return y if bias is None else y + bias.reshape(1, out_channels, 1, 1)

def check_deform_backend(backend='pytorch', device='cpu'):

    x = torch.randn(1, 2, 5, 5, device=device, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, device=device, requires_grad=True)
    offsets = torch.zeros(1, 18, 5, 5, device=device, requires_grad=True)
    mask = torch.full((1, 9, 5, 5), 0.5, device=device, requires_grad=True)
    with use_deform_backend(backend):
        y = modulated_deform_conv2d(x, offsets, mask, w, None, 1, 1, 1, 1, 1)
    y.square().mean().backward()
    finite = bool(torch.isfinite(y).all() and all(t.grad is not None and torch.isfinite(t.grad).all()
                                                for t in (x, w, offsets, mask)))
    if not finite:
        raise RuntimeError('Deformable-convolution output/gradient health check failed')
    return {'backend': resolve_deform_backend(backend), 'device': str(x.device), 'output_shape': list(y.shape),
            'finite_output_and_gradients': finite}
