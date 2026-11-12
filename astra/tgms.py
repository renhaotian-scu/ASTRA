# SPDX-License-Identifier: AGPL-3.0-only

import torch
from torch import nn
from torch.nn import functional as F
from .layers import Conv

class MFM(nn.Module):

    def __init__(self, inc, dim, reduction=8):
        super(MFM, self).__init__()
        self.height = len(inc)
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(dim, d, 1, bias=False), nn.ReLU(), nn.Conv2d(d, dim * self.height, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.ModuleList([])
        for i in inc:
            if i != dim:
                self.conv1x1.append(Conv(i, dim, 1))
            else:
                self.conv1x1.append(nn.Identity())

    def forward(self, in_feats_):
        assert len(in_feats_) == self.height, f'MFM 期望 {self.height} 个输入,实际收到 {len(in_feats_)} 个,请检查 YAML 该层的 from 索引。'
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx]))
        ref_hw = in_feats[0].shape[-2:]
        for i in range(1, len(in_feats)):
            if in_feats[i].shape[-2:] != ref_hw:
                in_feats[i] = F.interpolate(in_feats[i], size=ref_hw, mode='bilinear', align_corners=False)
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        out = torch.sum(in_feats * attn, dim=1)
        return out

class RTDETRDenseAuxHead(nn.Module):

    def __init__(self, nc=80, hd=256, strides=[4, 8, 16], ch=()):
        super().__init__()
        self.nc = nc
        self.register_buffer('stride', torch.tensor(strides).float(), persistent=False)
        self.aux_mask_preds = None
        assert len(ch) > 0, 'RTDETRDenseAuxHead 收到空的 ch,说明 parse_model 未把来源通道数传进来。请确认 ASTRA-PATCH P1 已生效。'
        assert len(ch) == len(strides), f'RTDETRDenseAuxHead: 输入层数 {len(ch)} 与 strides 数量 {len(strides)} 不一致。'
        self.mask_projs = nn.ModuleList([nn.Sequential(nn.Conv2d(c, hd // 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(hd // 4), nn.ReLU(inplace=True), nn.Conv2d(hd // 4, 1, kernel_size=1)) for c in ch])

    def forward(self, x):
        if self.training:
            self.aux_mask_preds = [proj(feat) for proj, feat in zip(self.mask_projs, x)]
        else:
            self.aux_mask_preds = None
        return x

TGMSHead = RTDETRDenseAuxHead
