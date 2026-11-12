# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics-derived integration; see THIRD_PARTY_NOTICES.md.
"""MARS reference-prior and geometric distance primitives.

Complete assignment/regression training integration follows after acceptance.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, PIoU=False, PIoU2=False, eps=1e-07, scale=0.0, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = (box1.chunk(4, -1), box2.chunk(4, -1))
        w1_, h1_, w2_, h2_ = (w1 / 2, h1 / 2, w2 / 2, h2 / 2)
        b1_x1, b1_x2, b1_y1, b1_y2 = (x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_)
        b2_x1, b2_x2, b2_y1, b2_y2 = (x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1, b1_y2 - b1_y1 + eps)
        w2, h2 = (b2_x2 - b2_x1, b2_y2 - b2_y1 + eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or PIoU or PIoU2:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU or EIoU or SIoU or PIoU or PIoU2 or ShapeIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = 4 / math.pi ** 2 * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            elif EIoU:
                rho_w2 = (b2_x2 - b2_x1 - (b1_x2 - b1_x1)) ** 2
                rho_h2 = (b2_y2 - b2_y1 - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
            elif SIoU:
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps
            elif ShapeIoU:
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
                c2 = cw ** 2 + ch ** 2 + eps
                center_distance_x = (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 / 4
                center_distance_y = (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
            elif PIoU or PIoU2:
                dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
                dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
                dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
                dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
                P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
                piou_v1 = 1 - iou - torch.exp(-P ** 2) + 1
                if PIoU:
                    return 1 - piou_v1
                elif PIoU2:
                    q = torch.exp(-P)
                    x = q * Lambda
                    return 1 - 3 * x * torch.exp(-x ** 2) * piou_v1
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou

def wasserstein_loss(box1, box2, xywh=True, eps=1e-07, constant=12.8):
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_min, y_min, x_max, y_max),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = (box1.chunk(4, -1), box2.chunk(4, -1))
        w1, h1, w2, h2 = (w1 / 2, h1 / 2, w2 / 2, h2 / 2)
        b1_x1, b1_x2, b1_y1, b1_y2 = (x1 - w1, x1 + w1, y1 - h1, y1 + h1)
        b2_x1, b2_x2, b2_y1, b2_y2 = (x2 - w2, x2 + w2, y2 - h2, y2 + h2)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1, b1_y2 - b1_y1 + eps)
        w2, h2 = (b2_x2 - b2_x1, b2_y2 - b2_y1 + eps)
    b1_x_center, b1_y_center = (b1_x1 + w1 / 2, b1_y1 + h1 / 2)
    b2_x_center, b2_y_center = (b2_x1 + w2 / 2, b2_y1 + h2 / 2)
    center_distance = (b1_x_center - b2_x_center) ** 2 + (b1_y_center - b2_y_center) ** 2 + eps
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)



legacy_exact = wasserstein_loss

def stride_aligned_reference_priors(shapes, *, anchor_scale=1.0,
                                    grid_size=None, dtype=torch.float32,
                                    device='cpu', eps=1e-3):
    """Return prior logits and validity mask using the decoder's exact grid rule.

    ``shapes`` contains (height,width) for each encoder scale. ``grid_size=None``
    uses width=1/W and height=1/H; these are initialization priors, not fixed
    prediction boxes. A non-square input keeps x/W and y/H separately.
    """
    anchors=[]
    for i,(h,w) in enumerate(shapes):
        sy=torch.arange(end=h,dtype=dtype,device=device)
        sx=torch.arange(end=w,dtype=dtype,device=device)
        gy,gx=torch.meshgrid(sy,sx,indexing='ij')
        xy=(torch.stack([gx,gy],-1).unsqueeze(0)+0.5)/torch.tensor([w,h],dtype=dtype,device=device)
        if grid_size is None:
            wh=(torch.tensor([1.0/w,1.0/h],dtype=dtype,device=device)*anchor_scale).expand_as(xy).contiguous()
        else:
            wh=torch.ones_like(xy)*grid_size*(2.0**i)
        anchors.append(torch.cat([xy,wh],-1).view(-1,h*w,4))
    anchors=torch.cat(anchors,1)
    valid=((anchors>eps)*(anchors<1-eps)).all(-1,keepdim=True)
    return torch.log(anchors/(1-anchors)).masked_fill(~valid,float('inf')),valid
