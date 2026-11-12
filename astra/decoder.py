# Ultralytics-derived code, AGPL-3.0-or-later.
# See LICENSE and THIRD_PARTY_NOTICES.md for source attribution.
"""Archived RT-DETR decoder and stride-aligned MARS reference priors."""
import torch
from torch import nn
from torch.nn.init import constant_,xavier_uniform_
from .compat import TORCH_1_10
from .transformer import MLP,DeformableTransformerDecoderLayer,DeformableTransformerDecoder,bias_init_with_prob,linear_init_

class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            reg_max=0,   # [ASTRA-PATCH P9] 分布回归 bins: 0 = 旧 4 通道架构 (旧权重兼容); >0 = DDF 全分布化
            reg_scale=16):  # bin 宽 = ref_w/reg_scale: 8/4/2px 目标 = 0.5/0.25/0.125 px/bin
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        # [ASTRA-PATCH P9] yaml 位置传参时 act 传 None 表示用默认 (yaml 无法表示函数对象)
        if act is None:
            act = nn.ReLU()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        # [ASTRA-PATCH P9] DDF 全分布化 (D-FINE 架构): 每层 bbox 头输出 4x(reg_max+1) 分布 logits,
        # decoder 循环内 Integral 解码 -> 训练/推理/L1/GIoU/DFL 共用同一框。reg_max=0 回退旧架构。
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        bbox_out = 4 if reg_max == 0 else 4 * (reg_max + 1)
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, bbox_out, num_layers=3) for _ in range(ndl)])
        self.dec_bbox_head_ddf = None  # 平行头已废弃 (保留属性名兼容旧代码)
        # [FIX 2026-09-17] _reset_parameters 从未被调用 (死代码): DDF 头此前一直是 kaiming 随机初始化
        # (输出量级小, softmax 近似均匀 -> 旧中心下 ≈ +s/2 偏移)。按修复方案显式置零最后一层:
        # 初始 logits=0 -> softmax 均匀 -> integral=16 -> d=0, 各层初始 0 偏移。
        if reg_max > 0:
            for _h in self.dec_bbox_head:
                constant_(_h.layers[-1].weight, 0.)
                constant_(_h.layers[-1].bias, 0.)

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        if self.training:
            raise RuntimeError('The initial release provides inference; complete training integration follows after acceptance.')
        feats, shapes = self._get_encoder_input(x)
        dn_embed = dn_bbox = attn_mask = dn_meta = None

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores, ddf_logits, last_refined_bbox = self.decoder(embed,
                                                                             refer_bbox,
                                                                             feats,
                                                                             shapes,
                                                                             self.dec_bbox_head,
                                                                             self.dec_score_head,
                                                                             self.query_pos_head,
                                                                             attn_mask=attn_mask,
                                                                             reg_max=getattr(self, 'reg_max', 0),
                                                                             reg_scale=getattr(self, 'reg_scale', 16))
        # [ASTRA-PATCH P9] dec_bboxes 在 reg_max>0 时已是 Integral 解码框 (decoder 内解码),
        # 训练/推理同一路径, 无需额外后处理。
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, ddf_logits, last_refined_bbox
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class RTDETRPriorDecoder(RTDETRDecoder):
    """
    面向红外微小目标的 RT-DETR 解码器。

    与原版 RTDETRDecoder 的唯一区别是 anchor 先验尺寸:原版对所有层级统一使用
    grid_size=0.05 并按 2**i 放大,对 <10x10 像素的红外小目标而言初始框过大,
    回归起点离 GT 太远,导致召回偏低。本类改为"每个 anchor 覆盖一个 stride 单位"。

    [ASTRA-PATCH P7] 相对原实现的两点修正:

    1. 原实现把先验硬编码为 [1/80, 1/40, 1/20],这等价于"假设输入恰好是 640x640"。
       一旦 imgsz != 640 或输入非正方形,先验就和真实 stride 脱钩了。
       这里改为直接由该层特征图尺寸推导 (1/w, 1/h) —— 在 640x640 下与原值
       [1/80, 1/40, 1/20] 完全相同(640/8=80, 640/16=40, 640/32=20),
       因此不影响已有权重与已发表结果的复现,但对任意分辨率都成立。

    2. 原版父类 _generate_anchors 里 `valid_WH = torch.tensor([h, w])` 与
       `grid_xy = stack([grid_x, grid_y])` 的顺序不匹配(用 h 归一化 x、用 w 归一化 y),
       在非正方形特征图上是错的。本类使用正确的 [w, h] 顺序。

    子类通过 `anchor_scale` 统一缩放先验大小。
    """

    anchor_scale = 1.0  # 1.0 => 每个 anchor 覆盖 1 个 stride 单位

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_anchors(self, shapes, grid_size=None, dtype=torch.float32, device='cpu', eps=1e-3):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)              # (h, w, 2) 顺序为 (x, y)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)  # 与 (x, y) 对齐
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH        # (1, h, w, 2)

            if grid_size is None:
                # 分辨率无关的 stride 对齐先验;640x640 下等于 1/80, 1/40, 1/20
                wh = torch.tensor([1.0 / w, 1.0 / h], dtype=dtype, device=device) * self.anchor_scale
                wh = wh.expand_as(grid_xy).contiguous()
            else:
                wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** i)

            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))

        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask
