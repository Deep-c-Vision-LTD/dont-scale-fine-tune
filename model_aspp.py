"""
ASPP-readout saliency model: frozen DINOv2-base + ASPP decoder.

Replaces the 3-conv readout with an Atrous Spatial Pyramid Pooling module
(parallel dilated convs + global-avg-pool branch) so the decoder captures
global context on the 16x16 patch grid — the same mechanism that gives
MSI-Net crisp, peaked maps.

Trainable params: ASPP + projection + center bias (~1.5M).
Backbone: fully frozen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Dinov2Config, Dinov2Model


class ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU()))
        # global context branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.proj(torch.cat(outs, dim=1))


class ASPPSaliency(nn.Module):
    def __init__(self, backbone="facebook/dinov2-base", aspp_ch=128,
                 dilations=(1, 3, 6, 12), use_center_bias=True, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone)
        else:
            cfg = Dinov2Config(image_size=518)
            self.backbone = Dinov2Model(cfg)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        C = self.backbone.config.hidden_size  # 768 for base
        self.aspp = ASPPModule(C, aspp_ch, dilations)
        self.head = nn.Conv2d(aspp_ch, 1, 1)
        self.use_center_bias = use_center_bias
        if use_center_bias:
            self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16))

    @torch.no_grad()
    def _features(self, x):
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        tok = out.last_hidden_state[:, 1:, :]
        B, N, C = tok.shape
        h = w = int(round(N ** 0.5))
        return tok.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, x):
        H, W = x.shape[-2:]
        f = self._features(x)
        s = self.head(self.aspp(f))
        if self.use_center_bias:
            cb = F.interpolate(self.center_bias, size=s.shape[-2:],
                               mode="bilinear", align_corners=False)
            s = s + cb
        return F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
