"""Segmentation model of the Track 1 submission: frozen DINOv3 ViT-L/16 + DPT-lite head (5 classes).

Same module tree as the training code (teacher_models.ViTTeacher), so the checkpoint loads with strict=True.
The only change: the backbone is built from a bundled config.json (AutoModel.from_config) instead of being
downloaded from the Hugging Face hub; all backbone weights are in the checkpoint itself.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return (x - self.mean) / self.std


def cbr(cin, cout, k=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DPTLite(nn.Module):
    """4 ViT feature levels (all at H/p) -> project to 256 -> fuse -> upsample x2 twice -> logits at H/p*4."""

    def __init__(self, dim, n_cls=N_CLASSES, c=256):
        super().__init__()
        self.proj = nn.ModuleList([cbr(dim, c) for _ in range(4)])
        self.fuse = cbr(4 * c, c, 3)
        self.up1 = cbr(c, c // 2, 3)
        self.up2 = cbr(c // 2, c // 4, 3)
        self.cls = nn.Conv2d(c // 4, n_cls, 1)

    def forward(self, feats):
        x = self.fuse(torch.cat([p(f) for p, f in zip(self.proj, feats)], 1))
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.cls(x)


class ViTSeg(nn.Module):
    def __init__(self, config_dir):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.norm = Normalize()
        self.bb = AutoModel.from_config(AutoConfig.from_pretrained(config_dir))
        cfg = self.bb.config
        self.patch, dim, nblk = cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers
        self.layers = [nblk // 4 - 1, nblk // 2 - 1, 3 * nblk // 4 - 1, nblk - 1]
        self.head = DPTLite(dim)
        self.mult = self.patch

    def feats(self, x):
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        hs = self.bb(pixel_values=x, output_hidden_states=True).hidden_states  # [embeddings, block1, ...]
        # drop CLS + register tokens: keep the last h*w tokens
        return [hs[i + 1][:, -h * w:, :].transpose(1, 2).reshape(B, -1, h, w) for i in self.layers]

    def forward(self, x):
        H, W = x.shape[-2:]
        assert H % self.patch == 0 and W % self.patch == 0, f"H, W must be multiples of the patch size {self.patch}"
        lg = self.head(self.feats(self.norm(x)))
        return F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
