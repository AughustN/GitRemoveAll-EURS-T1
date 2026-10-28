"""Track 1 inference entry point (team GitRemoveAll, testing-phase submission #19059).

The organizers call   main(model_dir, input_dir, output_dir, device).

Reproduces make_submission.py of notebook F8_1022_submit_test_v3 (Kaggle, Tesla T4) step by step:
  1. read_eval: decode one image, RGB, PIL BILINEAR down to longest side 1024 (evaluator formula); all models run here.
  2. segmentation: frozen DINOv3 ViT-L/16 + DPT-lite, TTA scales (0.75, 1.0) x horizontal flip, mean of softmax,
     fp16 autocast on CUDA, argmax (logit bias 0) -> seg_masks/<stem>.png (uint8 labels 0..4).
  3. restoration: RestoreNet (ConvNeXt-T, cond='est'); the water type / level are read from the file name
     (<scene>_<type>_<level>.png, declared as "Meta" on the leaderboard). If a file name does not carry them,
     the network falls back to its own type/level estimate (cond=None), which is the same model without metadata.
  4. 8-bit conversion with rounding, then the upload-size step of the leaderboard zip: values rounded to even
     numbers (shrink_submission.py, step 2) -> restored/<stem>.png.
Every output depends only on its own input image.
"""
import glob
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image

from src.infer import quantize, read_eval, restore_full, seg_probs, to_u8
from src.restore_big import build_restore, make_cond
from src.seg_model import ViTSeg

SEG_CKPT = "seg_dinov3_vitl16_ext_all.pt"
SEG_CONFIG_DIR = "dinov3_vitl16_config"      # config.json of facebook/dinov3-vitl16-pretrain-lvd1689m
RESTORE_CKPT = "restore_big_all.pt"
SCALES, FLIP = (0.75, 1.0), True
BIAS = [0.0, 0.0, 0.0, 0.0, 0.0]             # background, coral, fish, elasmobranch, other
QUANT_STEP = 2
META_RE = re.compile(r"_(\d+)_(\d+(?:\.\d+)?)$")  # <scene>_<type>_<level>


def _find(model_dir, name):
    hits = sorted(glob.glob(os.path.join(model_dir, "**", name), recursive=True))
    assert hits, f"{name} not found under {model_dir}"
    return hits[0]


def _load_seg_state(model, sd):
    """The checkpoint was saved with transformers 5.x, where the DINOv3 blocks live under `bb.model.layer.N`;
    transformers 4.57 (the version that runs on PyTorch 2.4) names them `bb.layer.N`. Same tensors, renamed prefix."""
    want = set(model.state_dict())
    for a, b in (("bb.model.layer.", "bb.layer."), ("bb.layer.", "bb.model.layer.")):
        sd = {(b + k[len(a):] if k.startswith(a) and k not in want and b + k[len(a):] in want else k): v
              for k, v in sd.items()}
    model.load_state_dict(sd)


def _cond(model, stem):
    if getattr(model, "cond", "none") != "est":
        return None
    m = META_RE.search(stem)
    return make_cond(int(m.group(1)), float(m.group(2))) if m else None


def _restore_fit(model, img, device, cond):
    """restore_full at scoring resolution; on CUDA OOM shrink the input by 0.75 (INTER_AREA) and upscale back."""
    h, w = img.shape[:2]
    x, s = img, 1.0
    while True:
        try:
            out = restore_full(model, x, device, cond=cond)
            break
        except torch.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        s *= 0.75
        assert s > 0.05, "OOM even at a very small restore input"
        x = cv2.resize(img, (max(32, round(w * s)), max(32, round(h * s))), interpolation=cv2.INTER_AREA)
    if s < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def main(model_dir, input_dir, output_dir, device):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    ck = torch.load(_find(model_dir, SEG_CKPT), map_location="cpu", weights_only=False)
    seg = ViTSeg(os.path.dirname(_find(model_dir, os.path.join(SEG_CONFIG_DIR, "config.json"))))
    _load_seg_state(seg, ck["model"])
    seg = seg.to(device).eval()
    del ck
    rest = build_restore(torch.load(_find(model_dir, RESTORE_CKPT), map_location="cpu", weights_only=False), device)

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)
    bt = torch.tensor(BIAS, device=device).view(5, 1, 1)
    files = sorted(f for f in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
                   if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        img = read_eval(f)
        p = seg_probs(seg, img, device, SCALES, FLIP, seg.mult)
        mask = (torch.log(p.clamp(min=1e-6)) + bt).argmax(0).byte().cpu().numpy()
        Image.fromarray(mask, mode="L").save(os.path.join(output_dir, "seg_masks", stem + ".png"))
        out = quantize(to_u8(_restore_fit(rest, img, device, _cond(rest, stem))), QUANT_STEP)
        Image.fromarray(out).save(os.path.join(output_dir, "restored", stem + ".png"))
