"""Inference helpers, copied unchanged from the code that produced the leaderboard submission
(train_common.py / make_submission.py / shrink_submission.py of notebook F8_1022_submit_test_v3)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

N_CLASSES = 5


def read_eval(path, max_side=1024):
    """Degraded image -> scoring resolution (PIL BILINEAR on uint8, evaluator's scale_to_max_side formula)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = max_side / max(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def landscape(*arrs):
    """Portrait image -> rotate by 90 degrees."""
    if arrs[0].shape[0] > arrs[0].shape[1]:
        return tuple(np.ascontiguousarray(np.rot90(a)) for a in arrs), True
    return arrs, False


def _pad_to(x, mult):
    H, W = x.shape[-2:]
    ph, pw = (mult - H % mult) % mult, (mult - W % mult) % mult
    return (F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x), (H, W)


@torch.no_grad()
def seg_probs(model, img, device, scales=(1.0,), flip=False, mult=32, out_scale=1.0, amp=True):
    """Softmax (5, H', W') for a numpy image (H, W, 3) in [0, 1]; portrait images are rotated and rotated back."""
    (im,), rot = landscape(img)
    h, w = im.shape[:2]
    oh, ow = max(1, round(h * out_scale)), max(1, round(w * out_scale))
    acc = torch.zeros(N_CLASSES, oh, ow, device=device)
    n = 0
    for s in scales:
        x = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR) if s != 1 else im
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
        xp, (H, W) = _pad_to(x, mult)
        views = [xp, xp.flip(-1)] if flip else [xp]
        for k, v in enumerate(views):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and amp):
                lg = model(v)
            lg = lg.float()
            if k == 1:
                lg = lg.flip(-1)
            lg = lg[..., :H, :W]
            acc += F.interpolate(lg.softmax(1), size=(oh, ow), mode="bilinear", align_corners=False)[0]
            n += 1
    p = acc / n
    if rot:
        p = torch.rot90(p, -1, dims=(1, 2))
    return p


@torch.no_grad()
def restore_full(model, img, device, mult=32, cond=None):
    """Restoration model at scoring resolution; returns numpy (H, W, 3) in [0, 1]."""
    (im,), rot = landscape(img)
    x = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))[None].to(device)
    xp, (H, W) = _pad_to(x, mult)
    r, _ = model(xp, heads=("restore",)) if cond is None else model(xp, heads=("restore",), cond=cond.to(device))
    r = r[0, :, :H, :W].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(np.rot90(r, -1)) if rot else r


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def quantize(a, step=2):
    """Upload-size step of the leaderboard zip (shrink_submission.py): restored values rounded to multiples of `step`."""
    a = a.astype(np.int32)
    return np.clip((a + step // 2) // step * step, 0, 255).astype(np.uint8)
