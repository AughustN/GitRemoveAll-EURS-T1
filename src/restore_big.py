"""RestoreNet – nhánh restore lớn cho Track 1 (không giới hạn compute).

Ý tưởng ghép từ 3 bài (plan/05_RESTORE_PAPERS.md):
- Trường affine màu theo vị trí như EURS-Lite (I = J·t + B(1 − t) ⇒ J là affine của I), nhưng dự đoán từ encoder
  ConvNeXt có pretrain ImageNet thay vì MobileNetV2 0.6M train từ đầu → tổng quát tốt hơn trên scene mới.
- DarkIR (CVPR 2025): sửa sáng ở độ phân giải thấp + loss phụ trên output H/8 (L_lol); decoder full-res dùng
  depthwise dilated 1/4/9 + gating (Di-SpAM / GatedFFN) để giữ độ nét (sharpness, SSIM) thay cho unsharp mask.
- HVI-CIDNet (CVPR 2025): tách màu / cường độ → loss cosine màu tách riêng khỏi L1 (đã có trong restore_loss).

forward(x) trả (restored (B,3,H,W) [0,1], aux (B,3,H/8,W/8)) – cùng giao diện (r, _) với EURS-Lite nên
train_common.restore_full dùng chung được. H, W phải chia hết cho 32 (restore_full tự pad).

Điều kiện hoá theo loại nước (cond, ý tưởng Osmosis ECCV 2024 + InstructIR ECCV 2024):
- vector điều kiện c = [xác suất 10 type, level/100] (COND_DIM = 11) → FiLM trên feature FPN và full-res + bù vào
  tham số affine; mọi lớp FiLM khởi tạo 0 → lúc đầu y hệt bản không điều kiện (nạp checkpoint cũ với strict=False).
- cond="est"   : head trên feature H/32 tự ước lượng type (10 lớp) + level (hồi quy) từ ảnh → không cần metadata.
- cond="oracle": c lấy từ type/level thật (tên file) → mức trần để so với "est".
- cond="none"  : như cũ.

Khuếch đại input (amp=True, ý tưởng Learning-to-See-in-the-Dark CVPR 2018 + DarkIR CVPR 2025): ảnh "blackout" chỉ còn
1–7 mức xám (chênh 1/255) → sau chuẩn hoá ImageNet encoder gần như không thấy. Thêm bản kéo giãn theo từng ảnh, từng kênh
xs = (x − min)/max(max − min, 1/255) làm 3 kênh phụ cho encoder (stem ConvNeXt 3 → 6 kênh) và cho stem full-res;
trọng số kênh phụ khởi tạo 0 → lúc đầu y hệt bản không amp, pretrain giữ nguyên.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
N_TYPES, LEVEL_MAX = 10, 100.0
COND_DIM = N_TYPES + 1


def make_cond(wtype, level):
    """type (1–10) + level (5–100) thật → vector điều kiện (B, 11): one-hot type, level/100."""
    wtype, level = torch.as_tensor(wtype).long().view(-1), torch.as_tensor(level).float().view(-1)
    return torch.cat([F.one_hot((wtype - 1).clamp(0, N_TYPES - 1), N_TYPES).float(), (level / LEVEL_MAX)[:, None]], 1)


def stretch(x):
    """Kéo giãn từng ảnh, từng kênh về [0, 1]; ảnh hằng số → 0. Tính fp32."""
    xf = x.float()
    lo, hi = xf.amin((2, 3), keepdim=True), xf.amax((2, 3), keepdim=True)
    return ((xf - lo) / (hi - lo).clamp(min=1 / 255)).to(x.dtype)


def widen_conv(conv, extra):
    """Conv2d cin → cin + extra; trọng số cũ giữ nguyên, kênh thêm = 0."""
    new = nn.Conv2d(conv.in_channels + extra, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def zero_linear(cin, cout):
    m = nn.Linear(cin, cout)
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


def film(x, prm):
    """x (B,C,H,W), prm (B,2C) = [scale, shift] → x·(1 + scale) + shift."""
    s, b = prm.chunk(2, 1)
    return x * (1 + s[:, :, None, None]) + b[:, :, None, None]


def up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def conv_gn(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w, self.b, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.w[:, None, None] + self.b[:, None, None]


class DiBlock(nn.Module):
    """Block decoder kiểu DarkIR: Di-SpAM (DW 3×3 dilation 1/4/9 cộng lại → SimpleGate → channel attention)
    + GatedFFN; cả hai nhánh residual có hệ số học được khởi tạo 0 (như NAFNet)."""

    def __init__(self, c, dil=(1, 4, 9), ffn=2):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.ModuleList([nn.Conv2d(2 * c, 2 * c, 3, padding=d, dilation=d, groups=2 * c) for d in dil])
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.pw2 = nn.Conv2d(c, c, 1)
        self.f1, self.f2 = nn.Conv2d(c, 2 * ffn * c, 1), nn.Conv2d(ffn * c, c, 1)
        self.g1, self.g2 = nn.Parameter(torch.zeros(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.pw1(self.n1(x))
        y = sum(d(y) for d in self.dw)
        a, b = y.chunk(2, 1)
        y = a * b
        y = self.pw2(y * self.sca(y))
        x = x + self.g1 * y
        a, b = self.f1(self.n2(x)).chunk(2, 1)
        return x + self.g2 * self.f2(a * b)


def apply_affine(p, x):
    """p (B,12,H,W) = ma trận 3×3 (hàng) + bias 3; x (B,3,H,W)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([p[:, 3 * c:3 * c + 1] * r + p[:, 3 * c + 1:3 * c + 2] * g + p[:, 3 * c + 2:3 * c + 3] * b
                      for c in range(3)], 1) + p[:, 9:12]


CI_EPS = 1e-3
CI_DIM = 3 + 12  # đường cong cường độ (log gain, log gamma, offset) + affine 3×3 + bias trên chroma


def apply_ci(q, x):
    """Hiệu chỉnh tách màu / cường độ (ý tưởng HVI-CIDNet, CVPR 2025) chồng lên output affine x (B,3,H,W).
    Y = max(R,G,B) (cường độ kiểu HSV/HVI), c = x / Y (chroma, kênh lớn nhất = 1).
    Y' = exp(q0)·Y^exp(q1) + q2 : đường cong phi tuyến cho vùng tối (affine RGB không biểu diễn được gamma);
    c' = (I + M)·c + d với M, d = q3..q14 : sửa ám màu tách khỏi độ sáng. Output = c'·Y'.
    q (B,15,H,W) = 0 → output = x (đồng nhất). Tính fp32 (pow / chia ổn định dưới autocast)."""
    dt, x, q = x.dtype, x.float(), q.float()
    xp = x.clamp(min=0)  # affine có thể ra giá trị âm → c = x/Y nổ; phần âm đi thẳng qua (residual)
    Y = xp.amax(1, keepdim=True).clamp(min=CI_EPS)
    c = xp / Y
    Y2 = torch.exp(q[:, 0:1]) * Y.pow(torch.exp(q[:, 1:2])) + q[:, 2:3]
    c2 = c + apply_affine(q[:, 3:15], c)  # (I + M)·c + d
    return (c2 * Y2 + (x - xp)).to(dt)


class RestoreNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, d=128, c_full=32, n_full=3, cond="none", ci=False,
                 amp=False):
        super().__init__()
        assert cond in ("none", "est", "oracle"), cond
        self.cond, self.ci, self.amp = cond, bool(ci), bool(amp)
        import timm
        self.enc = timm.create_model(backbone, features_only=True, pretrained=pretrained, out_indices=(0, 1, 2, 3))
        ch = self.enc.feature_info.channels()  # strides 4, 8, 16, 32
        if self.amp:  # conv đầu tiên nhận RGB (stem ConvNeXt) → thêm 3 kênh xs
            name = next(n for n, m in self.enc.named_modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)
            parent, attr = self.enc.get_submodule(name.rpartition(".")[0]) if "." in name else self.enc, name.rpartition(".")[2]
            setattr(parent, attr, widen_conv(getattr(parent, attr), 3))
        self.register_buffer("mean", MEAN.clone(), persistent=False)
        self.register_buffer("std", STD.clone(), persistent=False)
        # FPN top-down tới H/4
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in ch])
        self.smooth = nn.ModuleList([conv_gn(d, d) for _ in ch[:3]])
        # trường affine: local ở H/8 + global từ H/32, khởi tạo = biến đổi đồng nhất
        self.aff_local = nn.Sequential(conv_gn(d, 64), nn.Conv2d(64, 12, 1))
        self.aff_glob = nn.Sequential(nn.Linear(ch[3], 128), nn.GELU(), nn.Linear(128, 12))
        for m in (self.aff_local[-1], self.aff_glob[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.register_buffer("eye", torch.tensor([1., 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]).view(1, 12, 1, 1),
                             persistent=False)
        # refine full-res: [I, affine(I), feature H/4 nội suy] → DiBlock × n → residual 3 kênh (zero-init)
        self.f4 = nn.Conv2d(d, c_full, 1)
        self.stem = nn.Conv2d(6, c_full, 3, padding=1)
        if self.amp:
            self.stem = widen_conv(self.stem, 3)
        self.full = nn.Sequential(*[DiBlock(c_full) for _ in range(n_full)])
        self.out = nn.Conv2d(c_full, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.backbone_name = backbone
        if self.ci:  # tham số tách màu / cường độ ở H/8 (local) + toàn cục H/32, zero-init → đồng nhất
            self.ci_local = nn.Sequential(conv_gn(d, 64), nn.Conv2d(64, CI_DIM, 1))
            self.ci_glob = nn.Sequential(nn.Linear(ch[3], 128), nn.GELU(), nn.Linear(128, CI_DIM))
            for m in (self.ci_local[-1], self.ci_glob[-1]):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        if cond != "none":
            if cond == "est":  # type logits (10) + level (sigmoid → [0,1]) từ feature toàn cục H/32
                self.est = nn.Sequential(nn.Linear(ch[3], 256), nn.GELU(), nn.Linear(256, N_TYPES + 1))
            self.film_p = zero_linear(COND_DIM, 2 * d)       # FiLM chung cho 3 tầng FPN
            self.film_h = zero_linear(COND_DIM, 2 * c_full)  # FiLM cho feature full-res
            self.cond_aff = zero_linear(COND_DIM, 12)        # bù tham số affine theo loại nước
            if self.ci:
                self.cond_ci = zero_linear(COND_DIM, CI_DIM)  # đường cong cường độ / chroma theo loại nước

    def estimate(self, f3):
        """f3 (B,C,h,w) → (type_logits (B,10), level (B,) trong [0,1])."""
        o = self.est(f3.mean((2, 3)))
        return o[:, :N_TYPES], torch.sigmoid(o[:, N_TYPES])

    def encode(self, x, xs=None):
        """Feature encoder 4 tầng (strides 4–32); amp → ghép thêm bản kéo giãn xs."""
        z = (x - self.mean) / self.std
        if self.amp:
            z = torch.cat([z, ((stretch(x) if xs is None else xs) - self.mean) / self.std], 1)
        return self.enc(z)

    def backbone_parameters(self):
        return self.enc.parameters()

    def forward(self, x, heads=None, cond=None, return_est=False, mix=None):
        """cond (B,11) (make_cond) ghi đè vector điều kiện – bắt buộc khi self.cond == 'oracle'; với 'est' truyền
        cond = điều kiện thật để đo mức trần trên cùng model. return_est → thêm (type_logits, level) vào output.
        mix = (cond_thật (B,11), w (B,1)) – chỉ 'est' khi train: cond = w·thật + (1 − w)·ước lượng (teacher forcing)."""
        H, W = x.shape[-2:]
        xs = stretch(x) if self.amp else None
        f = self.encode(x, xs)
        est = None
        if self.cond == "est":
            est = self.estimate(f[3])
            if cond is None:
                cond = torch.cat([est[0].softmax(1), est[1][:, None]], 1)
                if mix is not None:
                    cond = mix[1] * mix[0] + (1 - mix[1]) * cond
        elif self.cond == "oracle" and cond is None:
            raise ValueError("RestoreNet(cond='oracle') cần cond = make_cond(type, level)")
        if self.cond != "none":
            cond = cond.to(x.dtype)
        p = [l(t) for l, t in zip(self.lat, f)]
        for i in range(3, 0, -1):
            p[i - 1] = p[i - 1] + up(p[i], p[i - 1].shape[-2:])
        p = [s(t) for s, t in zip(self.smooth, p[:3])]
        if self.cond != "none":
            fp = self.film_p(cond)
            p = [film(t, fp) for t in p]
        prm = self.aff_local(p[1]) + self.aff_glob(f[3].mean((2, 3)))[:, :, None, None] + self.eye  # (B,12,H/8,W/8)
        if self.cond != "none":
            prm = prm + self.cond_aff(cond)[:, :, None, None]
        self.last_prm = prm if self.training else None  # train_restore --field_w: giám sát trường affine trực tiếp
        x8 = F.avg_pool2d(x, 8)
        aux = apply_affine(prm, x8)                     # output sửa màu ở H/8 (loss phụ kiểu DarkIR)
        color = apply_affine(up(prm, (H, W)), x)
        if self.ci:
            q = self.ci_local(p[1]) + self.ci_glob(f[3].mean((2, 3)))[:, :, None, None]  # (B,15,H/8,W/8)
            if self.cond != "none":
                q = q + self.cond_ci(cond)[:, :, None, None]
            aux, color = apply_ci(q, aux), apply_ci(up(q, (H, W)), color)
        h = self.stem(torch.cat([x, color] + ([xs] if self.amp else []), 1)) + up(self.f4(p[0]), (H, W))
        if self.cond != "none":
            h = film(h, self.film_h(cond))
        out = color + self.out(self.full(h))
        if return_est:
            return out.clamp(0, 1), aux, est
        return out.clamp(0, 1), aux


@torch.no_grad()
def oracle_field(x, y, k=32, lam=1e-6, clip=30.0):
    """Trường affine 'đáp án' (tham số fit bằng GT): affine 3×3 + bias bình phương tối thiểu trong từng ô k×k của cặp
    (x suy giảm, y GT), xếp theo layout của apply_affine, nội suy bilinear về lưới H/8 của RestoreNet.
    ceiling_probe.py (12/10): affine theo ô 32 px đạt 33.3 dB trên val trong khi model chỉ 27.4 ≈ affine toàn ảnh (27.6)
    → model không học được phần thay đổi theo vị trí; giám sát thẳng θ cho tín hiệu dày hơn loss trên ảnh.
    x, y (B,3,H,W), H, W chia hết cho k và 8 → (B,12,H/8,W/8), tính fp32."""
    x, y = x.float(), y.float()
    B, _, H, W = x.shape
    ux = F.unfold(x, k, stride=k).view(B, 3, k * k, -1).permute(0, 3, 2, 1)   # (B,L,N,3)
    uy = F.unfold(y, k, stride=k).view(B, 3, k * k, -1).permute(0, 3, 2, 1)
    X = torch.cat([ux, torch.ones_like(ux[..., :1])], -1)                     # (B,L,N,4)
    # ridge kéo về biến đổi đồng nhất (không về 0): 3 kênh màu tương quan mạnh → bình phương tối thiểu suy biến,
    # kéo về 0 làm θ lệch xa đồng nhất; kéo về I thì x = y cho đúng θ = I và ô phẳng / tối giữ gần I
    I = lam * k * k * torch.eye(4, device=x.device)
    M0 = torch.eye(4, 3, device=x.device)
    M = torch.linalg.solve(X.transpose(-1, -2) @ X + I, X.transpose(-1, -2) @ uy + I @ M0)  # (B,L,4,3): M[i, c]
    p = torch.cat([M[..., :3, c] for c in range(3)] + [M[..., 3, :]], -1)     # (B,L,12) = layout apply_affine
    p = p.clamp(-clip, clip).permute(0, 2, 1).reshape(B, 12, H // k, W // k)
    return F.interpolate(p, size=(H // 8, W // 8), mode="bilinear", align_corners=False)


def build_restore(ck, device="cpu"):
    """Dựng model restore từ checkpoint (ck['arch'] = 'lite' | 'big'); trả model eval trên device."""
    arch = ck.get("arch", "lite")
    if arch == "big":
        net = RestoreNet(pretrained=False, **ck.get("arch_args", {}))
    else:
        from eurs_lite import EURSLite
        net = EURSLite(dynamic=True)
    net.load_state_dict(ck["model"])
    return net.to(device).eval()


if __name__ == "__main__":
    x = torch.rand(2, 3, 256, 320)
    c = make_cond([1, 9], [5, 100])
    for mode in ("none", "est", "oracle"):
        for ci in (False, True):
            torch.manual_seed(0)
            m = RestoreNet(pretrained=False, cond=mode, ci=ci, amp=ci)  # amp thử chung với ci cho gọn
            n = sum(p.numel() for p in m.parameters())
            r, a, e = m(x, cond=c if mode == "oracle" else None, return_est=True)
            print(mode, "ci+amp" if ci else "      ", f"params {n / 1e6:.3f}M", tuple(r.shape), tuple(a.shape),
                  None if e is None else tuple(e[0].shape), f"|out-x| {float((r - x).abs().mean()):.6f}")
    q = torch.zeros(2, CI_DIM, 4, 5)
    print("apply_ci(0) = đồng nhất:", float((apply_ci(q, x[..., :4, :5]) - x[..., :4, :5]).abs().max()))
    torch.manual_seed(0)
    m0 = RestoreNet(pretrained=False, cond="est")
    m1 = RestoreNet(pretrained=False, cond="est", amp=True)
    sd = m1.state_dict()
    for k, v in m0.state_dict().items():
        sd[k] = v if sd[k].shape == v.shape else torch.cat([v, torch.zeros_like(sd[k][:, v.shape[1]:])], 1)
    m1.load_state_dict(sd)
    for net in (m0, m1):  # đổi head để output khác x → kiểm tra amp với kênh phụ = 0 ra y hệt bản thường
        net.eval()
    with torch.no_grad():
        nn.init.normal_(m0.aff_glob[-1].weight, std=0.01); m1.aff_glob[-1].weight.copy_(m0.aff_glob[-1].weight)
        print("amp (kênh phụ = 0) = bản thường:", float((m0(x)[0] - m1(x)[0]).abs().max()))
    print("stretch ảnh hằng số:", float(stretch(torch.zeros(1, 3, 8, 8)).abs().max()),
          "| ảnh 0/1 mức:", stretch(torch.tensor([0., 1 / 255]).view(1, 1, 1, 2).repeat(1, 3, 1, 1))[0, 0].tolist())