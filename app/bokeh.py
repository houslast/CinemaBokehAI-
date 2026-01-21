from __future__ import annotations

"""
Bokeh (defocus) baseado em profundidade.

Implementa um desfoque por camadas (depth slicing) e usa um kernel circular (disk blur),
aproximando um efeito de lente (mais "bokeh" que um blur gaussiano comum).

Quando depth está em CUDA, tenta executar blur e composição via PyTorch para acelerar.
"""

from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=64)
def _disk_kernel_np(radius: int) -> np.ndarray:
    r = int(max(1, radius))
    k = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y) <= (r * r)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= float(kernel.sum() + 1e-6)
    return kernel


@lru_cache(maxsize=64)
def _disk_kernel_torch(radius: int, device: str, dtype: str) -> torch.Tensor:
    kernel_np = _disk_kernel_np(radius)
    k = torch.from_numpy(kernel_np).to(device=torch.device(device), dtype=getattr(torch, dtype))
    return k


def _blur_disk_cv2(img_bgr: np.ndarray, radius: int) -> np.ndarray:
    r = int(max(1, radius))
    if r <= 1:
        return img_bgr
    kernel = _disk_kernel_np(r)
    return cv2.filter2D(img_bgr, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)


def _blur_disk_torch(img_01: torch.Tensor, radius: int) -> torch.Tensor:
    r = int(max(1, radius))
    if r <= 1:
        return img_01

    device = str(img_01.device)
    dtype = "float16" if img_01.dtype == torch.float16 else "float32"
    k2d = _disk_kernel_torch(r, device=device, dtype=dtype)
    k = k2d.unsqueeze(0).unsqueeze(0)
    weight = k.repeat(3, 1, 1, 1)
    pad = r
    return F.conv2d(img_01, weight=weight, bias=None, stride=1, padding=pad, groups=3)


def apply_depth_bokeh(
    frame_bgr: np.ndarray,
    depth01: torch.Tensor,
    focus_pos: float,
    bokeh_strength: float,
    near_far_bias: float,
    num_layers: int,
    feather: float,
    prefer_cuda: bool = True,
) -> np.ndarray:
    focus_pos = float(np.clip(focus_pos, 0.0, 1.0))
    bokeh_strength = float(np.clip(bokeh_strength, 0.0, 1.0))
    near_far_bias = float(np.clip(near_far_bias, -1.0, 1.0))
    num_layers = int(max(2, num_layers))
    feather = float(np.clip(feather, 0.0, 1.0))

    h, w = frame_bgr.shape[:2]
    if depth01.ndim != 2:
        raise ValueError("depth01 deve ter shape [H, W].")
    if depth01.shape[0] != h or depth01.shape[1] != w:
        raise ValueError("depth01 deve estar na mesma resolução do frame.")

    max_radius = int(2 + bokeh_strength * 22)

    use_torch = bool(prefer_cuda and depth01.device.type == "cuda")

    if use_torch:
        img = torch.from_numpy(frame_bgr[:, :, ::-1].copy()).to(device=depth01.device)
        img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
        img = img.to(dtype=torch.float16 if depth01.dtype == torch.float16 else torch.float32) / 255.0

        out = img
        d = depth01.unsqueeze(0).unsqueeze(0)

        edges = torch.linspace(0.0, 1.0, steps=num_layers + 1, device=depth01.device, dtype=depth01.dtype)
        sigma = (1.0 / num_layers) * (0.25 + feather * 1.25)

        for i in range(num_layers):
            a = edges[i]
            b = edges[i + 1]
            center = (a + b) * 0.5

            center_val = float(center.item())
            dist = center_val - focus_pos
            bias = (1.0 + near_far_bias) if dist > 0.0 else (1.0 - near_far_bias)
            radius = int(max(1, round(abs(dist) * max_radius * 2.0 * bias)))
            if radius <= 1:
                continue

            blurred = _blur_disk_torch(img, radius=radius)

            band = torch.exp(-0.5 * ((d - center) / max(1e-6, sigma)) ** 2)
            band = torch.clamp(band, 0.0, 1.0)
            out = out * (1.0 - band) + blurred * band

        out = torch.clamp(out, 0.0, 1.0)
        out_np = (out.squeeze(0).permute(1, 2, 0).detach().to("cpu").numpy() * 255.0).astype(np.uint8)
        return out_np[:, :, ::-1]

    out = frame_bgr.copy()
    depth_cpu = depth01.detach().to("cpu").float().numpy()
    edges = np.linspace(0.0, 1.0, num_layers + 1, dtype=np.float32)
    sigma = (1.0 / num_layers) * (0.25 + feather * 1.25)

    for i in range(num_layers):
        a = edges[i]
        b = edges[i + 1]
        center = float((a + b) * 0.5)

        dist = center - focus_pos
        bias = (1.0 + near_far_bias) if dist > 0 else (1.0 - near_far_bias)
        radius = int(max(1, round(abs(dist) * max_radius * 2.0 * bias)))
        if radius <= 1:
            continue

        blurred = _blur_disk_cv2(frame_bgr, radius=radius)

        band = np.exp(-0.5 * ((depth_cpu - center) / max(1e-6, sigma)) ** 2).astype(np.float32)
        band = np.clip(band, 0.0, 1.0)
        band3 = np.repeat(band[:, :, None], 3, axis=2)

        out = (out.astype(np.float32) * (1.0 - band3) + blurred.astype(np.float32) * band3).astype(np.uint8)

    return out
