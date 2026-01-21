from __future__ import annotations

"""
Ajustes de cor com máscara por profundidade.

Cada ajuste (contraste, brilho, saturação, vibrance) recebe:
- intensidade: quanto aplicar
- depth_pos: onde aplicar (0=frente, 1=fundo)
- width: largura/softness da máscara ao redor da posição
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthAdjust:
    intensity: float
    depth_pos: float
    width: float


@dataclass(frozen=True)
class ColorConfig:
    contrast: DepthAdjust
    brightness: DepthAdjust
    saturation: DepthAdjust
    vibrance: DepthAdjust


def _depth_weight(depth01: np.ndarray, pos: float, width: float) -> np.ndarray:
    pos = float(np.clip(pos, 0.0, 1.0))
    width = float(np.clip(width, 0.01, 1.0))
    sigma = 0.08 + 0.45 * width
    w = np.exp(-0.5 * ((depth01 - pos) / sigma) ** 2).astype(np.float32)
    return np.clip(w, 0.0, 1.0)


def apply_depth_color(
    frame_bgr: np.ndarray,
    depth01_cpu: np.ndarray,
    cfg: ColorConfig,
) -> np.ndarray:
    if depth01_cpu.ndim != 2:
        raise ValueError("depth01_cpu deve ter shape [H, W].")
    if depth01_cpu.shape[:2] != frame_bgr.shape[:2]:
        raise ValueError("depth01_cpu deve estar na mesma resolução do frame.")

    img = frame_bgr.astype(np.float32) / 255.0

    w_c = _depth_weight(depth01_cpu, cfg.contrast.depth_pos, cfg.contrast.width)
    w_b = _depth_weight(depth01_cpu, cfg.brightness.depth_pos, cfg.brightness.width)
    w_s = _depth_weight(depth01_cpu, cfg.saturation.depth_pos, cfg.saturation.width)
    w_v = _depth_weight(depth01_cpu, cfg.vibrance.depth_pos, cfg.vibrance.width)

    w_c3 = w_c[:, :, None]
    w_b3 = w_b[:, :, None]

    contrast = float(np.clip(cfg.contrast.intensity, -1.0, 1.0))
    if abs(contrast) > 1e-6:
        pivot = 0.5
        factor = 1.0 + contrast * 1.25
        img = img * (1.0 - w_c3) + np.clip((img - pivot) * factor + pivot, 0.0, 1.0) * w_c3

    brightness = float(np.clip(cfg.brightness.intensity, -1.0, 1.0))
    if abs(brightness) > 1e-6:
        img = img * (1.0 - w_b3) + np.clip(img + brightness * 0.35, 0.0, 1.0) * w_b3

    hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    saturation = float(np.clip(cfg.saturation.intensity, -1.0, 1.0))
    if abs(saturation) > 1e-6:
        s2 = np.clip(s * (1.0 + saturation * 1.5), 0.0, 1.0)
        s = s * (1.0 - w_s) + s2 * w_s

    vibrance = float(np.clip(cfg.vibrance.intensity, -1.0, 1.0))
    if abs(vibrance) > 1e-6:
        boost = (1.0 - s)
        s2 = np.clip(s + vibrance * 0.8 * boost, 0.0, 1.0)
        s = s * (1.0 - w_v) + s2 * w_v

    hsv[:, :, 0] = h
    hsv[:, :, 1] = np.clip(s * 255.0, 0.0, 255.0)
    hsv[:, :, 2] = np.clip(v * 255.0, 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out
