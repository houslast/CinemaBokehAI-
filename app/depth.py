from __future__ import annotations

"""
Inferência de profundidade por frame usando MiDaS / DPT via PyTorch.

- Faz download automático do repositório/pesos via torch.hub.
- Mantém cache do torch.hub dentro de /models/torchhub.
- Suporta CPU ou CUDA (com seleção de GPU).
- Suporta FP16 para acelerar em GPUs compatíveis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def available_devices() -> List[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i} - {name}")
    return devices


def parse_device_label(label: str) -> torch.device:
    if label.strip().lower().startswith("cuda:"):
        prefix = label.split("-")[0].strip()
        return torch.device(prefix)
    if label.strip().lower().startswith("cuda"):
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class DepthConfig:
    """
    Configuração do estimador de depth.

    depth_input_size controla o tamanho máximo (maior lado) usado como entrada do modelo.
    """

    model_type: str
    device: torch.device
    fp16: bool
    depth_input_size: int


class DepthEstimator:
    """
    Wrapper de inferência de depth (normalizado 0..1) para um frame BGR.
    """

    def __init__(self, cfg: DepthConfig) -> None:
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(MODELS_DIR / "torchhub"))

        model_type = self.cfg.model_type
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        model.eval()
        model.to(self.cfg.device)
        if self.cfg.fp16 and self.cfg.device.type == "cuda":
            model.half()

        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resize_keep_aspect(self, w: int, h: int, target: int) -> Tuple[int, int]:
        # Mantém aspecto, limita o maior lado, e arredonda para múltiplos de 32 (comum em backbones).
        target = int(max(128, target))
        scale = float(target) / float(max(w, h))
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        nw = max(32, (nw // 32) * 32)
        nh = max(32, (nh // 32) * 32)
        return nw, nh

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # Preprocess padrão estilo ImageNet (normalização mean/std), entrada RGB float.
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        h, w = frame_rgb.shape[:2]
        nw, nh = self._resize_keep_aspect(w, h, self.cfg.depth_input_size)
        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        x = torch.from_numpy(resized).to(device=self.cfg.device)
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
        x = x.to(dtype=torch.float16 if (self.cfg.fp16 and self.cfg.device.type == "cuda") else torch.float32) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.cfg.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.cfg.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    def predict_depth01(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self.is_loaded:
            self.load()
        assert self._model is not None
        input_tensor = self._preprocess(frame_bgr)

        with torch.inference_mode():
            if self.cfg.fp16 and self.cfg.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = self._model(input_tensor)
            else:
                prediction = self._model(input_tensor)

            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            prediction = F.interpolate(
                prediction,
                size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze(0).squeeze(0)

            d_min = torch.amin(depth)
            d_max = torch.amax(depth)
            depth01 = (depth - d_min) / (d_max - d_min + 1e-6)
            depth01 = torch.clamp(depth01, 0.0, 1.0)
            return depth01
