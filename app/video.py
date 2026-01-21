from __future__ import annotations

"""
Utilitários de vídeo (OpenCV).

- Salva upload em arquivo temporário
- Lê metadados básicos do vídeo
- Lê frame específico para scrub (preview)
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def save_uploaded_video_to_temp(uploaded_bytes: bytes, suffix: str) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "fx_depth_bokeh"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload{suffix}"
    tmp_path.write_bytes(uploaded_bytes)
    return str(tmp_path)


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Falha ao ler metadados do vídeo.")

    if fps <= 0:
        fps = 30.0

    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def read_frame_bgr(path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Não foi possível ler o frame do vídeo.")
    return frame


def iter_frames_bgr(path: str) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
    finally:
        cap.release()


def ensure_even_dimensions(width: int, height: int) -> Tuple[int, int]:
    # H.264/H.265 tipicamente exigem dimensões pares para compatibilidade ampla.
    return width - (width % 2), height - (height % 2)
