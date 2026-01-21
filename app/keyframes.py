from __future__ import annotations

"""
Keyframes para animação do foco.

Armazena parâmetros por frame e interpola automaticamente entre keyframes vizinhos.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, TypeVar


@dataclass(frozen=True)
class KeyframeParams:
    focus_pos: float
    bokeh_strength: float
    near_far_bias: float
    num_layers: float
    feather: float

    contrast_intensity: float
    contrast_pos: float
    contrast_width: float

    brightness_intensity: float
    brightness_pos: float
    brightness_width: float

    saturation_intensity: float
    saturation_pos: float
    saturation_width: float

    vibrance_intensity: float
    vibrance_pos: float
    vibrance_width: float


T = TypeVar("T")


def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * float(a) + t * float(b)


class KeyframeStore:
    def __init__(self) -> None:
        self._items: Dict[int, KeyframeParams] = {}

    def frames(self) -> List[int]:
        return sorted(self._items.keys())

    def has(self, frame_index: int) -> bool:
        return frame_index in self._items

    def get(self, frame_index: int) -> Optional[KeyframeParams]:
        return self._items.get(frame_index)

    def upsert(self, frame_index: int, params: KeyframeParams) -> None:
        self._items[int(frame_index)] = params

    def delete(self, frame_index: int) -> None:
        self._items.pop(int(frame_index), None)

    def interpolate(self, frame_index: int, default: KeyframeParams) -> KeyframeParams:
        if not self._items:
            return default

        idx = int(frame_index)
        if idx in self._items:
            return self._items[idx]

        frames = self.frames()
        left = None
        right = None
        for f in frames:
            if f < idx:
                left = f
            elif f > idx and right is None:
                right = f
                break

        if left is None:
            return self._items[frames[0]]
        if right is None:
            return self._items[frames[-1]]

        a = self._items[left]
        b = self._items[right]
        t = (idx - left) / max(1, (right - left))

        values = {}
        for f in fields(KeyframeParams):
            name = f.name
            values[name] = _lerp(getattr(a, name), getattr(b, name), t)
        return KeyframeParams(**values)
