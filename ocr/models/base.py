from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import numpy as np

BBox = Sequence[Sequence[float]]
RawDetection = Tuple[BBox, str, float]


class OCRModel(Protocol):
    def warmup(self) -> None: ...

    def recognize(self, image: np.ndarray) -> List[RawDetection]: ...
