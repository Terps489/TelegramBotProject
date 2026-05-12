"""Структура данных результата распознавания."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class OCRBox:
    """Один распознанный фрагмент: текст, уверенность и bbox."""

    text: str
    confidence: float
    bbox: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)


@dataclass
class OCRResult:
    """Агрегированный результат распознавания всего изображения."""

    boxes: List[OCRBox]
    elapsed_seconds: float

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.boxes if b.text.strip())

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def average_confidence(self) -> Optional[float]:
        scores = [b.confidence for b in self.boxes if b.confidence is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)
