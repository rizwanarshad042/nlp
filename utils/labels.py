from __future__ import annotations

from typing import Dict, List


LABELS: List[str] = ["credible", "misleading", "false"]
LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


def normalize_label(label: str) -> str:
    return label.strip().lower()


__all__ = ["LABELS", "LABEL_TO_ID", "ID_TO_LABEL", "normalize_label"]


