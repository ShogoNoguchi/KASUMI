from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceBundle:
    root: Path
    development: dict[str, Any]
    holdout: dict[str, Any]
    verification: dict[str, Any]
    reviews: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evidence(root: str | Path) -> EvidenceBundle:
    root = Path(root)
    return EvidenceBundle(
        root=root,
        development=load_json(root / "development_selection_summary.json"),
        holdout=load_json(root / "multiseed_holdout_summary.json"),
        verification=load_json(root / "verification_summary.json"),
        reviews=load_json(root / "automated_reviews_public.json"),
    )
