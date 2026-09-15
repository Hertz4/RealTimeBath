"""Internal result type shared by realization backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import LindbladModel


@dataclass(frozen=True)
class BackendResult:
    model: LindbladModel
    condition_number: float
    conversion_residual: float
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
