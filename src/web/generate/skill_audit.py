"""Language-quality audit for web extractor skills."""

from __future__ import annotations

from slime.utils.types import Sample

from src.utils.skill_audit import audit_extractor_samples as _audit


async def audit_extractor_samples(args, extractor_samples: list[Sample]) -> None:
    await _audit(args, extractor_samples, domain="web")
