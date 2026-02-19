from __future__ import annotations

import dataclasses
from dataclasses import fields, replace
from typing import Any

from pydantic.dataclasses import dataclass


def resolve_session_limit(
    explicit_limit: int | None,
    settings: SessionSettings | None,
) -> int | None:
    """Safely resolve the effective limit for session operations."""
    if explicit_limit is not None:
        return explicit_limit
    if settings is not None:
        return settings.limit
    return None


@dataclass
class SessionSettings:

    limit: int | None = None

    def resolve(self, override: SessionSettings | None) -> SessionSettings:
        if override is None:
            return self

        changes = {
            field.name: getattr(override, field.name)
            for field in fields(self)
            if getattr(override, field.name) is not None
        }

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
