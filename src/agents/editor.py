from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .run_context import RunContextWrapper
from .util._types import MaybeAwaitable

ApplyPatchOperationType = Literal["create_file", "update_file", "delete_file"]

_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KWARGS)
class ApplyPatchOperation:

    type: ApplyPatchOperationType
    path: str
    diff: str | None = None
    ctx_wrapper: RunContextWrapper | None = None


@dataclass(**_DATACLASS_KWARGS)
class ApplyPatchResult:

    status: Literal["completed", "failed"] | None = None
    output: str | None = None


@runtime_checkable
class ApplyPatchEditor(Protocol):

    def create_file(
        self, operation: ApplyPatchOperation
    ) -> MaybeAwaitable[ApplyPatchResult | str | None]: ...

    def update_file(
        self, operation: ApplyPatchOperation
    ) -> MaybeAwaitable[ApplyPatchResult | str | None]: ...

    def delete_file(
        self, operation: ApplyPatchOperation
    ) -> MaybeAwaitable[ApplyPatchResult | str | None]: ...
