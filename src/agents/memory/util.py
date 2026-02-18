from __future__ import annotations

from typing import Callable

from ..items import TResponseInputItem
from ..util._types import MaybeAwaitable

SessionInputCallback = Callable[
    [list[TResponseInputItem], list[TResponseInputItem]],
    MaybeAwaitable[list[TResponseInputItem]],
]
