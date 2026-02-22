from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, overload

from typing_extensions import TypedDict, TypeVar

from .exceptions import UserError
from .tool_context import ToolContext
from .util._types import MaybeAwaitable

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class ToolInputGuardrailResult:

    guardrail: ToolInputGuardrail[Any]
    output: ToolGuardrailFunctionOutput


@dataclass
class ToolOutputGuardrailResult:

    guardrail: ToolOutputGuardrail[Any]
    output: ToolGuardrailFunctionOutput


class RejectContentBehavior(TypedDict):
    """Rejects the tool call/output but continues execution with a message to the model."""

    type: Literal["reject_content"]
    message: str


class RaiseExceptionBehavior(TypedDict):
    """Raises an exception to halt execution."""

    type: Literal["raise_exception"]


class AllowBehavior(TypedDict):
    """Allows normal tool execution to continue."""

    type: Literal["allow"]


@dataclass
class ToolGuardrailFunctionOutput:

    output_info: Any
    behavior: RejectContentBehavior | RaiseExceptionBehavior | AllowBehavior = field(
        default_factory=lambda: AllowBehavior(type="allow")
    )

    @classmethod
    def allow(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Create a guardrail output that allows the tool execution to continue normally.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to allow normal execution.
        """
        return cls(output_info=output_info, behavior=AllowBehavior(type="allow"))

    @classmethod
    def reject_content(cls, message: str, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Create a guardrail output that rejects the tool call/output but continues execution.

        Args:
            message: Message to send to the model instead of the tool result.
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to reject the content.
        """
        return cls(
            output_info=output_info,
            behavior=RejectContentBehavior(type="reject_content", message=message),
        )

    @classmethod
    def raise_exception(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Create a guardrail output that raises an exception to halt execution.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to raise an exception.
        """
        return cls(output_info=output_info, behavior=RaiseExceptionBehavior(type="raise_exception"))


@dataclass
class ToolInputGuardrailData:

    context: ToolContext[Any]
    agent: Agent[Any]


@dataclass
class ToolOutputGuardrailData(ToolInputGuardrailData):

    output: Any


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)


@dataclass
class ToolInputGuardrail(Generic[TContext_co]):

    guardrail_function: Callable[
        [ToolInputGuardrailData], MaybeAwaitable[ToolGuardrailFunctionOutput]
    ]
    name: str | None = None

    def get_name(self) -> str:
        return self.name or self.guardrail_function.__name__

    async def run(self, data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        result = self.guardrail_function(data)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class ToolOutputGuardrail(Generic[TContext_co]):

    guardrail_function: Callable[
        [ToolOutputGuardrailData], MaybeAwaitable[ToolGuardrailFunctionOutput]
    ]
    name: str | None = None

    def get_name(self) -> str:
        return self.name or self.guardrail_function.__name__

    async def run(self, data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        result = self.guardrail_function(data)
        if inspect.isawaitable(result):
            return await result
        return result


# Decorators
_ToolInputFuncSync = Callable[[ToolInputGuardrailData], ToolGuardrailFunctionOutput]
_ToolInputFuncAsync = Callable[[ToolInputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]]


@overload
def tool_input_guardrail(func: _ToolInputFuncSync): ...


@overload
def tool_input_guardrail(func: _ToolInputFuncAsync): ...


@overload
def tool_input_guardrail(
    *, name: str | None = None
) -> Callable[[_ToolInputFuncSync | _ToolInputFuncAsync], ToolInputGuardrail[Any]]: ...


def tool_input_guardrail(
    func: _ToolInputFuncSync | _ToolInputFuncAsync | None = None,
    *,
    name: str | None = None,
) -> (
    ToolInputGuardrail[Any]
    | Callable[[_ToolInputFuncSync | _ToolInputFuncAsync], ToolInputGuardrail[Any]]
):
    """Decorator to create a ToolInputGuardrail from a function."""

    def decorator(f: _ToolInputFuncSync | _ToolInputFuncAsync) -> ToolInputGuardrail[Any]:
        return ToolInputGuardrail(guardrail_function=f, name=name or f.__name__)

    if func is not None:
        return decorator(func)
    return decorator


_ToolOutputFuncSync = Callable[[ToolOutputGuardrailData], ToolGuardrailFunctionOutput]
_ToolOutputFuncAsync = Callable[[ToolOutputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]]


@overload
def tool_output_guardrail(func: _ToolOutputFuncSync): ...


@overload
def tool_output_guardrail(func: _ToolOutputFuncAsync): ...


@overload
def tool_output_guardrail(
    *, name: str | None = None
) -> Callable[[_ToolOutputFuncSync | _ToolOutputFuncAsync], ToolOutputGuardrail[Any]]: ...


def tool_output_guardrail(
    func: _ToolOutputFuncSync | _ToolOutputFuncAsync | None = None,
    *,
    name: str | None = None,
) -> (
    ToolOutputGuardrail[Any]
    | Callable[[_ToolOutputFuncSync | _ToolOutputFuncAsync], ToolOutputGuardrail[Any]]
):
    """Decorator to create a ToolOutputGuardrail from a function."""

    def decorator(f: _ToolOutputFuncSync | _ToolOutputFuncAsync) -> ToolOutputGuardrail[Any]:
        return ToolOutputGuardrail(guardrail_function=f, name=name or f.__name__)

    if func is not None:
        return decorator(func)
    return decorator
