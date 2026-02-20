from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from .guardrail import InputGuardrail, OutputGuardrail
from .handoffs import HandoffHistoryMapper, HandoffInputFilter
from .items import TResponseInputItem
from .lifecycle import RunHooks
from .memory import Session, SessionInputCallback, SessionSettings
from .model_settings import ModelSettings
from .models.interface import Model, ModelProvider
from .models.multi_provider import MultiProvider
from .run_context import TContext
from .run_error_handlers import RunErrorHandlers
from .tracing import TracingConfig
from .util._types import MaybeAwaitable

if TYPE_CHECKING:
    from .agent import Agent
    from .run_context import RunContextWrapper


DEFAULT_MAX_TURNS = 10


def _default_trace_include_sensitive_data() -> bool:
    val = os.getenv("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", "true")
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ModelInputData:
    """Container for the data that will be sent to the model."""

    input: list[TResponseInputItem]
    instructions: str | None


@dataclass
class CallModelData(Generic[TContext]):
    """Data passed to `RunConfig.call_model_input_filter` prior to model call."""

    model_data: ModelInputData
    agent: Agent[TContext]
    context: TContext | None


CallModelInputFilter = Callable[[CallModelData[Any]], MaybeAwaitable[ModelInputData]]


@dataclass
class ToolErrorFormatterArgs(Generic[TContext]):

    kind: Literal["approval_rejected"]
    tool_type: Literal["function", "computer", "shell", "apply_patch"]
    tool_name: str
    call_id: str
    default_message: str
    run_context: RunContextWrapper[TContext]


ToolErrorFormatter = Callable[[ToolErrorFormatterArgs[Any]], MaybeAwaitable[Optional[str]]]


@dataclass
class RunConfig:

    model: str | Model | None = None
    model_provider: ModelProvider = field(default_factory=MultiProvider)
    model_settings: ModelSettings | None = None
    handoff_input_filter: HandoffInputFilter | None = None
    nest_handoff_history: bool = False
    handoff_history_mapper: HandoffHistoryMapper | None = None
    input_guardrails: list[InputGuardrail[Any]] | None = None
    output_guardrails: list[OutputGuardrail[Any]] | None = None
    tracing_disabled: bool = False
    tracing: TracingConfig | None = None
    trace_include_sensitive_data: bool = field(
        default_factory=_default_trace_include_sensitive_data
    )
    workflow_name: str = "Agent workflow"
    trace_id: str | None = None
    group_id: str | None = None
    trace_metadata: dict[str, Any] | None = None
    session_input_callback: SessionInputCallback | None = None
    call_model_input_filter: CallModelInputFilter | None = None
    tool_error_formatter: ToolErrorFormatter | None = None
    session_settings: SessionSettings | None = None


class RunOptions(TypedDict, Generic[TContext]):
    """Arguments for ``AgentRunner`` methods."""

    context: NotRequired[TContext | None]
    """The context for the run."""

    max_turns: NotRequired[int]
    """The maximum number of turns to run for."""

    hooks: NotRequired[RunHooks[TContext] | None]
    """Lifecycle hooks for the run."""

    run_config: NotRequired[RunConfig | None]
    """Run configuration."""

    previous_response_id: NotRequired[str | None]
    """The ID of the previous response, if any."""

    auto_previous_response_id: NotRequired[bool]
    """Enable automatic response chaining for the first turn."""

    conversation_id: NotRequired[str | None]
    """The ID of the stored conversation, if any."""

    session: NotRequired[Session | None]
    """The session for the run."""

    error_handlers: NotRequired[RunErrorHandlers[TContext] | None]
    """Error handlers keyed by error kind. Currently supports max_turns."""


__all__ = [
    "DEFAULT_MAX_TURNS",
    "CallModelData",
    "CallModelInputFilter",
    "ModelInputData",
    "RunConfig",
    "RunOptions",
    "ToolErrorFormatter",
    "ToolErrorFormatterArgs",
    "_default_trace_include_sensitive_data",
]
