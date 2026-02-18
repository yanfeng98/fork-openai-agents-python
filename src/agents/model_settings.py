from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Annotated, Any, Literal, Union

from openai import Omit as _Omit
from openai._types import Body, Query
from openai.types.responses import ResponseIncludable
from openai.types.shared import Reasoning
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic.dataclasses import dataclass
from pydantic_core import core_schema
from typing_extensions import TypeAlias


class _OmitTypeAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_from_none(value: None) -> _Omit:
            return _Omit()

        from_none_schema = core_schema.chain_schema(
            [
                core_schema.none_schema(),
                core_schema.no_info_plain_validator_function(validate_from_none),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=from_none_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(_Omit),
                    from_none_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda instance: None),
        )


@dataclass
class MCPToolChoice:
    server_label: str
    name: str


Omit = Annotated[_Omit, _OmitTypeAnnotation]
Headers: TypeAlias = Mapping[str, Union[str, Omit]]
ToolChoice: TypeAlias = Union[Literal["auto", "required", "none"], str, MCPToolChoice, None]


@dataclass
class ModelSettings:

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    truncation: Literal["auto", "disabled"] | None = None
    max_tokens: int | None = None
    reasoning: Reasoning | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    metadata: dict[str, str] | None = None
    store: bool | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    include_usage: bool | None = None
    response_include: list[ResponseIncludable | str] | None = None
    top_logprobs: int | None = None
    extra_query: Query | None = None
    extra_body: Body | None = None
    extra_headers: Headers | None = None
    extra_args: dict[str, Any] | None = None

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        """Produce a new ModelSettings by overlaying any non-None values from the
        override on top of this instance."""
        if override is None:
            return self

        changes = {
            field.name: getattr(override, field.name)
            for field in fields(self)
            if getattr(override, field.name) is not None
        }

        # Handle extra_args merging specially - merge dictionaries instead of replacing
        if self.extra_args is not None or override.extra_args is not None:
            merged_args = {}
            if self.extra_args:
                merged_args.update(self.extra_args)
            if override.extra_args:
                merged_args.update(override.extra_args)
            changes["extra_args"] = merged_args if merged_args else None

        return replace(self, **changes)

    def to_json_dict(self) -> dict[str, Any]:
        dataclass_dict = dataclasses.asdict(self)

        json_dict: dict[str, Any] = {}

        for field_name, value in dataclass_dict.items():
            if isinstance(value, BaseModel):
                json_dict[field_name] = value.model_dump(mode="json")
            else:
                json_dict[field_name] = value

        return json_dict
