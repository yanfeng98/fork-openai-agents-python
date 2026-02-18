import copy
import os
from typing import Optional

from openai.types.shared.reasoning import Reasoning

from agents.model_settings import ModelSettings

OPENAI_DEFAULT_MODEL_ENV_VARIABLE_NAME = "OPENAI_DEFAULT_MODEL"
_GPT_5_DEFAULT_MODEL_SETTINGS: ModelSettings = ModelSettings(
    reasoning=Reasoning(effort="low"),
    verbosity="low",
)
_GPT_5_NONE_DEFAULT_MODEL_SETTINGS: ModelSettings = ModelSettings(
    reasoning=Reasoning(effort="none"),
    verbosity="low",
)

_GPT_5_NONE_EFFORT_MODELS = {"gpt-5.1", "gpt-5.2"}


def _is_gpt_5_none_effort_model(model_name: str) -> bool:
    return model_name in _GPT_5_NONE_EFFORT_MODELS


def gpt_5_reasoning_settings_required(model_name: str) -> bool:
    if model_name.startswith("gpt-5-chat"):
        return False
    return model_name.startswith("gpt-5")


def is_gpt_5_default() -> bool:
    return gpt_5_reasoning_settings_required(get_default_model())


def get_default_model() -> str:
    return os.getenv(OPENAI_DEFAULT_MODEL_ENV_VARIABLE_NAME, "gpt-4.1").lower()


def get_default_model_settings(model: Optional[str] = None) -> ModelSettings:
    _model = model if model is not None else get_default_model()
    if gpt_5_reasoning_settings_required(_model):
        if _is_gpt_5_none_effort_model(_model):
            return copy.deepcopy(_GPT_5_NONE_DEFAULT_MODEL_SETTINGS)
        return copy.deepcopy(_GPT_5_DEFAULT_MODEL_SETTINGS)
    return ModelSettings()
