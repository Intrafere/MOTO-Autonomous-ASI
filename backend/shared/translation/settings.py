"""Persisted settings for Auto Top-p Translate.

Settings hold the selected target languages (in priority order), the one
translator submitter and one translator validator model role shared by the
whole app, the opt-in batch translation switch, the modes whose current run is
translated automatically, and the Translate all flag. They persist as
non-secret runtime settings under the active data root; prompts and model
responses are never stored with them.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from backend.shared import runtime_settings
from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text
from backend.shared.models import ModelConfig, ModelProvider, RoleReasoningEffort
from backend.shared.subscription_providers.metadata import MAX_EFFORT_PROVIDER_IDS
from backend.shared.translation.languages import (
    CUSTOM_LANGUAGE_KEY_PREFIX,
    custom_language_key,
    resolve_language,
    sanitize_custom_language_name,
)

logger = logging.getLogger(__name__)

MAX_TRANSLATION_LANGUAGES = 40
GENERIC_MODE_PROVIDER_ERROR = "Hosted mode supports OpenRouter translator models only."

# Workflow modes whose current run can be translated automatically, in
# canonical order. LeanOJ is a developer-only mode, so it is off by default.
TranslationMode = Literal["autonomous", "aggregator", "compiler", "leanoj"]
TRANSLATION_MODES: Tuple[str, ...] = get_args(TranslationMode)
DEFAULT_TRANSLATION_MODES: Tuple[str, ...] = ("autonomous", "aggregator", "compiler")


class TranslationRoleConfig(BaseModel):
    """The translator submitter or translator validator model role."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: ModelProvider
    model_id: str = Field(min_length=1, max_length=300)
    openrouter_provider: Optional[str] = Field(default=None, max_length=200)
    openrouter_reasoning_effort: RoleReasoningEffort = "auto"
    lm_studio_fallback_id: Optional[str] = Field(default=None, max_length=300)
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)

    @field_validator("openrouter_provider", "lm_studio_fallback_id")
    @classmethod
    def _blank_to_none(cls, value: Optional[str]) -> Optional[str]:
        return value or None

    @model_validator(mode="after")
    def _validate_limits(self) -> "TranslationRoleConfig":
        if self.max_output_tokens >= self.context_window:
            raise ValueError("Max output tokens must be smaller than the context window.")
        # RoleReasoningEffort enforces this per field too; the explicit check keeps
        # the role contract self-contained.
        if self.openrouter_reasoning_effort == "max" and self.provider not in MAX_EFFORT_PROVIDER_IDS:
            raise ValueError(
                "Reasoning effort 'max' is supported only by "
                + ", ".join(sorted(MAX_EFFORT_PROVIDER_IDS))
            )
        return self

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            provider=self.provider,
            model_id=self.model_id,
            openrouter_model_id=self.model_id if self.provider == "openrouter" else None,
            openrouter_provider=self.openrouter_provider,
            openrouter_reasoning_effort=self.openrouter_reasoning_effort,
            lm_studio_fallback_id=self.lm_studio_fallback_id,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            supercharge_enabled=False,
        )


class TranslationLanguageSetting(BaseModel):
    """One selected target language."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=2, max_length=40)
    name: str = Field(min_length=1, max_length=60)
    custom: bool = False


class AutoTranslateSettings(BaseModel):
    """User settings for the Auto Top-p Translate background service.

    ``languages`` order is translation priority. Keys are made unique by
    ``normalize_settings``. ``batch_translation`` lets several documents
    translate at once when the configured models support it; otherwise one
    document translates at a time app-wide, beside at most one Translate all
    background document when the models support more. ``modes`` lists the
    workflow modes whose current run is translated automatically.
    ``translate_all`` is on while older displayed outputs translate in the
    background; like ``enabled`` and ``paused`` it changes only through
    control actions.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    paused: bool = False
    translate_all: bool = False
    batch_translation: bool = False
    modes: list[TranslationMode] = Field(default_factory=lambda: list(DEFAULT_TRANSLATION_MODES))
    languages: list[TranslationLanguageSetting] = Field(
        default_factory=list, max_length=MAX_TRANSLATION_LANGUAGES
    )
    translator: Optional[TranslationRoleConfig] = None
    validator: Optional[TranslationRoleConfig] = None


def normalize_settings(settings: AutoTranslateSettings, *, generic_mode: bool) -> AutoTranslateSettings:
    """Return settings with canonical, de-duplicated languages and modes and mode-valid roles.

    A language marked ``custom`` or whose key starts with "custom" gets its key
    recomputed from its sanitized name. Modes keep their canonical order.
    Raises ValueError with a user-safe message for unknown language keys,
    invalid custom language names, or (in hosted mode) non-OpenRouter roles.
    """
    languages: list[TranslationLanguageSetting] = []
    seen_keys: set[str] = set()
    for language in settings.languages:
        key = language.key
        if language.custom or key.strip().casefold().startswith("custom"):
            # Custom keys always derive from the sanitized name, so a client
            # placeholder such as "custom-pending" or a stale key is never trusted.
            key = custom_language_key(sanitize_custom_language_name(language.name))
        entry = resolve_language(key, language.name)
        if entry.key in seen_keys:
            continue
        seen_keys.add(entry.key)
        languages.append(
            TranslationLanguageSetting(
                key=entry.key,
                name=entry.name,
                custom=entry.key.startswith(CUSTOM_LANGUAGE_KEY_PREFIX),
            )
        )

    translator = settings.translator
    validator = settings.validator
    if generic_mode:
        roles = [role for role in (translator, validator) if role is not None]
        if any(role.provider != "openrouter" for role in roles):
            raise ValueError(GENERIC_MODE_PROVIDER_ERROR)
        translator = _without_fallback(translator)
        validator = _without_fallback(validator)

    checked = set(settings.modes)
    return settings.model_copy(
        update={
            "languages": languages,
            "modes": [mode for mode in TRANSLATION_MODES if mode in checked],
            "translator": translator,
            "validator": validator,
        }
    )


def _without_fallback(role: Optional[TranslationRoleConfig]) -> Optional[TranslationRoleConfig]:
    """Drop the LM Studio fallback, which hosted mode does not offer."""
    if role is None:
        return None
    return role.model_copy(update={"lm_studio_fallback_id": None})


def load_auto_translate_settings() -> AutoTranslateSettings:
    """Load persisted settings; a missing or invalid payload yields defaults."""
    payload = runtime_settings.get_auto_translate_runtime_settings()
    if payload is None:
        return AutoTranslateSettings()
    try:
        settings = AutoTranslateSettings.model_validate(payload)
        return normalize_settings(settings, generic_mode=bool(system_config.generic_mode))
    except (ValidationError, ValueError) as exc:
        logger.warning(
            "Ignoring invalid persisted Auto Top-p Translate settings: %s",
            redact_log_text(exc, 400),
        )
        return AutoTranslateSettings()


def save_auto_translate_settings(settings: AutoTranslateSettings) -> None:
    """Persist settings; raises RuntimeSettingsError when the write fails."""
    runtime_settings.save_auto_translate_runtime_settings(settings.model_dump(mode="json"))


def settings_ready(settings: AutoTranslateSettings) -> Tuple[bool, str]:
    """Return whether translation can run, with a user-facing reason when it cannot."""
    if not settings.languages:
        return False, "Select at least one translation language."
    if settings.translator is None:
        return False, "Configure the translator submitter model."
    if settings.validator is None:
        return False, "Configure the translation validator model."
    return True, ""
