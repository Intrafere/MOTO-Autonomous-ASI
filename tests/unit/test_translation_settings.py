from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.shared import runtime_settings
from backend.shared.config import bind_runtime_roots, system_config
from backend.shared.subscription_providers.metadata import MAX_EFFORT_PROVIDER_IDS
from backend.shared.translation import settings as translation_settings
from backend.shared.translation.languages import custom_language_key
from backend.shared.translation.settings import (
    DEFAULT_TRANSLATION_MODES,
    TRANSLATION_MODES,
    AutoTranslateSettings,
    TranslationLanguageSetting,
    TranslationRoleConfig,
    load_auto_translate_settings,
    normalize_settings,
    save_auto_translate_settings,
    settings_ready,
)


def _role(**overrides) -> dict:
    payload = {
        "provider": "openrouter",
        "model_id": "vendor/translator-model",
        "context_window": 128000,
        "max_output_tokens": 16000,
    }
    payload.update(overrides)
    return payload


def _settings(**overrides) -> AutoTranslateSettings:
    payload = {
        "enabled": True,
        "languages": [{"key": "es", "name": "Spanish"}],
        "translator": _role(),
        "validator": _role(model_id="vendor/validator-model"),
    }
    payload.update(overrides)
    return AutoTranslateSettings.model_validate(payload)


@pytest.fixture
def isolated_runtime_root(tmp_path):
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    bind_runtime_roots(tmp_path / "data", tmp_path / "logs")
    yield tmp_path / "data"
    bind_runtime_roots(original_data, original_logs)


def test_role_requires_output_below_context_window():
    with pytest.raises(ValidationError, match="smaller than the context window"):
        TranslationRoleConfig.model_validate(_role(context_window=8000, max_output_tokens=8000))
    with pytest.raises(ValidationError):
        TranslationRoleConfig.model_validate(_role(context_window=8000, max_output_tokens=9000))
    with pytest.raises(ValidationError):
        TranslationRoleConfig.model_validate(_role(context_window=0))
    assert TranslationRoleConfig.model_validate(_role(context_window=8000, max_output_tokens=7999))


def test_max_effort_only_for_max_capable_providers():
    with pytest.raises(ValidationError, match="'max' is supported only by"):
        TranslationRoleConfig.model_validate(_role(openrouter_reasoning_effort="max"))
    for provider in sorted(MAX_EFFORT_PROVIDER_IDS):
        role = TranslationRoleConfig.model_validate(
            _role(provider=provider, openrouter_reasoning_effort="max")
        )
        assert role.openrouter_reasoning_effort == "max"
    assert TranslationRoleConfig.model_validate(_role(openrouter_reasoning_effort="xhigh"))


def test_unknown_provider_and_blank_model_are_rejected():
    with pytest.raises(ValidationError):
        TranslationRoleConfig.model_validate(_role(provider="mystery_cloud"))
    with pytest.raises(ValidationError):
        TranslationRoleConfig.model_validate(_role(model_id="   "))


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        TranslationRoleConfig.model_validate(_role(supercharge_enabled=True))
    with pytest.raises(ValidationError):
        TranslationLanguageSetting.model_validate({"key": "es", "name": "Spanish", "extra": 1})
    with pytest.raises(ValidationError):
        AutoTranslateSettings.model_validate({"enabled": True, "unexpected": True})


def test_defaults_have_no_roles_and_batch_translation_off():
    settings = AutoTranslateSettings()
    assert settings.enabled is False
    assert settings.paused is False
    assert settings.batch_translation is False
    assert settings.languages == []
    assert settings.translator is None
    assert settings.validator is None
    assert settings.translate_all is False


def test_modes_default_to_every_standard_mode_with_leanoj_off():
    assert TRANSLATION_MODES == ("autonomous", "aggregator", "compiler", "leanoj")
    assert DEFAULT_TRANSLATION_MODES == ("autonomous", "aggregator", "compiler")
    assert AutoTranslateSettings().modes == ["autonomous", "aggregator", "compiler"]
    # The default list is a fresh copy per instance.
    first, second = AutoTranslateSettings(), AutoTranslateSettings()
    first.modes.append("leanoj")
    assert second.modes == ["autonomous", "aggregator", "compiler"]


def test_normalize_dedupes_modes_into_canonical_order():
    normalized = normalize_settings(
        _settings(modes=["leanoj", "compiler", "autonomous", "compiler"]),
        generic_mode=False,
    )
    assert normalized.modes == ["autonomous", "compiler", "leanoj"]
    assert normalize_settings(_settings(modes=[]), generic_mode=False).modes == []
    # Hosted mode accepts LeanOJ too; it has no Proof Solver runs, so it does nothing there.
    assert normalize_settings(_settings(modes=["leanoj"]), generic_mode=True).modes == ["leanoj"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValidationError):
        _settings(modes=["autonomous", "manual"])


def test_settings_hold_one_translator_and_one_validator():
    settings = _settings(batch_translation=True)
    assert settings.batch_translation is True
    assert settings.translator is not None
    assert settings.translator.model_id == "vendor/translator-model"
    assert settings.validator is not None
    assert settings.validator.model_id == "vendor/validator-model"
    assert set(settings.model_dump(mode="json")) == {
        "enabled",
        "paused",
        "translate_all",
        "batch_translation",
        "modes",
        "languages",
        "translator",
        "validator",
    }


def test_v1_submitter_lane_list_is_rejected():
    with pytest.raises(ValidationError):
        AutoTranslateSettings.model_validate({"submitters": [_role()]})
    with pytest.raises(ValidationError):
        AutoTranslateSettings.model_validate({"translator": [_role()]})


def test_blank_optional_role_strings_become_none():
    role = TranslationRoleConfig.model_validate(
        _role(openrouter_provider="  ", lm_studio_fallback_id="")
    )
    assert role.openrouter_provider is None
    assert role.lm_studio_fallback_id is None


def test_to_model_config_maps_openrouter_model_and_disables_supercharge():
    openrouter = TranslationRoleConfig.model_validate(
        _role(openrouter_provider="SomeHost", lm_studio_fallback_id="local-model")
    ).to_model_config()
    assert openrouter.provider == "openrouter"
    assert openrouter.model_id == "vendor/translator-model"
    assert openrouter.openrouter_model_id == "vendor/translator-model"
    assert openrouter.openrouter_provider == "SomeHost"
    assert openrouter.lm_studio_fallback_id == "local-model"
    assert openrouter.context_window == 128000
    assert openrouter.max_output_tokens == 16000
    assert openrouter.supercharge_enabled is False

    local = TranslationRoleConfig.model_validate(
        _role(provider="lm_studio", model_id="local-model")
    ).to_model_config()
    assert local.openrouter_model_id is None


def test_normalize_uses_canonical_names_and_dedupes_keeping_first():
    settings = _settings(
        languages=[
            {"key": "zh-Hans", "name": "chinese"},
            {"key": "es", "name": "Spanish"},
            {"key": "zh-Hans", "name": "Mandarin"},
            {"key": "ar", "name": "Arabic"},
        ]
    )

    normalized = normalize_settings(settings, generic_mode=False)

    assert [(item.key, item.name, item.custom) for item in normalized.languages] == [
        ("zh-Hans", "Mandarin Chinese", False),
        ("es", "Spanish", False),
        ("ar", "Modern Standard Arabic", False),
    ]
    # Input is not mutated.
    assert len(settings.languages) == 4


def test_normalize_rejects_unknown_language_key():
    with pytest.raises(ValueError, match="Unknown language key"):
        normalize_settings(_settings(languages=[{"key": "xx-Nope", "name": "Nope"}]), generic_mode=False)


def test_normalize_resolves_custom_languages():
    key = custom_language_key("Klingon")
    normalized = normalize_settings(
        _settings(languages=[{"key": key, "name": "  Klingon  ", "custom": False}]),
        generic_mode=False,
    )
    assert [(item.key, item.name, item.custom) for item in normalized.languages] == [
        (key, "Klingon", True)
    ]

    with pytest.raises(ValueError):
        normalize_settings(
            _settings(languages=[{"key": custom_language_key("123"), "name": "123", "custom": True}]),
            generic_mode=False,
        )


def test_normalize_recomputes_custom_and_placeholder_keys_from_names():
    klingon = custom_language_key("Klingon")
    elvish = custom_language_key("Elvish")
    normalized = normalize_settings(
        _settings(
            languages=[
                # Frontend placeholder for a not-yet-keyed custom language.
                {"key": "custom-pending", "name": "  Klingon ", "custom": True},
                # A stale custom key is replaced by the key of the current name.
                {"key": klingon, "name": "Elvish", "custom": True},
                # A "custom"-prefixed key is recomputed even when the flag is missing.
                {"key": "custom-anything", "name": "Klingon"},
                # The custom flag wins over a standard key.
                {"key": "ar", "name": "Tlhingan Hol", "custom": True},
            ]
        ),
        generic_mode=False,
    )

    assert [(item.key, item.name, item.custom) for item in normalized.languages] == [
        (klingon, "Klingon", True),
        (elvish, "Elvish", True),
        (custom_language_key("Tlhingan Hol"), "Tlhingan Hol", True),
    ]

    with pytest.raises(ValueError):
        normalize_settings(
            _settings(languages=[{"key": "custom-pending", "name": "1234", "custom": True}]),
            generic_mode=False,
        )


def test_generic_mode_requires_openrouter_roles_and_clears_fallbacks():
    with pytest.raises(ValueError, match="Hosted mode supports OpenRouter translator models only."):
        normalize_settings(
            _settings(translator=_role(provider="lm_studio", model_id="local")),
            generic_mode=True,
        )
    with pytest.raises(ValueError, match="Hosted mode supports OpenRouter translator models only."):
        normalize_settings(
            _settings(validator=_role(provider="xai_grok_oauth", model_id="grok")),
            generic_mode=True,
        )

    with_fallbacks = _settings(
        translator=_role(lm_studio_fallback_id="local-a"),
        validator=_role(lm_studio_fallback_id="local-b"),
    )
    hosted = normalize_settings(with_fallbacks, generic_mode=True)
    assert hosted.translator is not None
    assert hosted.translator.lm_studio_fallback_id is None
    assert hosted.validator is not None
    assert hosted.validator.lm_studio_fallback_id is None
    # Input is not mutated.
    assert with_fallbacks.translator.lm_studio_fallback_id == "local-a"

    desktop = normalize_settings(with_fallbacks, generic_mode=False)
    assert desktop.translator.lm_studio_fallback_id == "local-a"
    assert desktop.validator.lm_studio_fallback_id == "local-b"


def test_generic_mode_accepts_settings_without_roles():
    normalized = normalize_settings(_settings(translator=None, validator=None), generic_mode=True)
    assert normalized.translator is None
    assert normalized.validator is None


def test_normalize_keeps_batch_translation_and_service_flags():
    normalized = normalize_settings(
        _settings(enabled=False, paused=True, batch_translation=True, translate_all=True),
        generic_mode=True,
    )
    assert (normalized.enabled, normalized.paused, normalized.batch_translation) == (False, True, True)
    assert normalized.translate_all is True


def test_save_and_load_round_trip_through_runtime_settings(isolated_runtime_root):
    runtime_settings.set_subscription_endpoint_region("kimi_code", "intl")
    settings = normalize_settings(
        _settings(
            paused=True,
            batch_translation=True,
            translate_all=True,
            modes=["compiler", "leanoj"],
            languages=[
                {"key": "fr", "name": "French"},
                {"key": custom_language_key("Klingon"), "name": "Klingon", "custom": True},
            ],
            translator=_role(provider="lm_studio", model_id="local-model"),
            validator=_role(openrouter_reasoning_effort="high", openrouter_provider="Host"),
        ),
        generic_mode=False,
    )

    save_auto_translate_settings(settings)

    raw = json.loads((Path(isolated_runtime_root) / "runtime_settings.json").read_text(encoding="utf-8"))
    assert raw["auto_translate"] == settings.model_dump(mode="json")
    assert raw["auto_translate"]["batch_translation"] is True
    assert raw["auto_translate"]["translate_all"] is True
    assert raw["auto_translate"]["modes"] == ["compiler", "leanoj"]
    assert raw["auto_translate"]["translator"]["model_id"] == "local-model"
    assert raw["subscription_endpoint_regions"] == {"kimi_code": "intl"}
    assert runtime_settings.get_auto_translate_runtime_settings() == settings.model_dump(mode="json")
    assert load_auto_translate_settings() == settings


def test_load_without_persisted_payload_returns_defaults(isolated_runtime_root):
    assert runtime_settings.get_auto_translate_runtime_settings() is None
    assert load_auto_translate_settings() == AutoTranslateSettings()


def test_runtime_settings_helper_removes_payload_with_none(isolated_runtime_root):
    runtime_settings.save_auto_translate_runtime_settings({"enabled": True})
    runtime_settings.set_subscription_endpoint_region("kimi_code", "cn")
    runtime_settings.save_auto_translate_runtime_settings(None)

    raw = json.loads((Path(isolated_runtime_root) / "runtime_settings.json").read_text(encoding="utf-8"))
    assert "auto_translate" not in raw
    assert raw["subscription_endpoint_regions"] == {"kimi_code": "cn"}
    assert runtime_settings.get_auto_translate_runtime_settings() is None


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": True, "unexpected": 1},
        {"languages": [{"key": "xx-Nope", "name": "Nope"}]},
        {"translator": _role(max_output_tokens=200000)},
        {"validator": _role(openrouter_reasoning_effort="max")},
        {"batch_translation": "sometimes"},
        {"modes": ["autonomous", "unknown_mode"]},
        {"translate_all": "maybe"},
        # The v1 lane list is not migrated: v1 never shipped.
        {"enabled": True, "submitters": [_role()], "validator": _role()},
    ],
)
def test_invalid_persisted_payload_falls_back_to_defaults(isolated_runtime_root, payload):
    runtime_settings.save_auto_translate_runtime_settings(payload)
    assert load_auto_translate_settings() == AutoTranslateSettings()


def test_non_object_persisted_payload_is_ignored(isolated_runtime_root):
    path = Path(isolated_runtime_root) / "runtime_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"auto_translate": ["not", "a", "dict"]}), encoding="utf-8")
    assert runtime_settings.get_auto_translate_runtime_settings() is None
    assert load_auto_translate_settings() == AutoTranslateSettings()


def test_load_in_generic_mode_rejects_persisted_local_roles(isolated_runtime_root, monkeypatch):
    save_auto_translate_settings(_settings(translator=_role(provider="lm_studio", model_id="local")))
    monkeypatch.setattr(system_config, "generic_mode", True)
    assert load_auto_translate_settings() == AutoTranslateSettings()


def test_save_propagates_runtime_settings_error(monkeypatch):
    def fail_write(_payload):
        raise runtime_settings.RuntimeSettingsError("Failed to persist runtime settings")

    monkeypatch.setattr(runtime_settings, "_read_settings", lambda: {})
    monkeypatch.setattr(runtime_settings, "_write_settings", fail_write)
    with pytest.raises(runtime_settings.RuntimeSettingsError):
        translation_settings.save_auto_translate_settings(_settings())


def test_settings_ready_requires_languages_translator_and_validator():
    assert settings_ready(_settings()) == (True, "")
    assert settings_ready(_settings(batch_translation=True)) == (True, "")
    assert settings_ready(AutoTranslateSettings()) == (False, "Select at least one translation language.")

    assert settings_ready(_settings(translator=None)) == (
        False,
        "Configure the translator submitter model.",
    )
    assert settings_ready(_settings(validator=None)) == (
        False,
        "Configure the translation validator model.",
    )
