from unittest import TestCase, mock

from backend.shared.config import system_config
from backend.shared import runtime_settings


class LeanRuntimeCapabilityTests(TestCase):
    def setUp(self) -> None:
        self.original = {
            "data_dir": system_config.data_dir,
            "generic_mode": system_config.generic_mode,
            "lean4_enabled": system_config.lean4_enabled,
            "lean4_runtime_available": system_config.lean4_runtime_available,
            "lean4_user_enabled": system_config.lean4_user_enabled,
        }

    def tearDown(self) -> None:
        for field, value in self.original.items():
            setattr(system_config, field, value)

    def test_persisted_preference_cannot_override_unavailable_runtime(self) -> None:
        payload = {"proof_settings": {"lean4_user_enabled": True}}
        system_config.generic_mode = False
        system_config.lean4_runtime_available = False
        system_config.lean4_enabled = False
        system_config.lean4_user_enabled = None
        with mock.patch.object(runtime_settings, "_read_settings", return_value=payload):
            runtime_settings.apply_persisted_runtime_settings()

        self.assertTrue(system_config.lean4_user_enabled)
        self.assertFalse(system_config.lean4_enabled)

    def test_legacy_lean_enabled_migrates_to_preference(self) -> None:
        payload = {"proof_settings": {"lean4_enabled": True}}
        system_config.generic_mode = False
        system_config.lean4_runtime_available = False
        system_config.lean4_enabled = False
        system_config.lean4_user_enabled = None
        with mock.patch.object(runtime_settings, "_read_settings", return_value=payload):
            runtime_settings.apply_persisted_runtime_settings()

        self.assertTrue(system_config.lean4_user_enabled)
        self.assertFalse(system_config.lean4_enabled)

    def test_preserved_preference_reactivates_when_capability_returns(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_runtime_available = True
        system_config.lean4_user_enabled = True
        system_config.lean4_enabled = False

        runtime_settings.reconcile_lean4_runtime_state()

        self.assertTrue(system_config.lean4_enabled)

    def test_saved_settings_store_preference_not_effective_state(self) -> None:
        system_config.lean4_runtime_available = False
        system_config.lean4_user_enabled = True
        system_config.lean4_enabled = False
        with mock.patch.object(runtime_settings, "_read_settings", return_value={}):
            with mock.patch.object(runtime_settings, "_write_settings") as write:
                runtime_settings.save_proof_runtime_settings()

        payload = write.call_args.args[0]["proof_settings"]
        self.assertTrue(payload["lean4_user_enabled"])
        self.assertNotIn("lean4_enabled", payload)
