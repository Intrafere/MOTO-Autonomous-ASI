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
            "lean4_proof_timeout": system_config.lean4_proof_timeout,
            "smt_timeout": system_config.smt_timeout,
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

    def test_persisted_legacy_default_proof_timeouts_follow_current_default(self) -> None:
        system_config.generic_mode = False
        for legacy in (600, 900, "900"):
            with self.subTest(legacy=legacy):
                system_config.lean4_proof_timeout = 2000
                payload = {"proof_settings": {"lean4_proof_timeout": legacy}}
                with mock.patch.object(runtime_settings, "_read_settings", return_value=payload):
                    runtime_settings.apply_persisted_runtime_settings()

                self.assertEqual(system_config.lean4_proof_timeout, 2000)

    def test_persisted_user_chosen_proof_timeout_is_preserved(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_proof_timeout = 2000
        payload = {"proof_settings": {"lean4_proof_timeout": 1200}}
        with mock.patch.object(runtime_settings, "_read_settings", return_value=payload):
            runtime_settings.apply_persisted_runtime_settings()

        self.assertEqual(system_config.lean4_proof_timeout, 1200)

    def test_versioned_settings_keep_values_equal_to_legacy_defaults(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_proof_timeout = 2000
        system_config.smt_timeout = 300
        with mock.patch.object(runtime_settings, "_read_settings", return_value={}):
            with mock.patch.object(runtime_settings, "_write_settings") as write:
                runtime_settings.save_proof_runtime_settings()
        saved = write.call_args.args[0]["proof_settings"]
        saved.update({"lean4_proof_timeout": 900, "smt_timeout": 30})

        with mock.patch.object(
            runtime_settings,
            "_read_settings",
            return_value={"proof_settings": saved},
        ):
            runtime_settings.apply_persisted_runtime_settings()

        self.assertEqual(system_config.lean4_proof_timeout, 900)
        self.assertEqual(system_config.smt_timeout, 30)

    def test_later_default_change_migrates_only_that_versions_shipped_default(self) -> None:
        system_config.generic_mode = False
        shipped = {
            **runtime_settings._SHIPPED_PROOF_DEFAULTS_BY_VERSION,
            2: {"lean4_proof_timeout": frozenset({"2000"})},
        }
        with mock.patch.object(runtime_settings, "_PROOF_SETTINGS_VERSION", 3), mock.patch.object(
            runtime_settings,
            "_SHIPPED_PROOF_DEFAULTS_BY_VERSION",
            shipped,
        ):
            for stored, expected in ((2000, 2500), (900, 900), (600, 600)):
                with self.subTest(stored=stored):
                    system_config.lean4_proof_timeout = 2500
                    payload = {
                        "proof_settings": {"settings_version": 2, "lean4_proof_timeout": stored},
                    }
                    with mock.patch.object(runtime_settings, "_read_settings", return_value=payload):
                        runtime_settings.apply_persisted_runtime_settings()

                    self.assertEqual(system_config.lean4_proof_timeout, expected)
