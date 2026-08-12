from unittest import IsolatedAsyncioTestCase

from backend.autonomous.core.proof_verification_stage import ProofVerificationStage


class ProofSourceReservationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        ProofVerificationStage._active_sources = {}
        ProofVerificationStage._active_sources_lock = None

    async def asyncTearDown(self) -> None:
        ProofVerificationStage._active_sources = {}
        ProofVerificationStage._active_sources_lock = None

    async def test_tokenless_and_stale_release_cannot_remove_owned_reservation(self):
        token = await ProofVerificationStage.reserve_source(
            "paper",
            "paper-one",
            owner_token="current-owner",
        )

        self.assertFalse(
            await ProofVerificationStage.release_source("paper", "paper-one")
        )
        self.assertFalse(
            await ProofVerificationStage.release_source(
                "paper",
                "paper-one",
                owner_token="stale-owner",
            )
        )
        self.assertTrue(
            await ProofVerificationStage.is_source_running("paper", "paper-one")
        )

        self.assertTrue(
            await ProofVerificationStage.release_source(
                "paper",
                "paper-one",
                owner_token=token,
            )
        )
        self.assertFalse(
            await ProofVerificationStage.is_source_running("paper", "paper-one")
        )
