import os
from unittest import TestCase, mock

from backend.shared.config import SystemConfig


class ProofParallelConfigTests(TestCase):
    def test_default_max_parallel_candidates_is_six(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = SystemConfig(_env_file=None)

        self.assertEqual(config.proof_max_parallel_candidates, 6)

    def test_zero_remains_unlimited_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"MOTO_PROOF_MAX_PARALLEL_CANDIDATES": "0"},
            clear=True,
        ):
            config = SystemConfig(_env_file=None)

        self.assertEqual(config.proof_max_parallel_candidates, 0)
