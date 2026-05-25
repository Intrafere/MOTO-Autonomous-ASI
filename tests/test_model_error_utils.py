import unittest

from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_retryable_model_output_error,
)


class ModelErrorUtilsTests(unittest.TestCase):
    def test_codex_max_output_incomplete_is_retryable_output_failure(self) -> None:
        exc = RuntimeError(
            "OpenAI Codex failed for role 'autonomous_proof_formalization_paper' "
            "and no LM Studio fallback is configured: OpenAI Codex completion failed: "
            '{"type":"response.incomplete","response":{"status":"incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}}'
        )

        self.assertTrue(is_retryable_model_output_error(exc))
        self.assertFalse(is_non_retryable_model_error(exc))

    def test_missing_fallback_without_output_incomplete_remains_non_retryable(self) -> None:
        exc = RuntimeError(
            "OpenAI Codex failed for role 'autonomous_proof_formalization_paper' "
            "and no LM Studio fallback is configured: unauthorized"
        )

        self.assertFalse(is_retryable_model_output_error(exc))
        self.assertTrue(is_non_retryable_model_error(exc))


if __name__ == "__main__":
    unittest.main()
