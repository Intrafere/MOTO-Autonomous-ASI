import unittest

from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_retryable_model_output_error,
    is_transient_model_call_error,
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

    def test_codex_gateway_timeout_with_missing_fallback_is_transient(self) -> None:
        exc = RuntimeError(
            "OpenAI Codex failed for role 'autonomous_proof_formalization_brainstorm' "
            "and no LM Studio fallback is configured: OpenAI Codex completion failed: "
            "upstream connect error or disconnect/reset before headers. "
            "retried and the latest reset reason: connection timeout"
        )

        self.assertTrue(is_transient_model_call_error(exc))
        self.assertFalse(is_non_retryable_model_error(exc))

    def test_xai_grok_gateway_timeout_with_missing_fallback_is_transient(self) -> None:
        exc = RuntimeError(
            "xAI Grok failed for role 'autonomous_proof_formalization_brainstorm' "
            "and no LM Studio fallback is configured: xAI Grok connection failed after 3 attempts: "
            "HTTP 503: upstream provider timeout"
        )

        self.assertTrue(is_transient_model_call_error(exc))
        self.assertFalse(is_non_retryable_model_error(exc))

    def test_openrouter_missing_fallback_remains_non_retryable(self) -> None:
        exc = RuntimeError(
            "OpenRouter error for role 'agg_sub1': upstream connect error "
            "and no LM Studio fallback configured"
        )

        self.assertFalse(is_transient_model_call_error(exc))
        self.assertTrue(is_non_retryable_model_error(exc))


if __name__ == "__main__":
    unittest.main()
