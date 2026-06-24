from unittest import IsolatedAsyncioTestCase, mock

from backend.shared import embedding_readiness
from backend.shared.lm_studio_client import LMStudioClient


class LMStudioReadinessTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        embedding_readiness._lm_studio_embedding_status_cache = None
        embedding_readiness._lm_studio_embedding_status_cache_at = 0.0

    async def test_startup_probe_does_not_report_success_when_unavailable(self) -> None:
        client = LMStudioClient(base_url="http://127.0.0.1:1234")

        async def unavailable_status(*args, **kwargs):
            return {
                "available": False,
                "has_models": False,
                "model_count": 0,
                "models": [],
                "error": "Cannot connect to LM Studio server.",
            }

        with mock.patch.object(client, "check_availability", side_effect=unavailable_status):
            self.assertFalse(await client.test_connection())

    async def test_embedding_readiness_uses_quiet_lm_studio_probe(self) -> None:
        async def unavailable_embeddings(*args, **kwargs):
            raise RuntimeError("LM Studio unavailable")

        with mock.patch.object(
            embedding_readiness.lm_studio_client,
            "get_embeddings",
            side_effect=unavailable_embeddings,
        ) as get_embeddings:
            status = await embedding_readiness.check_lm_studio_embedding_ready(timeout_seconds=0.1)

        self.assertFalse(status["ready"])
        self.assertTrue(get_embeddings.call_args.kwargs["quiet"])

    async def test_embedding_readiness_caches_status_poll_probe(self) -> None:
        async def available_embeddings(*args, **kwargs):
            return [[0.1, 0.2, 0.3]]

        with mock.patch.object(
            embedding_readiness.lm_studio_client,
            "get_embeddings",
            side_effect=available_embeddings,
        ) as get_embeddings:
            first = await embedding_readiness.check_lm_studio_embedding_ready(timeout_seconds=0.1)
            second = await embedding_readiness.check_lm_studio_embedding_ready(timeout_seconds=0.1)

        self.assertTrue(first["ready"])
        self.assertTrue(second["ready"])
        self.assertEqual(get_embeddings.await_count, 1)

    async def test_embedding_provider_preflight_force_refreshes_lm_studio_probe(self) -> None:
        async def available_embeddings(*args, **kwargs):
            return [[0.1, 0.2, 0.3]]

        with mock.patch.object(embedding_readiness.rag_config, "openrouter_enabled", False), \
             mock.patch.object(embedding_readiness.rag_config, "openrouter_api_key", None), \
             mock.patch.object(
                 embedding_readiness.lm_studio_client,
                 "get_embeddings",
                 side_effect=available_embeddings,
             ) as get_embeddings:
            await embedding_readiness.check_embedding_provider_ready(timeout_seconds=0.1)
            await embedding_readiness.check_embedding_provider_ready(
                timeout_seconds=0.1,
                force_refresh=True,
            )

        self.assertEqual(get_embeddings.await_count, 2)
