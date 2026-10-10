from unittest import IsolatedAsyncioTestCase, mock

import httpx
from fastapi import HTTPException

from backend.api.routes import boost as boost_route
from backend.api.routes import openrouter as openrouter_route
from backend.shared.config import rag_config, system_config
from backend.shared.models import BoostConfig
from backend.shared.openrouter_client import OpenRouterClient


class OpenRouterCredentialTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.previous_key = rag_config.openrouter_api_key
        self.previous_enabled = rag_config.openrouter_enabled
        self.previous_generic = system_config.generic_mode
        rag_config.openrouter_api_key = None
        rag_config.openrouter_enabled = False
        system_config.generic_mode = True
        openrouter_route._openrouter_validation_state = "unknown"
        openrouter_route._validated_openrouter_key_fingerprint = None
        openrouter_route._validated_openrouter_key_at = None

    async def asyncTearDown(self) -> None:
        rag_config.openrouter_api_key = self.previous_key
        rag_config.openrouter_enabled = self.previous_enabled
        system_config.generic_mode = self.previous_generic

    async def test_client_validates_with_authenticated_key_endpoint(self) -> None:
        client = OpenRouterClient("candidate")
        response = httpx.Response(
            200,
            json={"data": {"label": "test"}},
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
        )
        with mock.patch.object(client.client, "get", mock.AsyncMock(return_value=response)) as request:
            result = await client.get_current_api_key()
        await client.close()
        self.assertEqual(result["label"], "test")
        self.assertTrue(request.await_args.args[0].endswith("/key"))

    async def test_invalid_candidate_does_not_replace_runtime_key(self) -> None:
        rag_config.openrouter_api_key = "old-key"
        rag_config.openrouter_enabled = True
        rejected = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
            response=httpx.Response(401),
        )
        with mock.patch.object(
            openrouter_route,
            "_validate_openrouter_key",
            mock.AsyncMock(side_effect=rejected),
        ):
            with self.assertRaises(HTTPException) as raised:
                await openrouter_route.set_api_key(
                    openrouter_route.SetApiKeyRequest(api_key="bad-key")
                )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(rag_config.openrouter_api_key, "old-key")

    async def test_test_connection_rejects_invalid_key(self) -> None:
        rejected = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
            response=httpx.Response(401),
        )
        with mock.patch.object(
            openrouter_route,
            "_validate_openrouter_key",
            mock.AsyncMock(side_effect=rejected),
        ):
            with self.assertRaises(HTTPException) as raised:
                await openrouter_route.test_connection(
                    openrouter_route.SetApiKeyRequest(api_key="bad-key")
                )
        self.assertEqual(raised.exception.status_code, 400)

    async def test_restored_invalid_key_is_reported_without_exposure(self) -> None:
        rag_config.openrouter_api_key = "stored-secret"
        rejected = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
            response=httpx.Response(401),
        )
        with mock.patch.object(
            openrouter_route,
            "_validate_openrouter_key",
            mock.AsyncMock(side_effect=rejected),
        ):
            result = await openrouter_route.validate_current_api_key()
        self.assertEqual(result["validation_state"], "invalid")
        self.assertNotIn("stored-secret", str(result))

    async def test_validation_state_does_not_transfer_to_a_replaced_runtime_key(self) -> None:
        rag_config.openrouter_api_key = "old-key"
        openrouter_route._openrouter_validation_state = "valid"
        openrouter_route._validated_openrouter_key_fingerprint = (
            openrouter_route._key_fingerprint("old-key")
        )
        rag_config.openrouter_api_key = "new-restored-key"

        status = await openrouter_route.get_api_key_status()

        self.assertTrue(status["has_key"])
        self.assertEqual(status["validation_state"], "unknown")

    async def test_valid_validation_state_expires(self) -> None:
        rag_config.openrouter_api_key = "stored-key"
        openrouter_route._openrouter_validation_state = "valid"
        openrouter_route._validated_openrouter_key_fingerprint = (
            openrouter_route._key_fingerprint("stored-key")
        )
        openrouter_route._validated_openrouter_key_at = 1.0

        with mock.patch.object(
            openrouter_route.time,
            "monotonic",
            return_value=1.0 + openrouter_route._OPENROUTER_VALIDATION_TTL_SECONDS + 1,
        ):
            status = await openrouter_route.get_api_key_status()

        self.assertEqual(status["validation_state"], "unknown")

    async def test_unreachable_current_key_can_be_revalidated_after_recovery(self) -> None:
        rag_config.openrouter_api_key = "stored-key"
        openrouter_route._openrouter_validation_state = "unreachable"
        openrouter_route._validated_openrouter_key_fingerprint = (
            openrouter_route._key_fingerprint("stored-key")
        )

        with mock.patch.object(
            openrouter_route,
            "_validate_openrouter_key",
            mock.AsyncMock(return_value={"label": "recovered"}),
        ) as validate:
            result = await openrouter_route.validate_current_api_key()

        validate.assert_awaited_once_with("stored-key")
        self.assertEqual(result["validation_state"], "valid")
        self.assertEqual((await openrouter_route.get_api_key_status())["validation_state"], "valid")

    async def test_boost_rejected_key_is_a_client_error(self) -> None:
        rejected = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
            response=httpx.Response(401),
        )
        config = BoostConfig(
            enabled=True,
            openrouter_api_key="bad-boost-key",
            boost_model_id="example/model",
        )
        with mock.patch.object(
            OpenRouterClient,
            "get_current_api_key",
            mock.AsyncMock(side_effect=rejected),
        ):
            with self.assertRaises(HTTPException) as raised:
                await boost_route.enable_boost(config)

        self.assertEqual(raised.exception.status_code, 400)

    async def test_boost_unreachable_validation_is_a_gateway_error(self) -> None:
        config = BoostConfig(
            enabled=True,
            openrouter_api_key="candidate-boost-key",
            boost_model_id="example/model",
        )
        with mock.patch.object(
            OpenRouterClient,
            "get_current_api_key",
            mock.AsyncMock(side_effect=httpx.ConnectError("offline")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await boost_route.enable_boost(config)

        self.assertEqual(raised.exception.status_code, 502)
