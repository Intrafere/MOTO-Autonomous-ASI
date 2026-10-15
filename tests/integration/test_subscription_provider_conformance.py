"""Provider recovery conformance suite for every registered API-key subscription provider.

A new subscription provider is covered here automatically once its profile is in
``backend.shared.subscription_providers.registry``. Each case drives the real
client, routes, and APIClientManager with mocked HTTP or a mocked client call.
"""
import json
from pathlib import Path
import re
import tempfile
import time
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException
import httpx

from backend.api.routes import cloud_access as cloud_access_route
from backend.api.routes import features as features_route
from backend.shared.api_client_manager import (
    APIClientManager,
    ProviderCooldownError,
    RetryableProviderError,
)
from backend.shared.config import system_config
from backend.shared.models import ModelConfig
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRepairRequiredError
from backend.shared.provider_recovery.models import FailureKind
from backend.shared.secret_store import SecretStoreError
from backend.shared.subscription_providers.api_key_client import SubscriptionApiKeyClient
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionRequestError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.metadata import (
    API_KEY_SUBSCRIPTION_PROVIDER_IDS,
    CLOUD_PROVIDERS,
    get_cloud_provider,
)
from backend.shared.subscription_providers.registry import (
    get_api_key_subscription_client,
    provider_adapter_registry,
)

_CLIENT_MODULE = "backend.shared.subscription_providers.api_key_client"
REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET = "sk-conformance-secret-0123456789abcdef"


def _success_response(client: SubscriptionApiKeyClient, text: str = "ok") -> httpx.Response:
    if client.settings.wire_api == "chat_completions_stream":
        chunks = [
            {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": text}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode("utf-8"))
    return httpx.Response(200, json={"id": "r", "model": "m", "output_text": text})


def _route(path: str, method: str):
    for route in cloud_access_route.router.routes:
        if route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def _error(cls, provider_id: str, message: str = "failure", **kwargs):
    label = get_cloud_provider(provider_id).label
    return cls(f"{label} {message}", provider=provider_id, provider_label=label, **kwargs)


class SubscriptionProviderConformanceTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._data_dir = tempfile.TemporaryDirectory()
        self._stack = [
            mock.patch.object(system_config, "data_dir", self._data_dir.name),
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch.object(system_config, "agent_conversation_memory_enabled", False),
        ]
        for patcher in self._stack:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._stack):
            patcher.stop()
        self._data_dir.cleanup()

    def providers(self):
        self.assertTrue(API_KEY_SUBSCRIPTION_PROVIDER_IDS)
        for provider_id in API_KEY_SUBSCRIPTION_PROVIDER_IDS:
            yield provider_id, get_api_key_subscription_client(provider_id)

    def fresh_client(self, provider_id: str, handler) -> SubscriptionApiKeyClient:
        client = SubscriptionApiKeyClient(get_api_key_subscription_client(provider_id).profile)
        client._api_key = SECRET
        client._persisted_region_checked = True
        client._region_id = client.settings.regions[0].region_id
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    def manager_for(self, provider_id: str, *, fallback: str | None = None) -> APIClientManager:
        manager = APIClientManager()
        manager.configure_role(
            "agg_sub1",
            ModelConfig(
                provider=provider_id,
                model_id="model-x",
                lm_studio_fallback_id=fallback,
                context_window=100000,
                max_output_tokens=4000,
            ),
        )
        return manager

    async def generate(self, manager: APIClientManager):
        return await manager._generate_completion_once(
            task_id="agg_sub1_001",
            role_id="agg_sub1",
            model="model-x",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_every_metadata_api_key_provider_is_registered_with_an_adapter(self) -> None:
        self.assertEqual(
            set(provider_adapter_registry.provider_ids()),
            set(API_KEY_SUBSCRIPTION_PROVIDER_IDS),
        )

    def test_frontend_registry_mirrors_backend_metadata(self) -> None:
        source = (REPO_ROOT / "frontend" / "src" / "utils" / "oauthProviders.js").read_text(encoding="utf-8")
        constants = dict(re.findall(r"export const (\w+_PROVIDER) = '([a-z_]+)';", source))
        entries = {}
        for block in re.split(r"\n  \{\n", source.split("export const CLOUD_ACCESS_PROVIDERS = [", 1)[1])[1:]:
            fields = dict(re.findall(r"^    (\w+): '([^']*)',$", block, flags=re.MULTILINE))
            provider_id = constants[re.search(r"^    id: (\w+),$", block, flags=re.MULTILINE).group(1)]
            entries[provider_id] = fields
        self.assertEqual(set(entries), {provider.provider_id for provider in CLOUD_PROVIDERS})
        for provider in CLOUD_PROVIDERS:
            fields = entries[provider.provider_id]
            self.assertEqual(fields["label"], provider.label)
            self.assertEqual(fields["shortLabel"], provider.short_label)
            self.assertEqual(fields["authKind"], provider.auth_kind)
            self.assertEqual(fields["apiSlug"], provider.api_slug)
            self.assertEqual(fields["featureFlag"], provider.capability_flag)

    def test_adapter_classifies_every_typed_failure(self) -> None:
        for provider_id in API_KEY_SUBSCRIPTION_PROVIDER_IDS:
            with self.subTest(provider=provider_id):
                adapter = provider_adapter_registry.get(provider_id)
                reset_at = int(time.time()) + 60
                usage = adapter.classify_error(_error(
                    SubscriptionUsageLimitError, provider_id, resets_at=reset_at, resets_in_seconds=60,
                ))
                self.assertEqual(usage.kind, FailureKind.TIMED_USAGE_LIMIT)
                self.assertEqual(usage.reset_at, reset_at)
                cases = (
                    (SubscriptionEntitlementError, "entitlement", FailureKind.ENTITLEMENT),
                    (SubscriptionAuthError, "rejected the key", FailureKind.AUTHENTICATION),
                    (SubscriptionContextLimitError, "context_length_exceeded", FailureKind.CONTEXT_LIMIT),
                    (SubscriptionRequestError, "connection failed after retries: HTTP 503", FailureKind.TRANSIENT_TRANSPORT),
                    (SubscriptionRequestError, "completion failed: bad request", FailureKind.UNKNOWN_HARD),
                )
                for cls, message, kind in cases:
                    failure = adapter.classify_error(_error(cls, provider_id, message))
                    self.assertEqual(failure.kind, kind, (cls.__name__, message))
                    self.assertEqual(failure.provider, provider_id)

    async def test_transient_http_failure_is_retried_through_the_shared_seam(self) -> None:
        for provider_id, registered in self.providers():
            with self.subTest(provider=provider_id):
                responses = [httpx.Response(503, text="service unavailable"), None]

                def handler(request, responses=responses, provider_id=provider_id):
                    response = responses.pop(0)
                    return response or _success_response(get_api_key_subscription_client(provider_id), "recovered")

                client = self.fresh_client(provider_id, handler)
                with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
                    result = await client.generate_completion(
                        model=client.settings.default_model,
                        messages=[{"role": "user", "content": "hi"}],
                    )
                self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
                wait.assert_awaited_once()
                payload = wait.await_args.args[0]
                self.assertEqual(payload["provider"], provider_id)
                self.assertEqual(payload["provider_label"], registered.provider_label)
                self.assertEqual(payload["reason"], "transient_http_error")

    async def test_provider_error_text_never_contains_the_api_key(self) -> None:
        for provider_id, _registered in self.providers():
            with self.subTest(provider=provider_id):
                client = self.fresh_client(
                    provider_id,
                    lambda request: httpx.Response(400, json={"error": {"message": f"rejected {SECRET}"}}),
                )
                with self.assertRaises(SubscriptionRequestError) as ctx:
                    await client.generate_completion(
                        model=client.settings.default_model,
                        messages=[{"role": "user", "content": "hi"}],
                        tools=[{"type": "function", "function": {"name": "t"}}],
                    )
                self.assertNotIn(SECRET, str(ctx.exception))

    async def test_hosted_mode_hides_and_rejects_the_provider(self) -> None:
        with mock.patch.object(system_config, "generic_mode", True):
            features = await features_route.get_features()
            status = await cloud_access_route.get_cloud_access_status()
            for provider_id, _registered in self.providers():
                with self.subTest(provider=provider_id):
                    self.assertFalse(features[get_cloud_provider(provider_id).capability_flag])
                    self.assertFalse(status["providers"][provider_id]["configured"])
                    slug = get_cloud_provider(provider_id).api_slug
                    with self.assertRaises(HTTPException) as ctx:
                        await _route(f"/api/cloud-access/{slug}/api-key", "POST")(
                            cloud_access_route.SubscriptionApiKeyRequest(api_key="sk-test")
                        )
                    self.assertEqual(ctx.exception.status_code, 501)

    async def test_saving_validates_first_and_a_failed_commit_keeps_the_previous_key(self) -> None:
        for provider_id, client in self.providers():
            with self.subTest(provider=provider_id):
                slug = get_cloud_provider(provider_id).api_slug
                endpoint = _route(f"/api/cloud-access/{slug}/api-key", "POST")
                rejected = SubscriptionAuthError("bad key", provider=provider_id, provider_label=client.provider_label)
                with (
                    mock.patch.object(client, "validate_candidate", mock.AsyncMock(side_effect=rejected)),
                    mock.patch.object(client, "set_api_key") as set_api_key,
                ):
                    with self.assertRaises(HTTPException) as ctx:
                        await endpoint(cloud_access_route.SubscriptionApiKeyRequest(api_key="bad"))
                self.assertEqual(ctx.exception.status_code, 400)
                set_api_key.assert_not_called()

                previous = client._api_key
                validation = mock.Mock(region_id=client.settings.regions[0].region_id, models=[])
                with (
                    mock.patch.object(client, "validate_candidate", mock.AsyncMock(return_value=validation)),
                    mock.patch(f"{_CLIENT_MODULE}.store_subscription_api_key", side_effect=SecretStoreError("keyring")),
                ):
                    with self.assertRaises(HTTPException) as ctx:
                        await endpoint(cloud_access_route.SubscriptionApiKeyRequest(api_key=SECRET))
                self.assertEqual(ctx.exception.status_code, 500)
                self.assertNotIn(SECRET, str(ctx.exception.detail))
                self.assertEqual(client._api_key, previous)

    async def test_timed_usage_limit_uses_fallback_then_confirms_one_resume(self) -> None:
        for provider_id, client in self.providers():
            with self.subTest(provider=provider_id):
                manager = self.manager_for(provider_id, fallback="local-fallback")
                usage = _error(SubscriptionUsageLimitError, provider_id, "usage limit", resets_in_seconds=120)
                with (
                    mock.patch.object(client, "generate_completion", mock.AsyncMock(side_effect=usage)),
                    mock.patch(
                        "backend.shared.api_client_manager.lm_studio_client.generate_completion",
                        new=mock.AsyncMock(return_value={"choices": [{"message": {"content": "fallback"}}]}),
                    ) as lm_studio,
                ):
                    await self.generate(manager)
                self.assertEqual(lm_studio.await_args.kwargs["model"], "local-fallback")
                self.assertTrue(manager.is_provider_cooling_down(provider_id))

                manager._oauth_provider_cooldowns[provider_id]["cooldown_until"] = int(time.time()) - 1
                success = {"choices": [{"message": {"content": "back"}}], "usage": {}}
                with (
                    mock.patch.object(client, "generate_completion", mock.AsyncMock(return_value=success)),
                    mock.patch.object(manager, "_broadcast", new=mock.AsyncMock()) as broadcast,
                ):
                    await self.generate(manager)
                    await self.generate(manager)
                resumed = [call for call in broadcast.await_args_list if call.args[0] == "provider_usage_limit_resumed"]
                self.assertEqual(len(resumed), 1)
                self.assertEqual(manager.get_fallback_state("agg_sub1"), provider_id)

    async def test_usage_limit_without_fallback_raises_a_stoppable_cooldown(self) -> None:
        for provider_id, client in self.providers():
            with self.subTest(provider=provider_id):
                manager = self.manager_for(provider_id)
                usage = _error(SubscriptionUsageLimitError, provider_id, "usage limit", resets_in_seconds=90)
                with mock.patch.object(client, "generate_completion", mock.AsyncMock(side_effect=usage)):
                    with self.assertRaises(ProviderCooldownError) as ctx:
                        await self.generate(manager)
                self.assertEqual(ctx.exception.provider, provider_id)

    async def test_untimed_entitlement_requires_repair_without_retrying(self) -> None:
        for provider_id, client in self.providers():
            with self.subTest(provider=provider_id):
                manager = self.manager_for(provider_id, fallback="local-fallback")
                entitlement = _error(SubscriptionEntitlementError, provider_id, "not entitled")
                generate = mock.AsyncMock(side_effect=entitlement)
                with (
                    mock.patch.object(client, "generate_completion", generate),
                    mock.patch.object(manager, "_broadcast", new=mock.AsyncMock()) as broadcast,
                ):
                    with self.assertRaises(ProviderRepairRequiredError) as ctx:
                        await self.generate(manager)
                generate.assert_awaited_once()
                self.assertEqual(ctx.exception.provider, provider_id)
                self.assertEqual(ctx.exception.reason, "entitlement_required")
                events = [call.args[0] for call in broadcast.await_args_list]
                self.assertIn("subscription_provider_error", events)

    async def test_transient_and_context_failures_keep_their_recovery_paths(self) -> None:
        for provider_id, client in self.providers():
            with self.subTest(provider=provider_id):
                manager = self.manager_for(provider_id)
                transient = _error(SubscriptionRequestError, provider_id, "connection failed after retries: HTTP 503")
                with mock.patch.object(client, "generate_completion", mock.AsyncMock(side_effect=transient)):
                    with self.assertRaises(RetryableProviderError) as ctx:
                        await self.generate(manager)
                self.assertNotIsInstance(ctx.exception, ProviderCooldownError)

                context = _error(SubscriptionContextLimitError, provider_id, "rejected the request: context_length_exceeded")
                with mock.patch.object(client, "generate_completion", mock.AsyncMock(side_effect=context)):
                    with self.assertRaises(ProviderContextLengthError):
                        await self.generate(manager)
