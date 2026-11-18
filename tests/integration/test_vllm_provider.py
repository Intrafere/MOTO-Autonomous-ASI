"""vLLM local-server provider: setup, model/context detection, routing, and RAG embeddings.

Every case drives the real client, routes, readiness checks, and APIClientManager
against a mocked vLLM HTTP server; no real server, keyring, or data root is used.
"""
import json
import tempfile
from typing import Callable, Dict, List
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from fastapi import HTTPException
import httpx

from backend.api.routes import cloud_access as cloud_access_route
from backend.api.routes import connectivity as connectivity_route
from backend.api.routes import features as features_route
from backend.shared import embedding_readiness
from backend.shared.api_client_manager import APIClientManager, RetryableProviderError
from backend.shared.config import rag_config, system_config
from backend.shared.models import ModelConfig
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRepairRequiredError
from backend.shared.runtime_settings import get_vllm_runtime_settings
from backend.shared.secret_store import SecretStoreError
from backend.shared.provider_route_identity import role_label
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionRequestError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.metadata import (
    API_KEY_SUBSCRIPTION_PROVIDER_IDS,
    LOCAL_SERVER_PROVIDER_IDS,
    MODEL_PROVIDER_IDS,
    get_cloud_provider,
    is_local_server_provider,
    local_fallback_id,
    split_local_fallback,
)
from backend.shared.subscription_providers.registry import (
    api_key_subscription_clients,
    get_api_key_subscription_client,
    provider_adapter_registry,
    vllm_client as registered_vllm_client,
)
from backend.shared.subscription_providers.vllm import (
    VLLM_PROFILE,
    VllmClient,
    normalize_vllm_server_url,
    normalize_vllm_servers,
)

_CLIENT_MODULE = "backend.shared.subscription_providers.api_key_client"
CHAT_SERVER = "http://127.0.0.1:8001/v1"
EMBED_SERVER = "http://gpu-box:8002/v1"
CHAT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SECOND_EMBED_MODEL = "BAAI/bge-m3"
KEY = "vllm-secret-key-0123456789"


def sse(text: str = "OK", *, finish: str = "stop") -> bytes:
    chunks = [
        {"id": "c1", "model": CHAT_MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": text}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    ]
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode("utf-8")


class FakeVllm:
    """Two mocked vLLM servers: one chat model, one embedding model."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.required_key = ""
        self.down: set[str] = set()
        self.chat_handler: Callable[[httpx.Request], httpx.Response] | None = None
        self.models: Dict[str, List[dict]] = {
            "127.0.0.1:8001": [{"id": CHAT_MODEL, "object": "model", "max_model_len": 32768}],
            "gpu-box:8002": [{"id": EMBED_MODEL, "object": "model", "max_model_len": 8192}],
        }
        self.embedding_models = {EMBED_MODEL}
        # model -> window; an input "longer" than it (by word count) is refused unless truncated.
        self.embedding_windows: Dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = f"{request.url.host}:{request.url.port}"
        if host in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if self.required_key and request.headers.get("Authorization") != f"Bearer {self.required_key}":
            return httpx.Response(401, json={"error": "Unauthorized"})
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": self.models.get(host, [])})
        body = json.loads(request.content)
        if path.endswith("/embeddings"):
            if body["model"] not in self.embedding_models:
                return httpx.Response(400, json={"error": {"message": "The model does not support Embeddings API"}})
            window = self.embedding_windows.get(body["model"])
            longest = max(len(text.split()) for text in body["input"])
            if window and longest > window and "truncate_prompt_tokens" not in body:
                return httpx.Response(400, json={"error": {"message": (
                    f"This model's maximum context length is {window} tokens. However, you requested "
                    f"{longest} tokens in the input for embedding generation. Please reduce the length of the input."
                )}})
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [float(index), 0.5, 0.25]} for index, _ in enumerate(body["input"])
            ]})
        if path.endswith("/chat/completions"):
            if self.chat_handler is not None:
                return self.chat_handler(request)
            return httpx.Response(200, content=sse())
        return httpx.Response(404, json={"error": {"message": "not found"}})

    def bodies(self, suffix: str) -> List[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith(suffix) and r.content]


class VllmHarness(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._data_dir = tempfile.TemporaryDirectory()
        self.stored_keys: Dict[str, str] = {}
        self._patches = [
            mock.patch.object(system_config, "data_dir", self._data_dir.name),
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch.object(system_config, "backend_port", 8000),
            mock.patch.object(system_config, "agent_conversation_memory_enabled", False),
            mock.patch.dict("os.environ", {}, clear=False),
            mock.patch(f"{_CLIENT_MODULE}.load_subscription_api_key", side_effect=self.stored_keys.get),
            mock.patch(f"{_CLIENT_MODULE}.store_subscription_api_key", side_effect=self.stored_keys.__setitem__),
            mock.patch(
                f"{_CLIENT_MODULE}.clear_subscription_api_key",
                side_effect=lambda name: self.stored_keys.pop(name, None),
            ),
        ]
        for patcher in self._patches:
            patcher.start()
        for name in ("MOTO_VLLM_BASE_URL", "MOTO_VLLM_API_KEY", "VLLM_API_KEY"):
            __import__("os").environ.pop(name, None)
        embedding_readiness.reset_vllm_embedding_status_cache()
        self.server = FakeVllm()
        self.client = self.new_client()

    def tearDown(self) -> None:
        embedding_readiness.reset_vllm_embedding_status_cache()
        for patcher in reversed(self._patches):
            patcher.stop()
        self._data_dir.cleanup()

    def new_client(self) -> VllmClient:
        client = VllmClient(VLLM_PROFILE)
        client._api_key = None
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(self.server))
        return client

    async def configure(self, **kwargs):
        return await self.client.configure((CHAT_SERVER, EMBED_SERVER), **kwargs)


class VllmAddressTests(TestCase):
    def test_addresses_are_normalized_to_the_openai_v1_base(self) -> None:
        self.assertEqual(normalize_vllm_server_url("localhost:8001"), "http://localhost:8001/v1")
        self.assertEqual(normalize_vllm_server_url("http://GPU-Box:8001/"), "http://gpu-box:8001/v1")
        self.assertEqual(normalize_vllm_server_url("https://vllm.lan/proxy/v1/"), "https://vllm.lan/proxy/v1")

    def test_invalid_or_credentialed_addresses_are_rejected(self) -> None:
        for bad in ("", "ftp://host:1", "http://user:pass@host:8001", "http://host:8001/v1?key=1", "http://:8001"):
            with self.subTest(address=bad), self.assertRaises(ValueError):
                normalize_vllm_server_url(bad)

    def test_the_list_is_deduplicated_and_never_points_at_motos_own_backend(self) -> None:
        with mock.patch.object(system_config, "backend_port", 8000):
            self.assertEqual(
                normalize_vllm_servers("localhost:8001, http://localhost:8001/v1 gpu:9000"),
                ("http://localhost:8001/v1", "http://gpu:9000/v1"),
            )
            with self.assertRaises(ValueError) as ctx:
                normalize_vllm_servers(["http://127.0.0.1:8000"])
            self.assertIn("--port 8001", str(ctx.exception))
            self.assertEqual(normalize_vllm_servers(["http://gpu-box:8000"]), ("http://gpu-box:8000/v1",))


class VllmRegistryTests(IsolatedAsyncioTestCase):
    def test_vllm_is_a_registered_local_server_role_provider(self) -> None:
        self.assertEqual(LOCAL_SERVER_PROVIDER_IDS, ("vllm",))
        self.assertIn("vllm", MODEL_PROVIDER_IDS)
        self.assertNotIn("vllm", API_KEY_SUBSCRIPTION_PROVIDER_IDS)
        self.assertTrue(is_local_server_provider("vllm"))
        self.assertIs(get_api_key_subscription_client("vllm"), registered_vllm_client)
        self.assertNotIn(registered_vllm_client, api_key_subscription_clients())
        self.assertIn("vllm", provider_adapter_registry.provider_ids())
        ModelConfig(provider="vllm", model_id=CHAT_MODEL, context_window=32768, max_output_tokens=4096)

    async def test_features_expose_the_capability_and_hosted_mode_turns_it_off(self) -> None:
        with mock.patch.object(system_config, "generic_mode", False):
            features = await features_route.get_features()
        self.assertTrue(features["vllm_available"])
        entry = next(item for item in features["subscription_providers"] if item["id"] == "vllm")
        self.assertEqual(entry, {"id": "vllm", "label": "vLLM", "auth_kind": "local_server", "available": True})
        with mock.patch.object(system_config, "generic_mode", True):
            hosted = await features_route.get_features()
        self.assertFalse(hosted["vllm_available"])


class VllmSetupTests(VllmHarness):
    async def test_probe_detects_models_context_windows_and_the_embedding_model(self) -> None:
        result = await self.client.probe((CHAT_SERVER, EMBED_SERVER), None)
        self.assertEqual([m["id"] for m in result.chat_models], [CHAT_MODEL])
        self.assertEqual(result.chat_models[0]["context_length"], 32768)
        self.assertEqual([m["id"] for m in result.embedding_models], [EMBED_MODEL])
        self.assertEqual(result.embedding_models[0]["embedding_dimension"], 3)
        self.assertEqual((result.embedding_model, result.embedding_server), (EMBED_MODEL, EMBED_SERVER))
        # Probing changes nothing.
        self.assertEqual(get_vllm_runtime_settings(), {})
        self.assertFalse((await self.client.status())["configured"])

    async def test_saving_persists_non_secret_settings_and_lists_chat_models_with_context(self) -> None:
        await self.configure()
        self.assertEqual(get_vllm_runtime_settings(), {
            "servers": [CHAT_SERVER, EMBED_SERVER],
            "embedding_model": EMBED_MODEL,
            "embedding_server": EMBED_SERVER,
            "embedding_models": [EMBED_MODEL],
        })
        status = await self.client.status()
        self.assertTrue(status["configured"])
        self.assertTrue(status["embedding_configured"])
        self.assertFalse(status["has_api_key"])

        # A new process reads the saved setup and lists only chat models.
        models = await self.new_client().list_models()
        self.assertEqual([m["id"] for m in models], [CHAT_MODEL])
        self.assertEqual(models[0]["context_length"], 32768)
        self.assertNotIn("max_output_tokens", models[0])
        self.assertEqual(models[0]["provider_metadata"]["server"], CHAT_SERVER)

    async def test_an_unreachable_server_is_never_saved(self) -> None:
        self.server.down.add("gpu-box:8002")
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.configure()
        self.assertIn(EMBED_SERVER, str(ctx.exception))
        self.assertEqual(get_vllm_runtime_settings(), {})

    async def test_an_api_key_is_optional_and_stored_only_in_the_keyring(self) -> None:
        self.server.required_key = KEY
        with self.assertRaises(SubscriptionAuthError) as ctx:
            await self.configure()
        self.assertIn("--api-key", ctx.exception.terminal_guidance)
        self.assertEqual(get_vllm_runtime_settings(), {})

        await self.configure(api_key=KEY)
        self.assertEqual(self.stored_keys, {"vllm_api_key": KEY})
        self.assertNotIn(KEY, json.dumps(get_vllm_runtime_settings()))
        self.assertTrue((await self.client.status())["has_api_key"])
        self.assertNotIn(KEY, json.dumps(await self.client.status()))

        # Saving again without a key keeps the stored one.
        await self.configure()
        self.assertEqual(self.stored_keys, {"vllm_api_key": KEY})
        self.server.required_key = ""
        await self.configure(clear_api_key=True)
        self.assertEqual(self.stored_keys, {})

    async def test_a_requested_embedding_model_must_answer_embedding_requests(self) -> None:
        with self.assertRaises(ValueError):
            await self.configure(embedding_model=CHAT_MODEL)
        result = await self.configure(embedding_model=EMBED_MODEL)
        self.assertEqual(result.embedding_model, EMBED_MODEL)

    async def test_saving_again_on_automatic_keeps_the_embedding_model_the_index_was_built_with(self) -> None:
        # A second embedding model is served by the first-listed server, so "first found" differs.
        self.server.models["127.0.0.1:8001"].append({"id": SECOND_EMBED_MODEL, "max_model_len": 8192})
        self.server.embedding_models.add(SECOND_EMBED_MODEL)
        chosen = await self.configure(embedding_model=EMBED_MODEL)
        self.assertEqual([m["id"] for m in chosen.embedding_models], [SECOND_EMBED_MODEL, EMBED_MODEL])

        again = await self.configure()
        self.assertEqual((again.embedding_model, again.embedding_server), (EMBED_MODEL, EMBED_SERVER))
        self.assertEqual(self.client.embedding_space(), f"vllm:{EMBED_MODEL}")
        preview = await self.client.probe((CHAT_SERVER, EMBED_SERVER), None)
        self.assertEqual(preview.embedding_model, EMBED_MODEL)

        # Once no listed server serves it, automatic moves to an embedding model that exists.
        self.server.models["gpu-box:8002"] = []
        moved = await self.configure()
        self.assertEqual((moved.embedding_model, moved.embedding_server), (SECOND_EMBED_MODEL, CHAT_SERVER))

    async def test_a_keyless_setup_asks_the_keyring_once(self) -> None:
        with mock.patch(
            f"{_CLIENT_MODULE}.load_subscription_api_key", side_effect=self.stored_keys.get
        ) as load:
            await self.configure()
            await self.client.status()
            await self.client.discover(force=True)
            await self.client.generate_completion(
                model=CHAT_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=64
            )
            self.assertEqual(load.call_count, 1)
            # A key saved later is used at once.
            await self.configure(api_key=KEY)
            self.assertTrue((await self.client.status())["has_api_key"])

    async def test_an_unavailable_keyring_does_not_stop_a_keyless_server(self) -> None:
        broken = SecretStoreError("Secure credential storage is unavailable.")
        with (
            mock.patch(f"{_CLIENT_MODULE}.load_subscription_api_key", side_effect=broken),
            mock.patch(f"{_CLIENT_MODULE}.clear_subscription_api_key", side_effect=broken),
        ):
            await self.configure()
            status = await self.client.status()
            self.assertTrue(status["configured"])
            self.assertFalse(status["has_api_key"])
            result = await self.client.generate_completion(
                model=CHAT_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=64
            )
            self.assertEqual(result["choices"][0]["message"]["content"], "OK")
            self.assertNotIn("Authorization", self.server.requests[-1].headers)
            # Removing a keyless setup needs no keyring either.
            await self.client.clear_configuration()
            self.assertFalse((await self.client.status())["configured"])

    async def test_a_model_a_running_server_stopped_serving_loses_its_route(self) -> None:
        await self.configure()
        self.assertEqual(self.client._model_servers[CHAT_MODEL], CHAT_SERVER)
        # While the server is down its route is kept, so the call stays a retryable failure.
        self.server.down.add("127.0.0.1:8001")
        await self.client.discover(force=True)
        self.assertEqual(self.client._model_servers[CHAT_MODEL], CHAT_SERVER)
        # Once it answers with another model, the old route is gone.
        self.server.down.clear()
        self.server.models["127.0.0.1:8001"] = [{"id": "other/model", "max_model_len": 4096}]
        await self.client.discover(force=True)
        self.assertNotIn(CHAT_MODEL, self.client._model_servers)
        self.assertEqual(self.client._model_servers["other/model"], CHAT_SERVER)

    async def test_clearing_forgets_servers_embedding_choice_and_key(self) -> None:
        await self.configure(api_key=KEY)
        await self.client.clear_configuration()
        self.assertEqual(get_vllm_runtime_settings(), {})
        self.assertEqual(self.stored_keys, {})
        self.assertFalse((await self.client.status())["configured"])
        self.assertIsNone(self.client.embedding_space())

    async def test_a_rebound_data_root_reloads_the_saved_setup(self) -> None:
        await self.configure()
        self.assertTrue((await self.client.status())["configured"])
        with tempfile.TemporaryDirectory() as other_root, mock.patch.object(system_config, "data_dir", other_root):
            # Another data root has no vLLM setup, and no route learned for the first one.
            self.assertFalse((await self.client.status())["configured"])
            self.assertIsNone(self.client.embedding_space())
            self.assertEqual(self.client._model_servers, {})
        self.assertEqual((await self.client.status())["servers"], [CHAT_SERVER, EMBED_SERVER])

    async def test_environment_servers_override_the_saved_list(self) -> None:
        await self.configure()
        with mock.patch.dict("os.environ", {"MOTO_VLLM_BASE_URL": "http://127.0.0.1:8001"}):
            status = await self.client.status()
        self.assertEqual(status["servers"], [CHAT_SERVER])
        self.assertEqual(status["server_source"], "environment")
        # The embedding server is not in the environment list, so it is not used.
        self.assertFalse(status["embedding_configured"])


class VllmCompletionTests(VllmHarness):
    async def complete(self, **kwargs):
        return await self.client.generate_completion(
            model=kwargs.pop("model", CHAT_MODEL),
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=kwargs.pop("max_tokens", 4096),
            **kwargs,
        )

    async def test_a_completion_goes_to_the_server_that_serves_the_model(self) -> None:
        await self.configure()
        self.server.requests.clear()
        result = await self.complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")
        self.assertEqual(result["usage"]["total_tokens"], 10)
        chat = [r for r in self.server.requests if r.url.path.endswith("/chat/completions")]
        self.assertEqual(len(chat), 1)
        self.assertEqual(f"{chat[0].url.host}:{chat[0].url.port}", "127.0.0.1:8001")
        self.assertNotIn("Authorization", chat[0].headers)
        self.assertIn("MOTO-Autonomous-ASI/", chat[0].headers["User-Agent"])
        body = json.loads(chat[0].content)
        self.assertEqual(body["model"], CHAT_MODEL)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["temperature"], 0.0)
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("reasoning", body)

    async def test_a_stored_api_key_is_sent_as_a_bearer_token(self) -> None:
        self.server.required_key = KEY
        await self.configure(api_key=KEY)
        result = await self.complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")

    async def test_a_model_no_server_serves_is_a_typed_repair_error(self) -> None:
        await self.configure()
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.complete(model="missing/model")
        self.assertEqual(ctx.exception.error_code, "invalid_model")
        self.assertIn("Inference Connectivity", ctx.exception.terminal_guidance)

    async def test_a_newly_started_model_is_found_without_saving_again(self) -> None:
        await self.configure()
        self.server.models["127.0.0.1:8001"].append({"id": "new/model", "max_model_len": 4096})
        result = await self.complete(model="new/model")
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")

    async def test_a_server_without_tool_calling_answers_without_tools(self) -> None:
        await self.configure()

        def handler(request: httpx.Request) -> httpx.Response:
            if "tools" in json.loads(request.content):
                return httpx.Response(400, json={"error": {"message": (
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )}})
            return httpx.Response(200, content=sse("no tools"))

        self.server.chat_handler = handler
        tools = [{"type": "function", "function": {"name": "wolfram_alpha_query"}}]
        result = await self.complete(tools=tools, tool_choice="auto")
        self.assertEqual(result["choices"][0]["message"]["content"], "no tools")
        self.server.requests.clear()
        await self.complete(tools=tools, tool_choice="auto")
        self.assertEqual(len(self.server.bodies("/chat/completions")), 1)
        self.assertNotIn("tools", self.server.bodies("/chat/completions")[0])

        # The server is restarted with tool calling: after the recheck window tools are offered again.
        self.server.chat_handler = None
        self.client._tools_unsupported_models[CHAT_MODEL] -= self.client.TOOLS_RECHECK_SECONDS
        self.server.requests.clear()
        await self.complete(tools=tools, tool_choice="auto")
        self.assertEqual(self.server.bodies("/chat/completions")[0]["tools"], tools)

    async def test_an_output_cap_that_overflows_the_window_is_refitted_once(self) -> None:
        await self.configure()

        def handler(request: httpx.Request) -> httpx.Response:
            if json.loads(request.content)["max_tokens"] > 32768 - 30000:
                return httpx.Response(400, json={"error": {"message": (
                    "This model's maximum context length is 32768 tokens. However, you requested 36096 tokens "
                    "(30000 in the messages, 6096 in the completion). Please reduce the length of the messages "
                    "or completion."
                )}})
            return httpx.Response(200, content=sse("fits"))

        self.server.chat_handler = handler
        self.server.requests.clear()
        result = await self.complete(max_tokens=6096)
        self.assertEqual(result["choices"][0]["message"]["content"], "fits")
        self.assertEqual([b["max_tokens"] for b in self.server.bodies("/chat/completions")], [6096, 2736])

    async def test_a_prompt_larger_than_the_window_is_a_context_limit_error(self) -> None:
        await self.configure()
        self.server.chat_handler = lambda request: httpx.Response(400, json={"error": {"message": (
            "This model's maximum context length is 32768 tokens. However, you requested 44096 tokens "
            "(40000 in the messages, 4096 in the completion)."
        )}})
        self.server.requests.clear()
        with self.assertRaises(SubscriptionContextLimitError) as ctx:
            await self.complete()
        self.assertIn("context_length_exceeded", str(ctx.exception))
        self.assertEqual(len(self.server.bodies("/chat/completions")), 1)


class VllmRoutingTests(VllmHarness):
    """vLLM roles through APIClientManager, with the registered client swapped for the harness client."""

    def setUp(self) -> None:
        super().setUp()
        registry = mock.patch.dict(
            "backend.shared.subscription_providers.registry._CLIENTS", {"vllm": self.client}
        )
        registry.start()
        self._patches.append(registry)

    def manager(self, *, fallback: str | None = None) -> APIClientManager:
        manager = APIClientManager()
        manager.configure_role("agg_sub1", ModelConfig(
            provider="vllm",
            model_id=CHAT_MODEL,
            lm_studio_fallback_id=fallback,
            context_window=32768,
            max_output_tokens=4096,
        ))
        return manager

    async def generate(self, manager: APIClientManager):
        return await manager._generate_completion_once(
            task_id="agg_sub1_001",
            role_id="agg_sub1",
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "hello"}],
        )

    async def test_a_vllm_role_completes_through_the_router(self) -> None:
        await self.configure()
        result = await self.generate(self.manager())
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")

    async def test_a_stopped_server_is_a_retryable_failure_not_a_repair_stop(self) -> None:
        await self.configure()
        self.server.down.add("127.0.0.1:8001")
        manager = self.manager()
        wait = mock.AsyncMock()
        with mock.patch.object(manager, "_provider_retry_callback", return_value=wait):
            with self.assertRaises(RetryableProviderError):
                await self.generate(manager)
        self.assertEqual(wait.await_count, 3)
        self.assertEqual(wait.await_args.args[0]["provider_label"], "vLLM")
        self.assertEqual(wait.await_args.args[0]["reason"], "connection_error")

    async def test_a_stopped_server_uses_the_lm_studio_fallback_for_that_call_only(self) -> None:
        await self.configure()
        self.server.down.add("127.0.0.1:8001")
        manager = self.manager(fallback="local-fallback")
        with (
            mock.patch.object(manager, "_provider_retry_callback", return_value=mock.AsyncMock()),
            mock.patch(
                "backend.shared.api_client_manager.lm_studio_client.generate_completion",
                new=mock.AsyncMock(return_value={"choices": [{"message": {"content": "fallback"}}]}),
            ) as lm_studio,
        ):
            result = await self.generate(manager)
        self.assertEqual(result["choices"][0]["message"]["content"], "fallback")
        self.assertEqual(lm_studio.await_args.kwargs["model"], "local-fallback")
        self.assertEqual(manager.get_fallback_state("agg_sub1"), "vllm")

    async def test_a_too_large_prompt_enters_the_context_overflow_lifecycle(self) -> None:
        await self.configure()
        self.server.chat_handler = lambda request: httpx.Response(400, json={"error": {"message": (
            "This model's maximum context length is 32768 tokens. However, you requested 44096 tokens "
            "(40000 in the messages, 4096 in the completion)."
        )}})
        with self.assertRaises(ProviderContextLengthError):
            await self.generate(self.manager())

    async def test_an_unrecoverable_failure_names_the_vllm_setup_not_an_api_key(self) -> None:
        await self.configure()
        self.server.chat_handler = lambda request: httpx.Response(400, json={"error": {"message": "bad request"}})
        manager = self.manager()
        with mock.patch.object(manager, "_broadcast", new=mock.AsyncMock()) as broadcast:
            with self.assertRaises(ProviderRepairRequiredError):
                await self.generate(manager)
        payloads = [c.args[1] for c in broadcast.await_args_list if c.args[0] == "subscription_provider_error"]
        self.assertEqual(len(payloads), 1)
        self.assertIn("Inference Connectivity > vLLM", payloads[0]["message"])
        self.assertNotIn("API key in OpenRouter/OAuth", payloads[0]["message"])

    async def test_hosted_mode_never_routes_to_vllm(self) -> None:
        with mock.patch.object(system_config, "generic_mode", True):
            manager = self.manager()
            self.assertEqual(manager._role_model_configs["agg_sub1"].provider, "openrouter")


class VllmFallbackTests(VllmHarness):
    """A role on another provider whose local fallback is a vLLM model (``vllm::<model>``)."""

    FALLBACK = f"vllm::{CHAT_MODEL}"

    def setUp(self) -> None:
        super().setUp()
        registry = mock.patch.dict(
            "backend.shared.subscription_providers.registry._CLIENTS", {"vllm": self.client}
        )
        registry.start()
        self._patches.append(registry)
        lm_studio = mock.patch(
            "backend.shared.api_client_manager.lm_studio_client.generate_completion",
            new=mock.AsyncMock(side_effect=AssertionError("LM Studio must not serve a vLLM fallback")),
        )
        self.lm_studio = lm_studio.start()
        self._patches.append(lm_studio)

    def manager(self, provider: str, *, fallback: str | None = None) -> APIClientManager:
        manager = APIClientManager()
        manager.configure_role("agg_val", ModelConfig(
            provider=provider,
            model_id="cloud-model",
            openrouter_model_id="cloud-model" if provider == "openrouter" else None,
            lm_studio_fallback_id=self.FALLBACK if fallback is None else fallback,
            context_window=32768,
            max_output_tokens=4096,
        ))
        return manager

    async def generate(self, manager: APIClientManager):
        return await manager._generate_completion_once(
            task_id="agg_val_001",
            role_id="agg_val",
            model="cloud-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_a_fallback_value_names_lm_studio_unless_tagged_with_a_local_server(self) -> None:
        self.assertEqual(split_local_fallback(self.FALLBACK), ("vllm", CHAT_MODEL))
        self.assertEqual(split_local_fallback("model:2"), ("lm_studio", "model:2"))
        self.assertEqual(split_local_fallback("vllm::"), ("lm_studio", "vllm::"))
        self.assertEqual(split_local_fallback(None), ("lm_studio", ""))
        self.assertEqual(local_fallback_id("vllm", CHAT_MODEL), self.FALLBACK)
        self.assertEqual(local_fallback_id("lm_studio", "model:2"), "model:2")
        self.assertEqual(role_label("aggregator_validator", "vllm_fallback"), "Validator (vLLM fallback)")
        self.assertEqual(role_label("aggregator_validator", "fallback"), "Validator (LM Studio fallback)")

    async def test_a_usage_limited_cloud_role_is_served_by_its_vllm_fallback(self) -> None:
        await self.configure()
        manager = self.manager("sakana_fugu")
        limited = SubscriptionUsageLimitError(
            "Sakana Fugu usage limit", provider="sakana_fugu", provider_label="Sakana Fugu", resets_in_seconds=120,
        )
        self.server.requests.clear()
        with (
            mock.patch.object(
                get_api_key_subscription_client("sakana_fugu"), "generate_completion", mock.AsyncMock(side_effect=limited)
            ),
            mock.patch.object(manager, "_broadcast", new=mock.AsyncMock()) as broadcast,
        ):
            result = await self.generate(manager)

        self.assertEqual(result["choices"][0]["message"]["content"], "OK")
        metadata = result[APIClientManager.CALL_METADATA_KEY]
        self.assertEqual((metadata["effective_provider"], metadata["effective_model"]), ("vllm", CHAT_MODEL))
        self.assertEqual(metadata["configured_provider"], "sakana_fugu")
        body = self.server.bodies("/chat/completions")[0]
        self.assertEqual((body["model"], body["max_tokens"]), (CHAT_MODEL, 4096))
        self.lm_studio.assert_not_awaited()

        limits = [c.args[1] for c in broadcast.await_args_list if c.args[0] == "oauth_provider_usage_limited"]
        self.assertEqual(len(limits), 1)
        self.assertEqual((limits[0]["fallback_provider"], limits[0]["fallback_model"]), ("vllm", CHAT_MODEL))
        self.assertIn(f"Using vLLM fallback model {CHAT_MODEL}", limits[0]["message"])
        self.assertNotIn("LM Studio", limits[0]["message"])

        # While the role stays on its fallback, its route identity names vLLM, not LM Studio.
        identity = manager.route_identity_for_role("agg_val")
        self.assertEqual((identity["provider"], identity["model"]), ("vllm", CHAT_MODEL))
        self.assertEqual(identity["route_kind"], "vllm_fallback")
        self.assertIn("vLLM fallback", identity["route_identity"])

    async def test_an_openrouter_role_without_a_key_uses_its_vllm_fallback(self) -> None:
        await self.configure()
        manager = self.manager("openrouter")
        with mock.patch.object(rag_config, "openrouter_api_key", None):
            result = await self.generate(manager)
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")
        self.assertEqual(result[APIClientManager.CALL_METADATA_KEY]["effective_provider"], "vllm")
        self.lm_studio.assert_not_awaited()

    async def test_a_stopped_vllm_fallback_server_is_retryable_and_names_the_fallback_route(self) -> None:
        await self.configure()
        self.server.down.add("127.0.0.1:8001")
        manager = self.manager("openrouter")
        wait = mock.AsyncMock()
        with (
            mock.patch.object(rag_config, "openrouter_api_key", None),
            mock.patch.object(manager, "_provider_retry_callback", return_value=wait) as retry_callback,
        ):
            with self.assertRaises(RetryableProviderError) as ctx:
                await self.generate(manager)
        self.assertEqual(retry_callback.call_args.kwargs["route_kind"], "vllm_fallback")
        self.assertEqual((ctx.exception.provider, ctx.exception.model), ("vllm", CHAT_MODEL))
        self.assertEqual(ctx.exception.route_kind, "vllm_fallback")
        self.assertEqual(ctx.exception.configured_provider, "openrouter")
        self.lm_studio.assert_not_awaited()

    async def test_a_fallback_model_no_vllm_server_serves_is_a_repair_stop_with_vllm_guidance(self) -> None:
        await self.configure()
        manager = self.manager("openrouter", fallback="vllm::missing/model")
        with mock.patch.object(rag_config, "openrouter_api_key", None):
            with self.assertRaises(ProviderRepairRequiredError) as ctx:
                await self.generate(manager)
        self.assertEqual((ctx.exception.provider, ctx.exception.model), ("vllm", "missing/model"))
        self.assertIn("Inference Connectivity", ctx.exception.terminal_guidance)
        self.lm_studio.assert_not_awaited()

    async def test_a_plain_fallback_value_still_goes_to_lm_studio(self) -> None:
        await self.configure()
        manager = self.manager("openrouter", fallback="local-model:2")
        self.lm_studio.side_effect = None
        self.lm_studio.return_value = {"choices": [{"message": {"content": "from lm studio"}}]}
        self.server.requests.clear()
        with mock.patch.object(rag_config, "openrouter_api_key", None):
            result = await self.generate(manager)
        self.assertEqual(result["choices"][0]["message"]["content"], "from lm studio")
        self.assertEqual(self.lm_studio.await_args.kwargs["model"], "local-model:2")
        self.assertEqual(self.server.bodies("/chat/completions"), [])

    async def test_hosted_mode_drops_a_vllm_fallback(self) -> None:
        with mock.patch.object(system_config, "generic_mode", True):
            manager = self.manager("openrouter")
            self.assertIsNone(manager._role_model_configs["agg_val"].lm_studio_fallback_id)


class VllmEmbeddingTests(VllmHarness):
    def setUp(self) -> None:
        super().setUp()
        for target in (
            "backend.shared.api_client_manager.vllm_client",
            "backend.shared.embedding_readiness.vllm_client",
        ):
            patcher = mock.patch(target, self.client)
            patcher.start()
            self._patches.append(patcher)
        lm_down = mock.patch(
            "backend.shared.api_client_manager.lm_studio_client.get_embeddings",
            new=mock.AsyncMock(side_effect=httpx.ConnectError("LM Studio is not running")),
        )
        self.lm_studio_embeddings = lm_down.start()
        self._patches.append(lm_down)
        lm_ready_down = mock.patch(
            "backend.shared.embedding_readiness.lm_studio_client.get_embeddings",
            new=mock.AsyncMock(side_effect=httpx.ConnectError("LM Studio is not running")),
        )
        lm_ready_down.start()
        self._patches.append(lm_ready_down)
        embedding_readiness._lm_studio_embedding_status_cache = None
        self.manager = APIClientManager()

    async def test_a_direct_embedding_retry_uses_the_shared_stop_aware_wait(self) -> None:
        await self.configure()
        calls = {"count": 0}
        original = self.server.__call__

        def flaky(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/embeddings"):
                calls["count"] += 1
                if calls["count"] == 1:
                    return httpx.Response(503, text="engine restarting")
            return original(request)

        self.client.client = httpx.AsyncClient(transport=httpx.MockTransport(flaky))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            vectors = await self.client.get_embeddings(["x"], max_attempts=3)
        self.assertEqual(len(vectors), 1)
        wait.assert_awaited_once()
        self.assertEqual(wait.await_args.kwargs["role_id"], "embeddings")
        payload = wait.await_args.args[0]
        self.assertEqual((payload["reason"], payload["max_attempts"], payload["retry_attempt"]), ("embedding_retry", 3, 1))

    async def test_a_text_longer_than_the_embedding_window_is_resent_once_cut_to_fit(self) -> None:
        await self.configure()
        self.server.embedding_windows[EMBED_MODEL] = 8
        self.server.requests.clear()
        with self.assertLogs("backend.shared.subscription_providers.vllm", level="WARNING") as logs:
            vectors = await self.client.get_embeddings(["short text", " ".join(["word"] * 20)])
            await self.client.get_embeddings([" ".join(["word"] * 30)])
        self.assertEqual(len(vectors), 2)
        bodies = self.server.bodies("/embeddings")
        self.assertEqual(
            [body.get("truncate_prompt_tokens") for body in bodies], [None, 8, None, 8]
        )
        # One explanation per model, not one per batch.
        self.assertEqual(len([line for line in logs.output if "cut to fit" in line]), 1)

    async def test_the_readiness_check_does_not_wait_behind_running_embedding_work(self) -> None:
        await self.configure()
        embedding_readiness.reset_vllm_embedding_status_cache()
        async with self.client._embedding_semaphore, self.client._embedding_semaphore:
            status = await embedding_readiness.check_vllm_embedding_ready(timeout_seconds=2.0, force_refresh=True)
        self.assertTrue(status["ready"])

    async def test_embeddings_are_batched_in_order_on_the_embedding_server(self) -> None:
        await self.configure()
        self.server.requests.clear()
        vectors = await self.client.get_embeddings([f"text {index}" for index in range(70)])
        self.assertEqual(len(vectors), 70)
        bodies = self.server.bodies("/embeddings")
        self.assertEqual([len(body["input"]) for body in bodies], [64, 6])
        self.assertTrue(all(body["model"] == EMBED_MODEL for body in bodies))
        hosts = {r.url.host for r in self.server.requests if r.url.path.endswith("/embeddings")}
        self.assertEqual(hosts, {"gpu-box"})

    async def test_rag_uses_vllm_when_lm_studio_is_absent_and_pins_the_space(self) -> None:
        await self.configure()
        vectors, space = await self.manager.get_embeddings_in_space(["alpha", "beta"])
        self.assertEqual(space, f"vllm:{EMBED_MODEL}")
        self.assertEqual(len(vectors), 2)

        self.lm_studio_embeddings.reset_mock()
        vectors, pinned = await self.manager.get_embeddings_in_space(["gamma"], embedding_space=space)
        self.assertEqual(pinned, space)
        self.assertEqual(len(vectors), 1)
        self.lm_studio_embeddings.assert_not_awaited()

    async def test_a_pinned_vllm_index_retries_vllm_and_never_mixes_another_model(self) -> None:
        await self.configure()
        space = self.client.embedding_space()
        self.server.down.add("gpu-box:8002")
        self.manager._openrouter_client = mock.Mock(get_embeddings=mock.AsyncMock(return_value=[[1.0]]))
        wait = mock.AsyncMock()
        with mock.patch.object(self.manager, "_provider_retry_callback", return_value=wait):
            with self.assertRaises(RuntimeError) as ctx:
                await self.manager.get_embeddings_in_space(["x"], embedding_space=space)
        self.assertIn("vLLM", str(ctx.exception))
        self.assertEqual(wait.await_count, 7)
        self.assertEqual(wait.await_args.args[0]["reason"], "embedding_retry")
        self.manager._openrouter_client.get_embeddings.assert_not_awaited()

    async def test_a_pinned_vllm_index_fails_clearly_after_the_embedding_model_is_removed(self) -> None:
        await self.configure()
        space = self.client.embedding_space()
        await self.client.clear_configuration()
        with self.assertRaises(RuntimeError) as ctx:
            await self.manager.get_embeddings_in_space(["x"], embedding_space=space)
        self.assertIn("no longer set up", str(ctx.exception))

    async def test_without_a_vllm_embedding_model_the_existing_fallback_is_unchanged(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            await self.manager.get_embeddings_in_space(["x"])
        self.assertIn("LM Studio is down and OpenRouter is not configured", str(ctx.exception))

    async def test_openrouter_is_still_the_last_embedding_fallback(self) -> None:
        await self.configure()
        self.server.down.add("gpu-box:8002")
        self.manager._openrouter_client = mock.Mock(get_embeddings=mock.AsyncMock(return_value=[[1.0]]))
        _vectors, space = await self.manager.get_embeddings_in_space(["x"])
        self.assertTrue(space.startswith("openrouter:"))

    async def test_a_vllm_embedding_model_satisfies_the_workflow_start_requirement(self) -> None:
        with mock.patch.object(rag_config, "openrouter_api_key", None), \
                mock.patch.object(rag_config, "openrouter_enabled", False):
            with self.assertRaises(ValueError) as ctx:
                await embedding_readiness.require_embedding_provider_ready()
            self.assertIn("vLLM", str(ctx.exception))

            await self.configure()
            embedding_readiness.reset_vllm_embedding_status_cache()
            await embedding_readiness.require_embedding_provider_ready()
            status = await embedding_readiness.check_embedding_provider_ready(force_refresh=True)
            self.assertEqual((status["ready"], status["provider"]), (True, "vllm"))

            self.server.down.add("gpu-box:8002")
            with self.assertRaises(ValueError):
                await embedding_readiness.require_embedding_provider_ready()

    async def test_availability_reports_chat_models_and_embedding_readiness(self) -> None:
        empty = await embedding_readiness.check_vllm_availability()
        self.assertFalse(empty["configured"])
        self.assertFalse(empty["available"])
        requests_before = len(self.server.requests)
        await embedding_readiness.check_vllm_availability()
        self.assertEqual(len(self.server.requests), requests_before)

        await self.configure()
        embedding_readiness.reset_vllm_embedding_status_cache()
        ready = await embedding_readiness.check_vllm_availability(force_refresh=True)
        self.assertTrue(ready["available"])
        self.assertTrue(ready["has_usable_chat_model"])
        self.assertEqual(ready["models"], [CHAT_MODEL])
        self.assertTrue(ready["embedding_ready"])
        self.assertEqual(ready["embedding_model"], EMBED_MODEL)


class VllmRouteTests(VllmHarness):
    def setUp(self) -> None:
        super().setUp()
        for target in (
            "backend.api.routes.cloud_access.vllm_client",
            "backend.shared.embedding_readiness.vllm_client",
        ):
            patcher = mock.patch(target, self.client)
            patcher.start()
            self._patches.append(patcher)
        lm_down = mock.patch(
            "backend.shared.embedding_readiness.lm_studio_client.get_embeddings",
            new=mock.AsyncMock(side_effect=httpx.ConnectError("LM Studio is not running")),
        )
        lm_down.start()
        self._patches.append(lm_down)

    async def test_test_route_checks_servers_without_saving(self) -> None:
        result = await cloud_access_route.test_vllm_servers(
            cloud_access_route.VllmTestRequest(servers=["127.0.0.1:8001", "gpu-box:8002"])
        )
        self.assertEqual([m["id"] for m in result["chat_models"]], [CHAT_MODEL])
        self.assertEqual(result["embedding_model"], EMBED_MODEL)
        self.assertEqual(get_vllm_runtime_settings(), {})

    async def test_config_route_saves_and_returns_models_with_context(self) -> None:
        with mock.patch.object(
            cloud_access_route.api_client_manager, "reset_provider_fallbacks", new=mock.AsyncMock()
        ) as reset:
            result = await cloud_access_route.set_vllm_config(
                cloud_access_route.VllmConfigRequest(servers=["127.0.0.1:8001", "gpu-box:8002"], api_key=KEY)
            )
        reset.assert_awaited_once_with("vllm")
        self.assertTrue(result["status"]["configured"])
        self.assertEqual(result["models"][0]["context_length"], 32768)
        self.assertNotIn(KEY, json.dumps(result))

        status = await cloud_access_route.get_vllm_status()
        self.assertEqual(status["status"]["servers"], [CHAT_SERVER, EMBED_SERVER])
        self.assertEqual(status["status"]["chat_models"], [CHAT_MODEL])
        models = await cloud_access_route.get_vllm_models()
        self.assertEqual([m["id"] for m in models["models"]], [CHAT_MODEL])
        availability = await cloud_access_route.get_vllm_availability(refresh=True)
        self.assertTrue(availability["has_usable_chat_model"])
        self.assertTrue(availability["embedding_ready"])

    async def test_bad_addresses_and_unreachable_servers_are_400s(self) -> None:
        for servers in ([], ["http://127.0.0.1:8000"], ["ftp://x"]):
            with self.subTest(servers=servers), self.assertRaises(HTTPException) as ctx:
                await cloud_access_route.set_vllm_config(cloud_access_route.VllmConfigRequest(servers=servers))
            self.assertEqual(ctx.exception.status_code, 400)
        self.server.down.add("127.0.0.1:8001")
        with self.assertRaises(HTTPException) as ctx:
            await cloud_access_route.set_vllm_config(cloud_access_route.VllmConfigRequest(servers=["127.0.0.1:8001"]))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(get_vllm_runtime_settings(), {})

    async def test_clear_route_removes_the_setup(self) -> None:
        await self.configure(api_key=KEY)
        await cloud_access_route.clear_vllm_config()
        self.assertFalse((await cloud_access_route.get_vllm_status())["status"]["configured"])
        with self.assertRaises(HTTPException) as ctx:
            await cloud_access_route.get_vllm_models()
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_a_key_that_cannot_be_removed_leaves_the_setup_unchanged(self) -> None:
        await self.configure(api_key=KEY)
        broken = SecretStoreError("Failed to delete the credential from the OS keyring.")
        with mock.patch(f"{_CLIENT_MODULE}.clear_subscription_api_key", side_effect=broken):
            with self.assertRaises(HTTPException) as ctx:
                await cloud_access_route.clear_vllm_config()
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertNotIn(KEY, str(ctx.exception.detail))
        status = (await cloud_access_route.get_vllm_status())["status"]
        self.assertTrue(status["configured"])
        self.assertTrue(status["has_api_key"])

    async def test_hosted_mode_rejects_setup_and_reports_unavailable(self) -> None:
        with mock.patch.object(system_config, "generic_mode", True):
            status = await cloud_access_route.get_vllm_status()
            self.assertEqual(status["status"], {"configured": False, "available": False, "desktop_only": True})
            availability = await cloud_access_route.get_vllm_availability()
            self.assertFalse(availability["available"])
            for call in (
                cloud_access_route.get_vllm_models(),
                cloud_access_route.test_vllm_servers(cloud_access_route.VllmTestRequest(servers=["h:1"])),
                cloud_access_route.set_vllm_config(cloud_access_route.VllmConfigRequest(servers=["h:1"])),
                cloud_access_route.clear_vllm_config(),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    await call
                self.assertEqual(ctx.exception.status_code, 501)
            self.assertEqual(len(self.server.requests), 0)

    async def test_connectivity_status_reports_vllm_separately_from_cloud_access(self) -> None:
        with mock.patch.object(connectivity_route, "check_vllm_availability", embedding_readiness.check_vllm_availability):
            inactive = await connectivity_route._vllm_status()
            self.assertEqual((inactive["status"], inactive["configured"]), ("inactive", False))
            await self.configure()
            embedding_readiness.reset_vllm_embedding_status_cache()
            active = await connectivity_route._vllm_status()
        self.assertEqual(active["status"], "ACTIVE")
        self.assertTrue(active["has_usable_chat_model"])
        self.assertTrue(active["embedding_ready"])

        cloud = {"providers": {"vllm": {"configured": True}}}
        with mock.patch.object(connectivity_route, "get_cloud_access_status", mock.AsyncMock(return_value=cloud)), \
                mock.patch.object(rag_config, "openrouter_api_key", None):
            openrouter_oauth = await connectivity_route._openrouter_oauth_status()
        self.assertFalse(openrouter_oauth["oauth_configured"])
        self.assertEqual(get_cloud_provider("vllm").auth_kind, "local_server")
