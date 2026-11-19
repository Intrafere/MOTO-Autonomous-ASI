"""vLLM profile and client (user-run OpenAI-compatible inference servers).

Access method: one or more vLLM server URLs the user enters, plus an optional
API key (the value the server was started with through ``--api-key``). A vLLM
process usually serves one model, so several servers may be listed: MOTO reads
each server's ``/v1/models`` to learn which models it serves and each model's
``max_model_len`` (its context window), then sends every completion to the
server that serves the requested model.

A server that serves an embedding model can also supply MOTO's RAG embeddings
through ``/v1/embeddings``. Embedding models are found by asking each served
model for one embedding when the configuration is saved or tested.

Server URLs and the chosen embedding model are non-secret runtime settings; the
API key lives in the desktop keyring like every other provider key.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from backend.shared.secret_store import SecretStoreError
from backend.shared.subscription_providers.api_key_client import (
    RetryCallback,
    SubscriptionApiKeyClient,
    moto_user_agent,
)
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionParameterRejectedError,
    SubscriptionProviderError,
    SubscriptionRequestError,
)
from backend.shared.subscription_providers.metadata import get_cloud_provider
from backend.shared.subscription_providers.profile import (
    ApiKeyProviderProfile,
    ApiKeyProviderSettings,
    EndpointRegion,
)

logger = logging.getLogger(__name__)

VLLM_PROVIDER_ID = "vllm"
VLLM_EMBEDDING_SPACE = "vllm"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_BASE_URL_ENV_VAR = "MOTO_VLLM_BASE_URL"
MAX_VLLM_SERVERS = 8
SETUP_LOCATION = "Inference Connectivity > vLLM"
TOOLS_UNSUPPORTED_CODE = "tools_unsupported"
# Tokens left for the answer below which a too-large request is not worth resending.
_MIN_REFIT_OUTPUT_TOKENS = 512
_REFIT_SAFETY_TOKENS = 32
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

VLLM_SETTINGS = ApiKeyProviderSettings(
    metadata=get_cloud_provider(VLLM_PROVIDER_ID),
    secret_name="vllm_api_key",
    env_var_names=("MOTO_VLLM_API_KEY", "VLLM_API_KEY"),
    # Placeholder only: the client resolves each request's server from the user's list.
    regions=(EndpointRegion("local", "vLLM server", DEFAULT_VLLM_BASE_URL),),
    region_selection="user",
    wire_api="chat_completions_stream",
    known_models=(),
    default_model="",
    console_url="https://docs.vllm.ai/",
    entitlement_with_fallback="fallback",
    # A restarted local server should get its role back, so only definitive
    # failures move a role to its local fallback until reset.
    generic_error_fallback="non_transient_only",
    repair_hint=(
        f"Check that the vLLM server is running and reachable, review it in {SETUP_LOCATION}, and retry."
    ),
)


def normalize_vllm_server_url(raw: str) -> str:
    """Return a canonical ``scheme://host[:port]/.../v1`` base URL or raise ``ValueError``."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Enter a vLLM server address, for example http://127.0.0.1:8001.")
    if "://" not in text:
        text = f"http://{text}"
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"'{raw}' is not a valid vLLM server address.") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"'{raw}' is not a valid vLLM server address. Use http://host:port.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "A vLLM server address must not contain a user name, password, or query string. "
            "Enter the API key in its own field."
        )
    host = parsed.hostname.lower()
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return f"{parsed.scheme}://{netloc}{path}"


def _is_moto_backend_address(url: str) -> bool:
    from backend.shared.config import system_config

    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return (parsed.hostname or "") in _LOOPBACK_HOSTS and port == int(system_config.backend_port)


def normalize_vllm_servers(raw_servers: Any) -> Tuple[str, ...]:
    """Validate and de-duplicate a user-entered server list."""
    if isinstance(raw_servers, str):
        raw_servers = [part for part in re.split(r"[\s,]+", raw_servers) if part]
    servers: List[str] = []
    for raw in raw_servers or []:
        url = normalize_vllm_server_url(raw)
        if _is_moto_backend_address(url):
            port = urlsplit(url).port
            raise ValueError(
                f"{url} is MOTO's own backend, not a vLLM server. vLLM also defaults to port {port}; "
                "start vLLM on another port (for example `--port 8001`) and enter that address."
            )
        if url not in servers:
            servers.append(url)
    if len(servers) > MAX_VLLM_SERVERS:
        raise ValueError(f"MOTO supports up to {MAX_VLLM_SERVERS} vLLM servers.")
    return tuple(servers)


@dataclass(frozen=True)
class VllmConfig:
    servers: Tuple[str, ...] = ()
    embedding_model: str = ""
    embedding_server: str = ""
    # Every served model that answered an embedding request when last tested.
    embedding_models: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "servers": list(self.servers),
            "embedding_model": self.embedding_model,
            "embedding_server": self.embedding_server,
            "embedding_models": list(self.embedding_models),
        }


@dataclass
class VllmServerProbe:
    url: str
    reachable: bool = False
    error: str = ""
    # The typed failure behind ``error``; kept so callers can raise it unchanged.
    failure: Optional[SubscriptionProviderError] = None
    models: List[Dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {"url": self.url, "reachable": self.reachable, "error": self.error, "models": list(self.models)}


@dataclass
class VllmProbeResult:
    servers: List[VllmServerProbe]
    embedding_model: str = ""
    embedding_server: str = ""

    @property
    def chat_models(self) -> List[Dict[str, Any]]:
        return [model for server in self.servers for model in server.models if model["kind"] == "chat"]

    @property
    def embedding_models(self) -> List[Dict[str, Any]]:
        return [model for server in self.servers for model in server.models if model["kind"] == "embedding"]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "servers": [server.to_payload() for server in self.servers],
            "chat_models": self.chat_models,
            "embedding_models": self.embedding_models,
            "embedding_model": self.embedding_model,
            "embedding_server": self.embedding_server,
        }


class VllmProfile(ApiKeyProviderProfile):
    """Request shaping and error classification for vLLM's OpenAI-compatible server."""

    def apply_stream_options(self, payload: Dict[str, Any]) -> None:
        # Token accounting needs the usage chunk, which vLLM sends only on request.
        payload["stream_options"] = {"include_usage": True}

    def _error_message(self, response: httpx.Response) -> str:
        payload = self.response_payload(response)
        for item in self.iter_error_objects(payload):
            message = item.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        return (response.text or "").strip()

    @staticmethod
    def context_numbers(message: str) -> Tuple[Optional[int], Optional[int]]:
        """Return (context limit, prompt tokens) named in a vLLM context rejection."""
        lowered = message.lower()
        limit_match = (
            re.search(r"maximum context length is (\d+)", lowered)
            or re.search(r"maximum model length of (\d+)", lowered)
            or re.search(r"max_model_len\D{0,24}(\d+)", lowered)
        )
        prompt_match = (
            re.search(r"\((\d+) in the messages", lowered)
            or re.search(r"prompt contains (?:at least )?(\d+)", lowered)
            or re.search(r"prompt \(length (\d+)\)", lowered)
            or re.search(r"request has (\d+) input tokens", lowered)
        )
        return (
            int(limit_match.group(1)) if limit_match else None,
            int(prompt_match.group(1)) if prompt_match else None,
        )

    def classify_error(self, response: httpx.Response) -> Optional[SubscriptionProviderError]:
        from backend.shared.openrouter_client import sanitize_provider_error_text

        status = response.status_code
        path = response.request.url.path.rstrip("/")
        if status == 404 and path.endswith("/models"):
            return None
        message = self._error_message(response)
        lowered = message.lower()
        safe = sanitize_provider_error_text(message)
        label = self.label
        if status in {401, 403}:
            return SubscriptionAuthError(
                f"{label} rejected MOTO's API key: {safe}",
                provider=self.provider_id,
                provider_label=label,
                status_code=status,
                terminal_guidance=(
                    "The vLLM server requires the API key it was started with (--api-key). "
                    f"Enter that key in {SETUP_LOCATION}, then retry."
                ),
            )
        if status in {400, 413, 422} and (
            "maximum context length" in lowered
            or "maximum model length" in lowered
            or "max_model_len" in lowered
            or ("context length" in lowered and ("exceed" in lowered or "longer than" in lowered))
        ):
            error = SubscriptionContextLimitError(
                f"{label} rejected the request: context_length_exceeded ({safe})",
                provider=self.provider_id,
                provider_label=label,
                status_code=status,
                error_code="context_length_exceeded",
            )
            error.context_limit, error.prompt_tokens = self.context_numbers(message)
            return error
        if status in {400, 422} and (
            "enable-auto-tool-choice" in lowered
            or "tool-call-parser" in lowered
            or ("tool choice" in lowered and "requires" in lowered)
            or ("tool" in lowered and "not supported" in lowered)
        ):
            return SubscriptionParameterRejectedError(
                f"{label} rejected a request setting: {safe}",
                provider=self.provider_id,
                provider_label=label,
                status_code=status,
                error_code=TOOLS_UNSUPPORTED_CODE,
                terminal_guidance=(
                    "This vLLM server was started without tool calling. Restart it with "
                    "--enable-auto-tool-choice and a --tool-call-parser for the model to use tools."
                ),
            )
        if status == 404 and path.endswith(("/chat/completions", "/embeddings")):
            return SubscriptionRequestError(
                f"{label} completion failed: {safe}",
                provider=self.provider_id,
                provider_label=label,
                status_code=status,
                error_code="invalid_model",
                terminal_guidance=(
                    "The vLLM server does not serve the model this role is set to. Select one of the "
                    "detected vLLM models in Settings, or start a vLLM server for this model and add it "
                    f"in {SETUP_LOCATION}."
                ),
            )
        return None


class VllmClient(SubscriptionApiKeyClient):
    """Client for the user's vLLM servers: model discovery, completions, and embeddings."""

    DISCOVERY_TTL_SECONDS = 30.0
    MODEL_LIST_TIMEOUT_SECONDS = 5.0
    EMBEDDING_PROBE_TIMEOUT_SECONDS = 30.0
    EMBEDDING_PROBE_TEXT = "MOTO embedding readiness check"
    EMBEDDING_BATCH_SIZE = 64
    EMBEDDING_RETRY_DELAY = 0.5
    MAX_EMBEDDING_RETRY_DELAY = 15.0
    # A server restarted with tool calling enabled gets tools offered again after this.
    TOOLS_RECHECK_SECONDS = 600.0

    def __init__(self, profile: ApiKeyProviderProfile) -> None:
        super().__init__(profile)
        self._config: Optional[VllmConfig] = None
        # Data root the cached settings were read from; a rebound root reloads them.
        self._config_root = ""
        self._model_servers: Dict[str, str] = {}
        self._model_context: Dict[str, int] = {}
        self._discovery: Dict[str, VllmServerProbe] = {}
        self._discovered_at = 0.0
        self._discovery_lock = asyncio.Lock()
        self._embedding_semaphore = asyncio.Semaphore(2)
        # model -> when its server last refused tool calling.
        self._tools_unsupported_models: Dict[str, float] = {}
        # Whether the keyring was already asked for a key (most servers need none).
        self._stored_key_checked = False
        self._ignored_env_servers = ""
        self._truncated_embedding_models: set[str] = set()

    # --- key --------------------------------------------------------------------------
    def _load_api_key(self) -> Optional[str]:
        """Return the optional API key, asking the keyring once.

        A vLLM server normally needs no key, so "no key" is remembered instead of
        re-read on every request, and an unavailable keyring means "no stored key"
        rather than a failed request.
        """
        if self._api_key:
            return self._api_key
        if self._stored_key_checked:
            return None
        try:
            key = super()._load_api_key()
        except SecretStoreError as exc:
            logger.warning(
                "The system keyring is unavailable (%s); vLLM continues without a stored API key.",
                type(exc).__name__,
            )
            key = None
        self._stored_key_checked = True
        return key

    # --- configuration --------------------------------------------------------------
    def _env_servers(self) -> Tuple[str, ...]:
        raw = (os.getenv(VLLM_BASE_URL_ENV_VAR) or "").strip()
        if not raw:
            return ()
        try:
            return normalize_vllm_servers(raw)
        except ValueError as exc:
            if raw != self._ignored_env_servers:
                self._ignored_env_servers = raw
                logger.warning("Ignoring %s: %s", VLLM_BASE_URL_ENV_VAR, exc)
            return ()

    def config(self) -> VllmConfig:
        from backend.shared.config import system_config

        root = str(system_config.data_dir)
        if self._config is None or self._config_root != root:
            from backend.shared.runtime_settings import get_vllm_runtime_settings

            if self._config is not None:
                # The data root changed under a running client.
                self._forget_discovery()
                self._stored_key_checked = False
            self._config_root = root
            stored = get_vllm_runtime_settings()
            try:
                servers = normalize_vllm_servers(stored.get("servers") or [])
            except ValueError as exc:
                logger.warning("Ignoring the saved vLLM server list: %s", exc)
                servers = ()
            self._config = VllmConfig(
                servers=servers,
                embedding_model=str(stored.get("embedding_model") or "").strip(),
                embedding_server=str(stored.get("embedding_server") or "").strip(),
                embedding_models=tuple(
                    str(item) for item in (stored.get("embedding_models") or []) if isinstance(item, str) and item
                ),
            )
        env_servers = self._env_servers()
        if env_servers:
            return VllmConfig(
                servers=env_servers,
                embedding_model=self._config.embedding_model,
                embedding_server=self._config.embedding_server,
                embedding_models=self._config.embedding_models,
            )
        return self._config

    def servers(self) -> Tuple[str, ...]:
        return self.config().servers

    @property
    def base_url(self) -> str:
        servers = self.servers()
        return servers[0] if servers else DEFAULT_VLLM_BASE_URL

    def _forget_discovery(self) -> None:
        self._model_servers.clear()
        self._model_context.clear()
        self._discovery = {}
        self._discovered_at = 0.0
        self._chat_base_urls.clear()
        self._tools_unsupported_models.clear()

    def _store_config(self, config: Optional[VllmConfig]) -> None:
        from backend.shared.runtime_settings import save_vllm_runtime_settings

        from backend.shared.config import system_config

        save_vllm_runtime_settings(config.to_payload() if config and config.servers else None)
        self._config = config if config and config.servers else VllmConfig()
        self._config_root = str(system_config.data_dir)
        self._forget_discovery()

    async def clear_configuration(self) -> None:
        """Forget the stored API key, the servers, and the embedding choice."""
        had_key = bool(self._load_api_key())
        try:
            await self.clear_api_key()
        except SecretStoreError:
            if had_key:
                raise
        self._stored_key_checked = True
        self._store_config(None)

    async def status(self) -> Dict[str, Any]:
        config = self.config()
        has_embedding = bool(config.embedding_model and config.embedding_server in config.servers)
        return {
            "configured": bool(config.servers),
            "provider": self.provider_id,
            "servers": list(config.servers),
            "server_source": "environment" if self._env_servers() else "settings",
            "has_api_key": bool(self._load_api_key()),
            "embedding_model": config.embedding_model if has_embedding else "",
            "embedding_server": config.embedding_server if has_embedding else "",
            "embedding_configured": has_embedding,
            "default_server": DEFAULT_VLLM_BASE_URL,
            "max_servers": MAX_VLLM_SERVERS,
        }

    # --- headers ----------------------------------------------------------------------
    def _headers_for_key(self, key: Optional[str]) -> Dict[str, str]:
        # A vLLM server only checks a key when it was started with --api-key.
        headers = {"Content-Type": "application/json", "User-Agent": moto_user_agent()}
        if key:
            self._remember_secret(key)
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # --- discovery --------------------------------------------------------------------
    def _unreachable(self, server: str, detail: str) -> SubscriptionRequestError:
        return SubscriptionRequestError(
            f"{self.provider_label} connection failed: {server} did not answer ({detail}).",
            provider=self.provider_id,
            provider_label=self.provider_label,
        )

    async def _fetch_server_models(self, server: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get(
                f"{server}/models",
                headers=self._headers_for_key(api_key),
                timeout=self.MODEL_LIST_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise self._unreachable(server, "timed out") from exc
        except httpx.TransportError as exc:
            raise self._unreachable(server, self._detail(str(exc) or type(exc).__name__)) from exc
        if response.status_code >= 400:
            classified = await self._classified(response)
            if classified is not None:
                raise classified
            if response.status_code == 404:
                raise SubscriptionRequestError(
                    f"{self.provider_label} model list failed: {server} is not a vLLM OpenAI-compatible "
                    "server (no /v1/models).",
                    provider=self.provider_id,
                    provider_label=self.provider_label,
                    status_code=404,
                )
            raise self._http_error(response, "model list")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SubscriptionRequestError(
                f"{self.provider_label} model list failed: {server} returned an invalid model list.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            ) from exc
        records = payload.get("data") if isinstance(payload, dict) else None
        models: List[Dict[str, Any]] = []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            model_id = str(record.get("id") or "").strip()
            if not model_id:
                continue
            model: Dict[str, Any] = {"id": model_id, "server": server, "kind": "chat"}
            context_length = self.profile.coerce_positive_int(record.get("max_model_len"))
            if context_length:
                model["context_length"] = context_length
            models.append(model)
        return models

    async def _probe_server(self, server: str, api_key: Optional[str]) -> VllmServerProbe:
        probe = VllmServerProbe(url=server)
        try:
            probe.models = await self._fetch_server_models(server, api_key)
            probe.reachable = True
        except SubscriptionProviderError as exc:
            probe.error = str(exc)
            probe.failure = exc
        return probe

    def _apply_discovery(self, probes: List[VllmServerProbe]) -> None:
        embedding_models = set(self.config().embedding_models)
        model_servers: Dict[str, str] = {}
        model_context: Dict[str, int] = {}
        for probe in probes:
            for model in probe.models:
                if model["id"] in embedding_models:
                    model["kind"] = "embedding"
                model_servers.setdefault(model["id"], probe.url)
                if model.get("context_length"):
                    model_context.setdefault(model["id"], int(model["context_length"]))
        # A server that is briefly down keeps the routes learned while it was up; a
        # server that answered is trusted about what it no longer serves.
        unreachable = {probe.url for probe in probes if not probe.reachable}
        for model_id, server in self._model_servers.items():
            if server in unreachable:
                model_servers.setdefault(model_id, server)
        self._model_servers = model_servers
        self._model_context = {**self._model_context, **model_context}
        self._discovery = {probe.url: probe for probe in probes}
        self._discovered_at = time.monotonic()

    async def discover(self, *, force: bool = False) -> Dict[str, VllmServerProbe]:
        """Return each configured server's model list (cached briefly, single-flight)."""
        servers = self.servers()
        if not servers:
            return {}
        fresh = (
            self._discovered_at > 0
            and time.monotonic() - self._discovered_at < self.DISCOVERY_TTL_SECONDS
            and set(self._discovery) == set(servers)
        )
        if fresh and not force:
            return self._discovery
        async with self._discovery_lock:
            if not force and self._discovered_at > 0 and (
                time.monotonic() - self._discovered_at < self.DISCOVERY_TTL_SECONDS
                and set(self._discovery) == set(servers)
            ):
                return self._discovery
            key = self._load_api_key()
            probes = await asyncio.gather(*(self._probe_server(server, key) for server in servers))
            self._apply_discovery(list(probes))
            return self._discovery

    def _not_configured(self) -> SubscriptionRequestError:
        return SubscriptionRequestError(
            f"{self.provider_label} is not set up.",
            provider=self.provider_id,
            provider_label=self.provider_label,
            terminal_guidance=f"Add your vLLM server address in {SETUP_LOCATION}, then retry.",
        )

    def _model_record(self, model: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "id": model["id"],
            "name": model["id"],
            "pricing": {"prompt": "local", "completion": "local"},
            "provider_metadata": {
                "source": self.provider_id,
                "server": model["server"],
                "supports_reasoning_effort": False,
            },
        }
        if model.get("context_length"):
            record["context_length"] = int(model["context_length"])
        return record

    async def list_models(self, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the chat models the configured servers serve, with their context windows."""
        if not self.servers():
            raise self._not_configured()
        discovery = await self.discover(force=True)
        reachable = [probe for probe in discovery.values() if probe.reachable]
        if not reachable:
            raise next(iter(discovery.values())).failure
        seen: set[str] = set()
        records: List[Dict[str, Any]] = []
        for probe in reachable:
            for model in probe.models:
                if model["kind"] != "chat" or model["id"] in seen:
                    continue
                seen.add(model["id"])
                records.append(self._model_record(model))
        return records

    def cached_overview(self) -> Dict[str, Any]:
        """Return the last discovery without any network call (for status views)."""
        probes = [self._discovery[server] for server in self.servers() if server in self._discovery]
        chat_models = [model["id"] for probe in probes for model in probe.models if model["kind"] == "chat"]
        return {
            "checked": bool(probes),
            "reachable_servers": sum(1 for probe in probes if probe.reachable),
            "server_errors": {probe.url: probe.error for probe in probes if probe.error},
            "chat_models": list(dict.fromkeys(chat_models)),
        }

    # --- test and save ------------------------------------------------------------------
    async def _model_embeds(self, server: str, model_id: str, api_key: Optional[str]) -> Optional[int]:
        """Return the embedding dimension when ``model_id`` answers an embedding request."""
        try:
            response = await self.client.post(
                f"{server}/embeddings",
                json={"model": model_id, "input": [self.EMBEDDING_PROBE_TEXT]},
                headers=self._headers_for_key(api_key),
                timeout=self.EMBEDDING_PROBE_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                return None
            data = response.json().get("data")
            vector = data[0].get("embedding") if isinstance(data, list) and data else None
            return len(vector) if isinstance(vector, list) and vector else None
        except (httpx.HTTPError, ValueError, AttributeError, TypeError, KeyError, IndexError):
            return None

    async def probe(
        self,
        servers: Tuple[str, ...],
        api_key: Optional[str],
        *,
        embedding_model: str = "",
        keep_saved_embedding: bool = True,
    ) -> VllmProbeResult:
        """Check candidate servers without changing any stored or in-memory state.

        With no ``embedding_model`` the saved embedding model is kept while a listed
        server still serves it (a RAG index is tied to that model); otherwise the
        first embedding model found is used.
        """
        if not servers:
            raise ValueError("Enter at least one vLLM server address.")
        probes = list(await asyncio.gather(*(self._probe_server(server, api_key) for server in servers)))
        for probe in probes:
            if probe.failure is not None:
                raise probe.failure

        async def classify(probe: VllmServerProbe, model: Dict[str, Any]) -> None:
            dimension = await self._model_embeds(probe.url, model["id"], api_key)
            if dimension:
                model["kind"] = "embedding"
                model["embedding_dimension"] = dimension

        await asyncio.gather(*(classify(probe, model) for probe in probes for model in probe.models))
        result = VllmProbeResult(servers=probes)
        embedding_models = result.embedding_models
        wanted = (embedding_model or "").strip()
        if wanted:
            chosen = next((model for model in embedding_models if model["id"] == wanted), None)
            if chosen is None:
                raise ValueError(
                    f"'{wanted}' did not answer an embedding request on any listed vLLM server. "
                    "Serve an embedding model with vLLM (for example nomic-ai/nomic-embed-text-v1.5) "
                    "and choose it, or leave the embedding model on automatic."
                )
        else:
            saved = self.config() if keep_saved_embedding else VllmConfig()
            chosen = (
                next(
                    (
                        model for model in embedding_models
                        if model["id"] == saved.embedding_model and model["server"] == saved.embedding_server
                    ),
                    None,
                )
                or next((model for model in embedding_models if model["id"] == saved.embedding_model), None)
                or (embedding_models[0] if embedding_models else None)
            )
        if chosen is not None:
            result.embedding_model = chosen["id"]
            result.embedding_server = chosen["server"]
        return result

    async def configure(
        self,
        servers: Tuple[str, ...],
        *,
        api_key: Optional[str] = None,
        clear_api_key: bool = False,
        embedding_model: str = "",
    ) -> VllmProbeResult:
        """Validate the servers, then store them (and a new API key when one is given)."""
        candidate_key = (api_key or "").strip() or None
        if candidate_key is None and not clear_api_key:
            candidate_key = self._load_api_key()
        result = await self.probe(servers, candidate_key, embedding_model=embedding_model)
        if clear_api_key and not (api_key or "").strip():
            await self.clear_api_key()
        elif (api_key or "").strip():
            self.set_api_key(api_key.strip())
        self._store_config(VllmConfig(
            servers=tuple(servers),
            embedding_model=result.embedding_model,
            embedding_server=result.embedding_server,
            embedding_models=tuple(model["id"] for model in result.embedding_models),
        ))
        self._apply_discovery(result.servers)
        return result

    # --- completions --------------------------------------------------------------------
    async def _server_for_model(self, model: str) -> str:
        if not self.servers():
            raise self._not_configured()
        server = self._model_servers.get(model)
        if server in self.servers():
            return server
        discovery = await self.discover(force=True)
        server = self._model_servers.get(model)
        if server:
            return server
        unreachable = [probe for probe in discovery.values() if not probe.reachable]
        if unreachable:
            # The model may live on a server that is down: a retryable connection failure.
            raise self._request_failed(unreachable[0].error)
        raise SubscriptionRequestError(
            f"{self.provider_label} completion failed: no configured vLLM server serves model '{model}'.",
            provider=self.provider_id,
            provider_label=self.provider_label,
            error_code="invalid_model",
            terminal_guidance=(
                f"No vLLM server listed in {SETUP_LOCATION} serves '{model}'. Select one of the detected "
                "vLLM models in Settings, or start a vLLM server for this model and add its address."
            ),
        )

    def _tools_refused(self, model: str) -> bool:
        refused_at = self._tools_unsupported_models.get(model)
        if refused_at is None:
            return False
        if time.monotonic() - refused_at >= self.TOOLS_RECHECK_SECONDS:
            del self._tools_unsupported_models[model]
            return False
        return True

    def _refit_output_limit(self, error: SubscriptionContextLimitError, max_tokens: Optional[int]) -> Optional[int]:
        """Return a smaller output cap that fits the server's window, when the prompt itself fits."""
        limit = getattr(error, "context_limit", None)
        prompt_tokens = getattr(error, "prompt_tokens", None)
        if not limit or not prompt_tokens or not max_tokens:
            return None
        room = int(limit) - int(prompt_tokens) - _REFIT_SAFETY_TOKENS
        if room < _MIN_REFIT_OUTPUT_TOKENS or room >= int(max_tokens):
            return None
        return room

    async def generate_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        retry_callback: Optional[RetryCallback] = None,
        cache_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        server = await self._server_for_model(model)
        tools_refused = self._tools_refused(model)
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "tools": None if tools_refused else tools,
            "tool_choice": None if tools_refused else tool_choice,
            "retry_callback": retry_callback,
            "cache_scope": cache_scope,
        }
        refitted = False
        while True:
            # Set per send: a setup saved while this call waited clears learned routes.
            self._chat_base_urls[model] = server
            try:
                return await super().generate_completion(**request)
            except SubscriptionParameterRejectedError as exc:
                if exc.error_code != TOOLS_UNSUPPORTED_CODE or not request["tools"]:
                    raise
                # The server runs without tool calling: answer without tools, as a model
                # that ignores tool schemas would, and stop offering them to this model.
                logger.warning(
                    "vLLM server for model %s has tool calling disabled; continuing without tools.", model
                )
                self._tools_unsupported_models[model] = time.monotonic()
                request["tools"] = None
                request["tool_choice"] = None
            except SubscriptionContextLimitError as exc:
                # vLLM rejects a request whose prompt plus output cap exceeds the window.
                # When the prompt fits, resend once with the output cap that still fits.
                room = None if refitted else self._refit_output_limit(exc, request["max_tokens"])
                if room is None:
                    raise
                logger.info(
                    "vLLM model %s: prompt plus output cap exceeds its window; retrying with max_tokens=%s.",
                    model,
                    room,
                )
                refitted = True
                request["max_tokens"] = room

    # --- embeddings ---------------------------------------------------------------------
    def embedding_target(self) -> Optional[Tuple[str, str]]:
        """Return ``(server, model)`` for RAG embeddings, or None when none is set up."""
        config = self.config()
        if config.embedding_model and config.embedding_server in config.servers:
            return config.embedding_server, config.embedding_model
        return None

    def embedding_space(self) -> Optional[str]:
        target = self.embedding_target()
        return f"{VLLM_EMBEDDING_SPACE}:{target[1]}" if target else None

    async def _embed_batch(
        self,
        server: str,
        model: str,
        texts: List[str],
        *,
        retry_callback: Optional[RetryCallback],
        attempts: int,
    ) -> List[List[float]]:
        payload: Dict[str, Any] = {"model": model, "input": texts}
        attempt = 0
        while True:
            attempt += 1
            failure = ""
            try:
                response = await self.client.post(
                    f"{server}/embeddings",
                    json=payload,
                    headers=self._headers(),
                )
                if response.status_code < 400:
                    data = response.json()["data"]
                    return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
                classified = await self._classified(response)
                if isinstance(classified, SubscriptionContextLimitError) and "truncate_prompt_tokens" not in payload:
                    # vLLM refuses an input longer than the embedding model's window
                    # instead of cutting it. Resend once, cut to that window.
                    limit = getattr(classified, "context_limit", None) or self._model_context.get(model)
                    if limit:
                        payload["truncate_prompt_tokens"] = int(limit)
                        if model not in self._truncated_embedding_models:
                            self._truncated_embedding_models.add(model)
                            logger.warning(
                                "vLLM embedding model %s accepts %s tokens; longer RAG texts are cut to fit. "
                                "Use an embedding model with a larger window to embed them whole.",
                                model,
                                limit,
                            )
                        attempt -= 1
                        continue
                if classified is not None:
                    raise classified
                if not self.profile.is_transient_response(response):
                    raise self._http_error(response, "embedding request")
                failure = f"HTTP {response.status_code}: {self._detail(response.text)}"
            except httpx.TransportError as exc:
                failure = f"[{type(exc).__name__}] {self._detail(str(exc) or repr(exc))}"
            except (ValueError, KeyError, TypeError) as exc:
                raise SubscriptionRequestError(
                    f"{self.provider_label} embedding request returned an invalid response.",
                    provider=self.provider_id,
                    provider_label=self.provider_label,
                ) from exc
            if attempt >= attempts:
                raise self._request_failed(failure)
            await self._wait_before_retry(
                attempt=attempt - 1,
                delay=min(self.EMBEDDING_RETRY_DELAY * (2 ** (attempt - 1)), self.MAX_EMBEDDING_RETRY_DELAY),
                reason="embedding_retry",
                error_summary=failure[:1000],
                retry_model=model,
                retry_callback=retry_callback,
                max_attempts=attempts,
                role_id="embeddings",
            )

    async def check_embedding(self) -> bool:
        """Ask the configured embedding model for one embedding (readiness check).

        Sent outside the embedding work queue so a busy RAG ingestion does not make
        a healthy server look unavailable.
        """
        target = self.embedding_target()
        if target is None:
            return False
        vectors = await self._embed_batch(
            target[0], target[1], [self.EMBEDDING_PROBE_TEXT], retry_callback=None, attempts=1
        )
        return bool(vectors and vectors[0])

    async def get_embeddings(
        self,
        texts: List[str],
        *,
        retry_callback: Optional[RetryCallback] = None,
        max_attempts: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed texts with the configured vLLM embedding model.

        ``max_attempts`` defaults to failing fast so an absent server does not delay
        the next embedding provider; a caller pinned to this model passes more.
        """
        if not texts:
            return []
        target = self.embedding_target()
        if target is None:
            raise SubscriptionRequestError(
                f"{self.provider_label} has no embedding model set up.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        server, model = target
        attempts = max(1, int(max_attempts or 1))
        embeddings: List[List[float]] = []
        async with self._embedding_semaphore:
            for start in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
                embeddings.extend(await self._embed_batch(
                    server,
                    model,
                    texts[start:start + self.EMBEDDING_BATCH_SIZE],
                    retry_callback=retry_callback,
                    attempts=attempts,
                ))
        return embeddings


VLLM_PROFILE = VllmProfile(VLLM_SETTINGS)
