#!/usr/bin/env python3
"""moto - command-line control for MOTO Autonomous ASI.

Humans and AI agents (OpenClaw, Claude Code, shell scripts) use this to launch
MOTO, start/stop workflows, read results, and stream live activity. It talks to
the running desktop backend over the same authenticated REST/WebSocket routes the
browser uses, discovering the backend through ``<data root>/.moto_control.json``.

Agent contract: every command accepts ``--json`` (one JSON document on stdout;
``events`` prints one JSON object per line), never prompts, and exits with:
0 success, 1 error, 2 usage, 3 MOTO not running, 4 request rejected by MOTO.

This module is deliberately named ``moto_cli`` (not ``moto``) so it never shadows
the unrelated PyPI ``moto`` package. It needs only the standard library, except
``events``, which uses the ``websockets`` package MOTO already installs.
"""
from __future__ import annotations

import argparse
import copy
import getpass
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent
CONTROL_FILE_NAME = ".moto_control.json"
START_TEMPLATE_DIR_NAME = "cli_start_templates"
LAUNCHER_STATE_PATH = REPO_ROOT / ".moto_launcher_state.json"
LAST_INSTANCE_PATH = REPO_ROOT / ".moto_last_instance.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data"
SKILL_SOURCE = REPO_ROOT / "agent-skills" / "moto" / "SKILL.md"
TOKEN_HEADER = "X-Moto-Desktop-Token"
USER_AGENT = "moto-cli"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_RUNNING = 3
EXIT_REJECTED = 4

# Command names the installer manages. ``moto`` is skipped when the name already
# belongs to another program; ``moto-asi`` is always available as a fallback.
PRIMARY_COMMAND = "moto"
ALIAS_COMMAND = "moto-asi"
SHIM_MARKER = "MOTO-Autonomous-ASI command shim"


class CliError(Exception):
    """A user-facing failure with an exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    instance_id: str
    backend_url: str
    token: str
    data_root: Optional[Path] = None
    frontend_url: Optional[str] = None
    pid: Optional[int] = None
    version: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "backend_url": self.backend_url,
            "frontend_url": self.frontend_url,
            "data_root": str(self.data_root) if self.data_root else None,
            "pid": self.pid,
            "version": self.version,
        }


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def candidate_data_roots(explicit: Optional[str] = None) -> list[Path]:
    """Every data root a local MOTO backend might be using, most specific first."""
    if explicit:
        return [Path(explicit).expanduser().resolve()]
    roots: list[Path] = []
    env_root = os.environ.get("MOTO_DATA_ROOT") or os.environ.get("DATA_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())
    state = _read_json_file(LAUNCHER_STATE_PATH)
    if isinstance(state, dict):
        for record in state.get("instances") or []:
            if isinstance(record, dict) and record.get("data_root"):
                roots.append(Path(str(record["data_root"])))
    last = _read_json_file(LAST_INSTANCE_PATH)
    if isinstance(last, dict) and last.get("data_root"):
        roots.append(Path(str(last["data_root"])))
    roots.append(DEFAULT_DATA_ROOT)
    instances_dir = REPO_ROOT / ".moto_instances"
    if instances_dir.is_dir():
        roots.extend(sorted(path / "data" for path in instances_dir.iterdir() if path.is_dir()))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = os.path.normcase(str(root.resolve()))
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(root.resolve())
    return unique


def read_control_file(data_root: Path) -> Optional[Instance]:
    payload = _read_json_file(data_root / CONTROL_FILE_NAME)
    if not isinstance(payload, dict):
        return None
    url = str(payload.get("backend_url") or "").strip()
    token = str(payload.get("desktop_api_token") or "").strip()
    if not url.startswith("http") or not token:
        return None
    return Instance(
        instance_id=str(payload.get("instance_id") or "default"),
        backend_url=url.rstrip("/"),
        token=token,
        data_root=data_root,
        frontend_url=payload.get("frontend_url"),
        pid=payload.get("pid") if isinstance(payload.get("pid"), int) else None,
        version=payload.get("version"),
    )


def probe_health(backend_url: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        request = urllib.request.Request(
            f"{backend_url}/api/health", headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def token_is_current(instance: Instance, timeout: float = 3.0) -> bool:
    """True when the backend accepts this control file's per-launch token.

    A control file left by a killed backend can point at a port that a newer
    launch now serves (often with the same instance ID); only the current
    launch's token authenticates, so stale files are ignored.
    """
    request = urllib.request.Request(
        f"{instance.backend_url}/api/token-stats",
        headers={TOKEN_HEADER: instance.token, "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def discover_instances(explicit_root: Optional[str] = None) -> list[Instance]:
    """Running instances whose control file matches a healthy backend and its current token."""
    live: list[Instance] = []
    seen_urls: set[str] = set()
    for root in candidate_data_roots(explicit_root):
        instance = read_control_file(root)
        if instance is None or instance.backend_url in seen_urls:
            continue
        health = probe_health(instance.backend_url)
        if not health or str(health.get("instance_id")) != instance.instance_id:
            continue
        if not token_is_current(instance):
            continue
        instance.version = health.get("version") or instance.version
        seen_urls.add(instance.backend_url)
        live.append(instance)
    return live


def resolve_instance(args: argparse.Namespace) -> Instance:
    url = getattr(args, "url", None) or os.environ.get("MOTO_URL")
    if url:
        token = getattr(args, "token", None) or os.environ.get("MOTO_DESKTOP_API_TOKEN") or ""
        if not token:
            raise CliError("--url needs a token: pass --token or set MOTO_DESKTOP_API_TOKEN.", EXIT_USAGE)
        return Instance(instance_id="custom", backend_url=url.rstrip("/"), token=token)
    instances = discover_instances(getattr(args, "data_root", None))
    wanted = getattr(args, "instance", None)
    if wanted:
        instances = [item for item in instances if item.instance_id == wanted]
    if not instances:
        hint = f" named '{wanted}'" if wanted else ""
        raise CliError(
            f"No running MOTO instance{hint} was found. Start it with `{PRIMARY_COMMAND} launch` "
            "(or double-click the MOTO launcher) and try again.",
            EXIT_NOT_RUNNING,
        )
    if len(instances) > 1:
        defaults = [item for item in instances if item.instance_id == "default"]
        if len(defaults) == 1:
            return defaults[0]
        names = ", ".join(item.instance_id for item in instances)
        raise CliError(f"Several MOTO instances are running ({names}); choose one with --instance.", EXIT_USAGE)
    return instances[0]


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class MotoClient:
    def __init__(self, instance: Instance, timeout: float = 60.0):
        self.instance = instance
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        query: Any = None,
        raw: bool = False,
        files: Optional[list[tuple[str, str, bytes]]] = None,
        binary: bool = False,
    ) -> Any:
        """Call one route. ``files`` sends a multipart upload; ``binary`` returns (bytes, content type)."""
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.instance.backend_url}{path}"
        if query:
            pairs = query.items() if isinstance(query, dict) else query
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(
                [(key, value) for key, value in pairs if value is not None]
            )
        data = None
        headers = {TOKEN_HEADER: self.instance.token, "User-Agent": USER_AGENT, "Accept": "application/json"}
        if files:
            data, headers["Content-Type"] = encode_multipart(files)
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = _decode_body(exc.read(), exc.headers.get("Content-Type", ""))
            message = detail.get("detail") if isinstance(detail, dict) and "detail" in detail else detail
            raise CliError(_short(message) or f"HTTP {exc.code}", EXIT_REJECTED, {"status": exc.code, "detail": message})
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._timeout_error(method, path)
            raise CliError(f"Could not reach MOTO at {self.instance.backend_url}: {exc.reason}", EXIT_NOT_RUNNING)
        except TimeoutError:
            raise self._timeout_error(method, path)
        except OSError as exc:
            raise CliError(f"Could not reach MOTO at {self.instance.backend_url}: {exc}", EXIT_NOT_RUNNING)
        if binary:
            return payload, content_type
        if raw:
            return payload.decode("utf-8", errors="replace")
        return _decode_body(payload, content_type)

    def _timeout_error(self, method: str, path: str) -> CliError:
        return CliError(
            f"MOTO did not answer {method.upper()} {path} within {self.timeout:g} seconds. "
            "It may still be working on it; check `moto status` before retrying.",
            EXIT_ERROR,
        )

    def get(self, path: str, **query: Any) -> Any:
        return self.request("GET", path, query=query or None)

    def post(self, path: str, body: Any = None, **query: Any) -> Any:
        return self.request("POST", path, body=body, query=query or None)

    def delete(self, path: str, **query: Any) -> Any:
        return self.request("DELETE", path, query=query or None)


def encode_multipart(files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """Encode ``(field, filename, content)`` parts as multipart/form-data."""
    boundary = f"----moto-cli-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for field_name, filename, content in files:
        safe_field = "".join(ch for ch in field_name if ch not in '"\r\n')
        safe_name = "".join(ch for ch in filename if ch not in "\r\n").replace('"', "%22")
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{safe_field}"; filename="{safe_name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8")
            + content
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _decode_body(payload: bytes, content_type: str) -> Any:
    text = payload.decode("utf-8", errors="replace")
    if "json" in content_type or text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def _short(value: Any, limit: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("message") or value.get("reason") or value.get("detail") or json.dumps(value)
    elif isinstance(value, list):
        value = "; ".join(_short(item, 200) for item in value)
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def emit(args: argparse.Namespace, payload: Any, human: Optional[Callable[[Any], str]] = None) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return
    if human is not None:
        print(human(payload))
    elif isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Workflow modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowMode:
    name: str
    label: str
    prefix: str
    prompt_field: str
    clear: tuple[str, dict]
    aliases: tuple[str, ...] = ()
    allowed_outputs: bool = False


MODES: dict[str, WorkflowMode] = {
    "autonomous": WorkflowMode(
        "autonomous", "Autonomous Research", "/api/auto-research", "user_research_prompt",
        ("/clear", {"confirm": "true"}), aliases=("auto", "research"), allowed_outputs=True,
    ),
    "aggregator": WorkflowMode(
        "aggregator", "Aggregator", "/api/aggregator", "user_prompt", ("/clear-all", {}), aliases=("agg",),
    ),
    "compiler": WorkflowMode(
        "compiler", "Single Paper Writer", "/api/compiler", "compiler_prompt",
        ("/clear-paper", {"confirm": "true"}), aliases=("writer",), allowed_outputs=True,
    ),
    "leanoj": WorkflowMode(
        "leanoj", "LeanOJ Proof Solver", "/api/leanoj", "user_prompt",
        ("/clear", {"confirm": "true"}), aliases=("proof-solver",),
    ),
}


def _is_running(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    return status.get("is_running") is True


def template_path(instance: Instance, mode: str) -> Optional[Path]:
    if instance.data_root is None:
        return None
    return instance.data_root / START_TEMPLATE_DIR_NAME / f"{mode}.json"


def load_start_template(instance: Instance, mode: str) -> Optional[dict]:
    path = template_path(instance, mode)
    payload = _read_json_file(path) if path else None
    if isinstance(payload, dict) and isinstance(payload.get("request"), dict):
        return payload["request"]
    return None


def _read_text_arg(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return Path(value).expanduser().read_text(encoding="utf-8")


def _load_config_file(value: str) -> dict:
    try:
        payload = json.loads(_read_text_arg(value))
    except (OSError, ValueError) as exc:
        raise CliError(f"Could not read config {value}: {exc}", EXIT_USAGE)
    if isinstance(payload, dict) and isinstance(payload.get("request"), dict):
        payload = payload["request"]
    if not isinstance(payload, dict):
        raise CliError("A config file must hold a JSON object (a start request).", EXIT_USAGE)
    return payload


def _parse_set(pairs: Iterable[str]) -> list[tuple[str, Any]]:
    updates: list[tuple[str, Any]] = []
    for pair in pairs or ():
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise CliError(f"--set expects key=value, got '{pair}'.", EXIT_USAGE)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        updates.append((key.strip(), value))
    return updates


def _assign_path(node: Any, parts: list[str], value: Any, full_key: str) -> None:
    """Set ``value`` at a dotted path; list indices are numbers and ``*`` means every item."""
    head, rest = parts[0], parts[1:]
    if isinstance(node, list):
        if head == "*":
            indexes = range(len(node))
        else:
            try:
                index = int(head)
            except ValueError:
                raise CliError(f"--set {full_key}: '{head}' is not a list index (use a number or *).", EXIT_USAGE)
            if not -len(node) <= index < len(node):
                raise CliError(f"--set {full_key}: index {index} is out of range (the list has {len(node)} items).", EXIT_USAGE)
            indexes = [index]
        for index in indexes:
            if rest:
                _assign_path(node[index], rest, value, full_key)
            else:
                node[index] = copy.deepcopy(value)
        return
    if not isinstance(node, dict):
        raise CliError(f"--set {full_key}: cannot set a field inside a {type(node).__name__}.", EXIT_USAGE)
    keys = list(node) if head == "*" else [head]
    for key in keys:
        if not rest:
            node[key] = copy.deepcopy(value)
            continue
        if node.get(key) is None:
            node[key] = {}
        _assign_path(node[key], rest, value, full_key)


def apply_set_overrides(body: dict, pairs: Iterable[str]) -> dict:
    result = copy.deepcopy(body)
    for key, value in _parse_set(pairs):
        parts = key.split(".")
        if any(not part for part in parts):
            raise CliError(f"--set {key}: empty path segment.", EXIT_USAGE)
        _assign_path(result, parts, value, key)
    return result


PROVIDER_ALIASES = {
    "openrouter": "openrouter",
    "lm_studio": "lm_studio", "lmstudio": "lm_studio", "lm-studio": "lm_studio",
    "openai_codex_oauth": "openai_codex_oauth", "codex": "openai_codex_oauth",
    "xai_grok_oauth": "xai_grok_oauth", "grok": "xai_grok_oauth",
    "kimi_code": "kimi_code", "kimi-code": "kimi_code", "kimi": "kimi_code",
    "sakana_fugu": "sakana_fugu", "sakana-fugu": "sakana_fugu", "fugu": "sakana_fugu",
    "claude_api": "claude_api", "claude-api": "claude_api", "claude": "claude_api", "anthropic": "claude_api",
    "amazon_bedrock": "amazon_bedrock", "amazon-bedrock": "amazon_bedrock", "bedrock": "amazon_bedrock",
    "vllm": "vllm",
}


MIN_RUNTIME_SECONDS = 60
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
MAX_RUNTIME_HELP = ("Stop this run after this long, for example 45m, 2h, 1h30m, or 1d (default: no limit). "
                    "MOTO enforces it; progress is kept.")
THINK_MAX_RUNTIME_HELP = ("Explore for at most this long, for example 45m, 2h, 1h30m, or 1d, then answer from what was "
                          "explored (default: until MOTO finishes exploring on its own).")


def parse_duration(value: Any) -> int:
    """Seconds from '90m', '2h', '1h30m', '1.5h', or a plain number of seconds."""
    text = str(value).strip().lower().replace(" ", "")
    try:
        seconds = float(text)
    except ValueError:
        pieces = re.findall(r"(\d+(?:\.\d+)?)([smhd])", text)
        if not pieces or "".join(number + unit for number, unit in pieces) != text:
            raise CliError(f"'{value}' is not a duration. Use seconds or a form like 90m, 2h, 1h30m, or 1d.", EXIT_USAGE)
        seconds = sum(float(number) * _DURATION_UNITS[unit] for number, unit in pieces)
    if seconds != seconds or seconds < MIN_RUNTIME_SECONDS:  # NaN or too short
        raise CliError(f"A maximum runtime must be at least {MIN_RUNTIME_SECONDS} seconds.", EXIT_USAGE)
    return int(seconds)


def format_duration(seconds: Any) -> str:
    try:
        remaining = max(0, int(seconds))
    except (TypeError, ValueError):
        return ""
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, remaining = divmod(remaining, size)
        if count:
            parts.append(f"{count}{unit}")
    return " ".join(parts) or "0s"


@dataclass(frozen=True)
class RoleSpec:
    provider: str
    model: str
    context: int
    max_tokens: int
    host: Optional[str] = None
    reasoning: Optional[str] = None


def _openrouter_limits(client: Optional["MotoClient"], model: str) -> tuple[Optional[int], Optional[int]]:
    """Context window and max output from OpenRouter metadata (20% of context, capped by the top host)."""
    if client is None:
        return None, None
    try:
        listing = client.get("/api/openrouter/models")
    except CliError:
        return None, None
    models = listing.get("models") if isinstance(listing, dict) else None
    for entry in models or []:
        if isinstance(entry, dict) and entry.get("id") == model:
            context = entry.get("context_length")
            cap = (entry.get("top_provider") or {}).get("max_completion_tokens")
            context = int(context) if isinstance(context, (int, float)) and context > 0 else None
            if context is None:
                return None, None
            output = int(context * 0.2)
            if isinstance(cap, (int, float)) and cap > 0:
                output = min(output, int(cap))
            else:
                return context, None  # unknown host output cap: never guess one
            return context, output
    raise CliError(f"OpenRouter does not list a model named '{model}'. Check the model ID (for example openai/gpt-5).", EXIT_USAGE)


def _vllm_limits(client: Optional["MotoClient"], model: str) -> tuple[Optional[int], Optional[int]]:
    """Context window a vLLM server reports for a model, and 20% of it as the output cap."""
    if client is None:
        return None, None
    try:
        listing = client.get("/api/cloud-access/vllm/models")
    except CliError:
        return None, None
    models = listing.get("models") if isinstance(listing, dict) else None
    for entry in models or []:
        if isinstance(entry, dict) and entry.get("id") == model:
            context = entry.get("context_length")
            if not isinstance(context, (int, float)) or context <= 0:
                return None, None  # the server did not report a window: never guess one
            return int(context), max(1, int(context * 0.2))
    raise CliError(
        f"No vLLM server set up in MOTO serves a model named '{model}'. "
        f"Run `{PRIMARY_COMMAND} vllm status` to see the detected models.",
        EXIT_USAGE,
    )


def resolve_role_spec(args: argparse.Namespace, client: Optional["MotoClient"]) -> RoleSpec:
    provider = PROVIDER_ALIASES.get(str(args.provider or "openrouter").strip().lower())
    if provider is None:
        raise CliError(f"Unknown provider '{args.provider}'. Use one of: {', '.join(sorted(set(PROVIDER_ALIASES.values())))}.", EXIT_USAGE)
    context, max_tokens = args.context, args.max_tokens
    if (context is None or max_tokens is None) and provider == "openrouter":
        auto_context, auto_output = _openrouter_limits(client, args.model)
        context = context if context is not None else auto_context
        max_tokens = max_tokens if max_tokens is not None else auto_output
    if (context is None or max_tokens is None) and provider == "vllm":
        auto_context, auto_output = _vllm_limits(client, args.model)
        context = context if context is not None else auto_context
        max_tokens = max_tokens if max_tokens is not None else auto_output
    if context is None or max_tokens is None:
        raise CliError(
            "Pass --context and --max-tokens for this model (MOTO only fills them in automatically for "
            "OpenRouter and vLLM models it can look up while MOTO is running).",
            EXIT_USAGE,
        )
    if context <= 0 or max_tokens <= 0 or max_tokens >= context:
        raise CliError("--context and --max-tokens must be positive, and max tokens smaller than the context.", EXIT_USAGE)
    return RoleSpec(provider, args.model, int(context), int(max_tokens), args.host, args.reasoning)


def _flat_role(prefix: str, spec: RoleSpec, context_key: str, output_key: str) -> dict:
    role = {
        f"{prefix}_provider": spec.provider,
        f"{prefix}_model": spec.model,
        f"{prefix}_{context_key}": spec.context,
        f"{prefix}_{output_key}": spec.max_tokens,
    }
    if spec.host:
        role[f"{prefix}_openrouter_provider"] = spec.host
    if spec.reasoning:
        role[f"{prefix}_openrouter_reasoning_effort"] = spec.reasoning
    return role


def _nested_role(spec: RoleSpec) -> dict:
    role = {"provider": spec.provider, "model_id": spec.model, "context_window": spec.context, "max_output_tokens": spec.max_tokens}
    if spec.host:
        role["openrouter_provider"] = spec.host
    if spec.reasoning:
        role["openrouter_reasoning_effort"] = spec.reasoning
    return role


def fresh_start_request(mode: str, spec: RoleSpec, submitters: int = 3) -> dict:
    """A complete start request that uses one model for every role (edit roles with --set)."""
    if not 1 <= submitters <= 10:
        raise CliError("--submitters must be between 1 and 10.", EXIT_USAGE)
    submitter_configs = [{"submitter_id": index, **_nested_role(spec)} for index in range(1, submitters + 1)]
    if mode == "autonomous":
        body: dict = {"user_research_prompt": "", "submitter_configs": submitter_configs,
                      "allow_mathematical_proofs": False, "allow_research_papers": True, "tier3_enabled": False}
        for prefix in ("validator", "writer", "high_param", "critique_submitter", "assistant"):
            body.update(_flat_role(prefix, spec, "context_window", "max_tokens"))
        return body
    if mode == "aggregator":
        body = {"user_prompt": "", "submitter_configs": submitter_configs, "uploaded_files": []}
        for prefix in ("validator", "assistant"):
            body.update(_flat_role(prefix, spec, "context_size", "max_output_tokens"))
        return body
    if mode == "compiler":
        body = {"compiler_prompt": "", "allow_mathematical_proofs": False, "allow_research_papers": True}
        for prefix in ("validator", "writer", "high_param", "assistant"):
            body.update(_flat_role(prefix, spec, "context_size", "max_output_tokens"))
        body.update(_flat_role("critique_submitter", spec, "context_window", "max_tokens"))
        return body
    role = _nested_role(spec)
    return {
        "user_prompt": "", "lean_template": "",
        "topic_generator": dict(role), "topic_validator": dict(role),
        "brainstorm_submitters": [dict(role) for _ in range(submitters)],
        "brainstorm_validator": dict(role), "path_decider": dict(role),
        "final_solver": dict(role), "assistant": dict(role),
    }


def build_start_request(args: argparse.Namespace, mode: WorkflowMode, instance: Instance, client: Optional["MotoClient"] = None,
                        template_mode: Optional[str] = None) -> dict:
    if args.config:
        body = _load_config_file(args.config)
    elif getattr(args, "model", None):
        body = fresh_start_request(mode.name, resolve_role_spec(args, client), args.submitters)
    else:
        body = (template_mode and load_start_template(instance, template_mode)) or load_start_template(instance, mode.name) or {}
        if not body:
            raise CliError(
                f"No saved {mode.label} settings yet. Choose a model with --model (for example "
                f"`{PRIMARY_COMMAND} {mode.name} start --model openai/gpt-5 --prompt ...`), start {mode.label} "
                f"once from the MOTO window, or pass --config FILE (see `{PRIMARY_COMMAND} config init {mode.name}`).",
                EXIT_USAGE,
            )
    body = copy.deepcopy(body)
    prompt = args.prompt
    if args.prompt_file:
        prompt = _read_text_arg(args.prompt_file)
    if prompt is not None:
        body[mode.prompt_field] = prompt
    elif str(body.get(mode.prompt_field) or "").strip():
        print("moto: no --prompt given; reusing the saved prompt.", file=sys.stderr)
    if not str(body.get(mode.prompt_field) or "").strip():
        raise CliError("A research prompt is required: pass --prompt or --prompt-file.", EXIT_USAGE)
    if mode.allowed_outputs:
        if args.proofs is not None:
            body["allow_mathematical_proofs"] = args.proofs
        if args.papers is not None:
            body["allow_research_papers"] = args.papers
    if mode.name == "leanoj" and getattr(args, "lean_template", None):
        body["lean_template"] = _read_text_arg(args.lean_template)
    if mode.name == "autonomous":
        if getattr(args, "final_answer", None) is not None:
            body["tier3_enabled"] = args.final_answer
        if getattr(args, "max_runtime", None):
            body["max_runtime_seconds"] = parse_duration(args.max_runtime)
    body = apply_set_overrides(body, args.set)
    if mode.name == "leanoj" and not str(body.get("lean_template") or "").strip():
        raise CliError("The LeanOJ Proof Solver needs a Lean template: pass --lean-template FILE.", EXIT_USAGE)
    return body


def _upload_files(client: "MotoClient", paths: Iterable[str]) -> list[str]:
    """Upload text files for the Aggregator; returns the stored logical filenames."""
    names: list[str] = []
    for value in paths:
        path = Path(value).expanduser()
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise CliError(f"Could not read {value}: {exc}", EXIT_USAGE)
        result = client.request("POST", "/api/aggregator/upload-file", files=[("file", path.name, content)])
        names.append(str(result.get("filename") if isinstance(result, dict) else path.name))
    return names


def cmd_workflow(args: argparse.Namespace) -> int:
    mode = MODES[args.mode]
    action = args.action
    if action == "clear" and not args.yes:
        raise CliError(f"Clearing {mode.label} progress cannot be undone; re-run with --yes to confirm.", EXIT_USAGE)
    if action == "start" and args.dry_run:
        instance = resolve_instance_or_offline(args)
        lookup = MotoClient(instance, timeout=30) if instance.backend_url else None
        emit(args, build_start_request(args, mode, instance, lookup))
        return EXIT_OK
    # Starting can wait on embedding/provider readiness checks before it returns.
    client = MotoClient(resolve_instance(args), timeout=300 if action == "start" else 60)
    if action == "start":
        body = build_start_request(args, mode, client.instance, client)
        if getattr(args, "file", None):
            uploaded = _upload_files(client, args.file)
            body["uploaded_files"] = list(dict.fromkeys([*(body.get("uploaded_files") or []), *uploaded]))
        result = client.post(f"{mode.prefix}/start", body)
        emit(args, result, lambda r: f"{mode.label} started. {_short(r.get('message') if isinstance(r, dict) else r)}")
    elif action == "stop":
        result = client.post(f"{mode.prefix}/stop")
        emit(args, result, lambda r: f"{mode.label} stop requested. {_short(r.get('message') if isinstance(r, dict) else '')}".strip())
    elif action == "status":
        result = client.get(f"{mode.prefix}/status")
        emit(args, result)
    elif action == "clear":
        path, query = mode.clear
        result = client.post(f"{mode.prefix}{path}", **query)
        emit(args, result, lambda r: f"{mode.label} progress cleared.")
    else:
        handler = EXTRA_ACTIONS[(mode.name, action)]
        return handler(args, client, mode)
    return EXIT_OK


def _extra_get(path: str, raw_text_key: Optional[str] = None) -> Callable:
    def handler(args: argparse.Namespace, client: MotoClient, mode: WorkflowMode) -> int:
        target = path.format(id=urllib.parse.quote(str(getattr(args, "item_id", "")), safe=""))
        result = client.get(f"{mode.prefix}{target}")
        if raw_text_key and not args.json and isinstance(result, dict) and isinstance(result.get(raw_text_key), str):
            print(result[raw_text_key])
        else:
            emit(args, result)
        return EXIT_OK

    return handler


def _extra_post(path: str, message: str, query: Optional[tuple[str, str]] = None) -> Callable:
    """POST helper; ``query`` maps a backend query parameter to an argparse attribute."""

    def handler(args: argparse.Namespace, client: MotoClient, mode: WorkflowMode) -> int:
        params = {query[0]: getattr(args, query[1])} if query else {}
        result = client.post(f"{mode.prefix}{path}", **params)
        emit(args, result, lambda r: f"{message} {_short(r.get('message') if isinstance(r, dict) else '')}".strip())
        return EXIT_OK

    return handler


def _aggregator_upload(args: argparse.Namespace, client: MotoClient, mode: WorkflowMode) -> int:
    names = _upload_files(client, args.files)
    emit(args, {"uploaded": names}, lambda r: "Uploaded " + ", ".join(r["uploaded"]) + ". Add them to a run with "
         f"`{PRIMARY_COMMAND} aggregator start --file ...` or --set uploaded_files=[...].")
    return EXIT_OK


# (mode, action) -> (help, handler, takes_item_id, extra_args)
EXTRA_ACTION_SPECS: dict[tuple[str, str], tuple[str, Callable, bool, list]] = {
    ("autonomous", "brainstorms"): ("List brainstorm topics.", _extra_get("/brainstorms"), False, []),
    ("autonomous", "brainstorm"): ("Show one brainstorm database.", _extra_get("/brainstorm/{id}", "content"), True, []),
    ("autonomous", "papers"): ("List completed papers.", _extra_get("/papers"), False, []),
    ("autonomous", "paper"): ("Show one paper.", _extra_get("/paper/{id}", "content"), True, []),
    ("autonomous", "progress"): ("Show the paper being written now.", _extra_get("/current-paper-progress"), False, []),
    ("autonomous", "final-answer"): ("Show the Tier 3 final answer.", _extra_get("/tier3/final-answer", "content"), False, []),
    ("autonomous", "sessions"): ("List research sessions.", _extra_get("/sessions"), False, []),
    ("autonomous", "force-paper-writing"): (
        "End the current brainstorm and write its paper now.",
        _extra_post("/force-paper-writing", "Paper writing forced."), False, [],
    ),
    ("autonomous", "force-tier3"): (
        "Move to Tier 3 final answer generation.",
        _extra_post("/force-tier3", "Tier 3 requested.", ("mode", "tier3_mode")), False,
        [(("--tier3-mode",), {"dest": "tier3_mode", "choices": ["complete_current", "skip_incomplete"],
                              "default": "complete_current", "help": "Finish current work first, or skip it."})],
    ),
    ("aggregator", "results"): ("Show accepted submissions.", _extra_get("/results", "results"), False, []),
    ("aggregator", "events"): ("Show the saved event log.", _extra_get("/events"), False, []),
    ("aggregator", "upload"): (
        "Upload .txt or .lean reference files (UTF-8, up to 5 MB each).", _aggregator_upload, False,
        [(("files",), {"nargs": "+", "metavar": "FILE", "help": "Files to upload."})],
    ),
    ("compiler", "paper"): ("Show the paper being written.", _extra_get("/paper", "paper"), False, []),
    ("compiler", "outline"): ("Show the paper outline.", _extra_get("/outline", "outline"), False, []),
    ("leanoj", "proofs"): ("List proofs from the current run.", _extra_get("/proofs"), False, []),
    ("leanoj", "master-proof"): ("Show the master proof draft.", _extra_get("/master-proof", "content"), False, []),
    ("leanoj", "skip-brainstorm"): ("Go straight to final proof solving.", _extra_post("/skip-brainstorm", "Skipping brainstorm."), False, []),
    ("leanoj", "force-brainstorm"): ("Return to brainstorming.", _extra_post("/force-brainstorm", "Returning to brainstorm."), False, []),
}
EXTRA_ACTIONS = {key: spec[1] for key, spec in EXTRA_ACTION_SPECS.items()}


# ---------------------------------------------------------------------------
# General commands
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    try:
        instance = resolve_instance(args)
    except CliError as exc:
        if exc.code != EXIT_NOT_RUNNING:
            raise
        emit(args, {"running": False, "message": str(exc)}, lambda p: f"MOTO is not running.\n{p['message']}")
        return EXIT_NOT_RUNNING
    if instance.data_root is None and probe_health(instance.backend_url) is None:  # explicit --url
        unreachable = {"running": False, "message": f"Nothing answered at {instance.backend_url}."}
        emit(args, unreachable, lambda p: f"MOTO is not running.\n{p['message']}")
        return EXIT_NOT_RUNNING
    client = MotoClient(instance, timeout=15)
    workflows, raw = workflow_statuses(client)
    active = [name for name, info in workflows.items() if info.get("running")]
    proof_runs = active_proof_runs(client)
    payload = {
        "running": True,
        "instance": instance.describe(),
        "active_workflow": active[0] if active else None,
        "workflows": workflows,
        "proof_runs": proof_runs,
        "attention": attention_items(client, raw, proof_runs),
    }

    def human(p: dict) -> str:
        inst = p["instance"]
        lines = [
            f"MOTO {inst.get('version') or ''} is running (instance '{inst['instance_id']}').".replace("  ", " "),
            f"  Backend:  {inst['backend_url']}",
        ]
        if inst.get("frontend_url"):
            lines.append(f"  App:      {inst['frontend_url']}")
        if p["active_workflow"]:
            lines.append(f"  Active workflow: {MODES[p['active_workflow']].label}")
        else:
            lines.append("  No workflow is running.")
        for run in p["proof_runs"]:
            lines.append(f"  Proof check {run['proof_run_id']}: {run['status']} ({run['source_type']} {run['source_id']}, round {run['current_round']})")
        for item in p["attention"]:
            lines.append(f"  Needs attention: {item['message']}")
            if item.get("hint"):
                lines.append(f"    {item['hint']}")
        return "\n".join(lines)

    emit(args, payload, human)
    return EXIT_OK


ACTIVE_PROOF_RUN_STATUSES = ("queued", "running", "provider_paused", "stopping")
TERMINAL_PROOF_RUN_STATUSES = ("completed", "stopped", "error")


def workflow_statuses(client: "MotoClient") -> tuple[dict[str, Any], dict[str, Any]]:
    """Summary and raw status for every workflow mode."""
    workflows: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    for name, mode in MODES.items():
        try:
            status = client.get(f"{mode.prefix}/status")
            raw[name] = status
            workflows[name] = {"label": mode.label, "running": _is_running(status)}
        except CliError as exc:
            if exc.code == EXIT_NOT_RUNNING:
                raise
            workflows[name] = {"label": mode.label, "running": False, "error": str(exc)}
    return workflows, raw


def active_proof_runs(client: "MotoClient") -> list[dict]:
    try:
        listing = client.get("/api/proofs/runs", limit=50)
    except CliError as exc:
        if exc.code == EXIT_NOT_RUNNING:
            raise
        return []
    runs = listing.get("runs") if isinstance(listing, dict) else None
    keep = ("proof_run_id", "scope", "source_type", "source_id", "source_title", "status", "run_mode",
            "current_round", "lifecycle_generation", "last_error_summary")
    return [
        {key: run.get(key) for key in keep}
        for run in runs or []
        if isinstance(run, dict) and run.get("status") in ACTIVE_PROOF_RUN_STATUSES
    ]


def attention_items(client: "MotoClient", raw_statuses: dict, proof_runs: list[dict]) -> list[dict]:
    """Problems the user or agent must act on: a stopped run's reason, blocked updates, paused checks."""
    items: list[dict] = []
    autonomous = raw_statuses.get("autonomous")
    terminal = autonomous.get("terminal_event") if isinstance(autonomous, dict) else None
    if isinstance(terminal, dict) and not _is_running(autonomous) and terminal.get("reason") not in QUIET_STOP_REASONS:
        keep = ("reason", "title", "message", "resolution", "terminal_guidance", "route_identity",
                "role_label", "recoverable", "fatal", "occurred_at", "error_detail")
        items.append({
            "kind": "autonomous_stopped",
            **{key: terminal.get(key) for key in keep if terminal.get(key) not in (None, "")},
            "message": f"Autonomous Research stopped: {_short(terminal.get('title') or terminal.get('message'), 300)}",
            "hint": _short(terminal.get("terminal_guidance") or terminal.get("resolution") or
                           "Fix the cause, then start Autonomous Research again to resume.", 400),
        })
    for run in proof_runs:
        if run.get("status") == "provider_paused":
            items.append({
                "kind": "proof_run_provider_paused",
                "proof_run_id": run.get("proof_run_id"),
                "message": f"Proof check {run.get('proof_run_id')} is paused because the provider ran out of credits.",
                "hint": f"Add credits, then run `{PRIMARY_COMMAND} retry openrouter`.",
            })
    try:
        path = client.get("/api/workflow/solution-path")
    except CliError as exc:
        if exc.code == EXIT_NOT_RUNNING:
            raise
        path = None
    if isinstance(path, dict) and int(path.get("repair_required_proposals") or 0) > 0:
        items.append({
            "kind": "solution_path_repair",
            "repairs": path.get("repairs") or [],
            "message": "A Progressive Solution Path update is blocked: "
                       + _short(path.get("repair_detail") or path.get("repair_reason") or "reviewer failure", 300),
            "hint": f"Fix the Main Submitter 1 settings if needed, then run `{PRIMARY_COMMAND} retry solution-path`.",
        })
    return items


def cmd_instances(args: argparse.Namespace) -> int:
    instances = [item.describe() for item in discover_instances(getattr(args, "data_root", None))]

    def human(items: list) -> str:
        if not items:
            return "No running MOTO instances."
        return "\n".join(f"{i['instance_id']}: {i['backend_url']} (data root {i['data_root']})" for i in items)

    emit(args, instances, human)
    return EXIT_OK if instances else EXIT_NOT_RUNNING


def normalize_route_path(path: str) -> str:
    """Accept ``/api/x`` or ``api/x``, and undo Git Bash rewriting ``/api/x`` into ``C:/.../api/x``."""
    candidate = path.replace("\\", "/").strip()
    if ":/" in candidate[:3] or candidate.startswith("//"):
        marker = candidate.find("/api/")
        if marker == -1 and candidate.endswith("/api"):
            marker = len(candidate) - 4
        if marker == -1:
            raise CliError(f"'{path}' is not a MOTO route; use a path such as api/health.", EXIT_USAGE)
        candidate = candidate[marker:]
    candidate = "/" + candidate.lstrip("/")
    if any(ord(char) < 33 for char in candidate):
        raise CliError(f"Route paths cannot contain spaces or control characters: {path!r}", EXIT_USAGE)
    return candidate


def cmd_api(args: argparse.Namespace) -> int:
    args.path = normalize_route_path(args.path)
    body = None
    if args.data is not None:
        raw = _read_text_arg(args.data[1:]) if args.data.startswith("@") else args.data
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise CliError(f"--data is not valid JSON: {exc}", EXIT_USAGE)
    query: list[tuple[str, str]] = []
    for item in args.query or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise CliError(f"--query expects KEY=VALUE, got '{item}'.", EXIT_USAGE)
        query.append((key, value))
    files: list[tuple[str, str, bytes]] = []
    for item in args.file or []:
        field_name, sep, value = item.partition("=")
        if not sep:
            field_name, value = "file", item
        path = Path(value).expanduser()
        try:
            files.append((field_name or "file", path.name, path.read_bytes()))
        except OSError as exc:
            raise CliError(f"Could not read {value}: {exc}", EXIT_USAGE)
    if files and body is not None:
        raise CliError("Use either --data (JSON) or --file (upload), not both.", EXIT_USAGE)
    client = MotoClient(resolve_instance(args), timeout=args.timeout)
    if args.output:
        payload, content_type = client.request(
            args.method, args.path, body=body, query=query or None, files=files or None, binary=True
        )
        destination = Path(args.output).expanduser()
        destination.write_bytes(payload)
        emit(args, {"saved": str(destination), "bytes": len(payload), "content_type": content_type},
             lambda r: f"Saved {r['bytes']} bytes to {r['saved']}.")
        return EXIT_OK
    result = client.request(args.method, args.path, body=body, query=query or None, files=files or None)
    emit(args, result)
    return EXIT_OK


HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")


def _resolve_schema(schema: Any, components: dict, depth: int = 0, seen: tuple = ()) -> Any:
    """Inline ``$ref`` schemas from /openapi.json so one route's shape reads on its own."""
    if isinstance(schema, list):
        return [_resolve_schema(item, components, depth, seen) for item in schema]
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        name = ref.rsplit("/", 1)[-1]
        if name in seen or depth >= 8:
            return {"$ref": name}
        target = components.get(name)
        return _resolve_schema(target, components, depth + 1, seen + (name,)) if isinstance(target, dict) else {"$ref": name}
    return {key: _resolve_schema(value, components, depth, seen) for key, value in schema.items()}


def cmd_routes(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=30)
    spec = client.get("/openapi.json")
    if not isinstance(spec, dict):
        raise CliError("MOTO did not return its API description (/openapi.json).")
    components = (spec.get("components") or {}).get("schemas") or {}
    needle = (args.filter or "").lower()
    rows: list[dict] = []
    for path, operations in sorted((spec.get("paths") or {}).items()):
        if not isinstance(operations, dict):
            continue
        for method, operation in operations.items():
            method = method.upper()
            if method not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            summary = str(operation.get("summary") or "")
            if args.method and method != args.method:
                continue
            if needle and needle not in f"{method} {path} {summary}".lower():
                continue
            row: dict = {"method": method, "path": path, "summary": summary}
            if args.schema:
                row["description"] = operation.get("description") or ""
                row["parameters"] = [
                    {"name": param.get("name"), "in": param.get("in"), "required": bool(param.get("required")),
                     "schema": _resolve_schema(param.get("schema"), components)}
                    for param in operation.get("parameters") or [] if isinstance(param, dict)
                ]
                content = ((operation.get("requestBody") or {}).get("content")) or {}
                if content:
                    content_type, media = next(iter(content.items()))
                    row["request_body"] = {
                        "content_type": content_type,
                        "required": bool((operation.get("requestBody") or {}).get("required")),
                        "schema": _resolve_schema((media or {}).get("schema"), components),
                    }
            rows.append(row)

    def human(items: list) -> str:
        if not items:
            return "No matching routes."
        if args.schema:
            return json.dumps(items, indent=2, default=str)
        return "\n".join(f"{row['method']:<6} {row['path']}  {row['summary']}".rstrip() for row in items)

    emit(args, rows, human)
    return EXIT_OK


def _load_websockets() -> tuple[Callable, type, type]:
    try:
        from websockets.exceptions import ConnectionClosed, WebSocketException
        from websockets.sync.client import connect
    except ImportError:
        raise CliError("Live activity needs the websockets package (installed with MOTO's requirements.txt).")
    return connect, ConnectionClosed, WebSocketException


def stream_events(
    args: argparse.Namespace,
    on_event: Callable[[dict], bool],
    timeout: float = 0,
    reconnect: bool = True,
) -> bool:
    """Feed live events to ``on_event`` until it returns True (-> True) or ``timeout`` passes (-> False).

    A dropped stream or a restarting backend is reconnected with backoff; each
    reconnect fetches a fresh ticket from whichever launch now owns the data root.
    Events published while disconnected are not replayed.
    """
    connect, connection_closed, websocket_error = _load_websockets()
    deadline = time.monotonic() + timeout if timeout else None
    backoff = 1.0
    connected_once = False
    while True:
        try:
            client = MotoClient(resolve_instance(args), timeout=15)
            ticket = client.post("/api/ws-ticket")
            if not isinstance(ticket, dict) or not ticket.get("ticket"):
                raise CliError("MOTO did not issue a WebSocket ticket.")
            ws_base = client.instance.backend_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
            url = f"{ws_base}/ws?ticket={urllib.parse.quote(ticket['ticket'])}"
            with connect(url, open_timeout=15, max_size=None) as socket:
                if connected_once:
                    print("moto: live activity stream reconnected (events sent while disconnected were missed).",
                          file=sys.stderr, flush=True)
                connected_once = True
                backoff = 1.0
                while True:
                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    if remaining == 0.0:
                        return False
                    try:
                        message = socket.recv(timeout=min(remaining, 30) if remaining is not None else 30)
                    except TimeoutError:
                        socket.send("ping")
                        continue
                    if message == "pong":
                        continue
                    try:
                        event = json.loads(message)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and on_event(event):
                        return True
        except CliError as exc:
            if exc.code != EXIT_NOT_RUNNING or not reconnect:
                raise
            reason = str(exc)
        except connection_closed as exc:
            if not reconnect:
                raise CliError(f"MOTO closed the live activity stream ({exc}).", EXIT_NOT_RUNNING)
            reason = f"stream closed ({exc})"
        except (OSError, websocket_error) as exc:
            if not reconnect:
                raise CliError(f"Could not stream live activity: {exc}", EXIT_NOT_RUNNING)
            reason = str(exc)
        if deadline is not None and time.monotonic() >= deadline:
            return False
        print(f"moto: live activity unavailable ({_short(reason, 200)}); retrying in {backoff:g}s.",
              file=sys.stderr, flush=True)
        delay = backoff if deadline is None else min(backoff, max(0.0, deadline - time.monotonic()))
        time.sleep(delay)
        backoff = min(backoff * 2, 30.0)


def _print_event(args: argparse.Namespace, event: dict) -> None:
    if args.json:
        print(json.dumps(event, default=str), flush=True)
        return
    data = event.get("data")
    text = ""
    if isinstance(data, dict):
        text = data.get("message") or data.get("summary") or data.get("status") or ""
    print(f"[{event.get('timestamp', '')}] {event.get('type')}: {_short(text, 300)}", flush=True)


def cmd_events(args: argparse.Namespace) -> int:
    wanted_types = set(args.type or [])
    count = 0

    def on_event(event: dict) -> bool:
        nonlocal count
        if wanted_types and event.get("type") not in wanted_types:
            return False
        _print_event(args, event)
        count += 1
        return bool(args.limit and count >= args.limit)

    try:
        stream_events(args, on_event, timeout=args.timeout, reconnect=not args.no_reconnect)
    except KeyboardInterrupt:
        return EXIT_OK
    return EXIT_OK


def resolve_instance_or_offline(args: argparse.Namespace) -> Instance:
    """The running instance, or (when MOTO is closed) its data root for reading saved settings."""
    try:
        return resolve_instance(args)
    except CliError as exc:
        if exc.code != EXIT_NOT_RUNNING:
            raise
    for root in candidate_data_roots(getattr(args, "data_root", None)):
        if (root / START_TEMPLATE_DIR_NAME).is_dir():
            return Instance("offline", "", "", data_root=root)
    return Instance("offline", "", "", data_root=candidate_data_roots(getattr(args, "data_root", None))[0])


def cmd_config(args: argparse.Namespace) -> int:
    mode = MODES[args.mode]
    if args.config_action == "init":
        instance = resolve_instance_or_offline(args)
        lookup = MotoClient(instance, timeout=30) if instance.backend_url else None
        body = apply_set_overrides(fresh_start_request(mode.name, resolve_role_spec(args, lookup), args.submitters), args.set)
        if args.output:
            destination = Path(args.output).expanduser()
            destination.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            emit(args, {"saved": str(destination), "request": body},
                 lambda r: f"Saved a {mode.label} start request to {r['saved']}. Edit it if you like, then run "
                           f"`{PRIMARY_COMMAND} {mode.name} start --config {r['saved']} --prompt ...`.")
        else:
            emit(args, body)
        return EXIT_OK
    template = load_start_template(resolve_instance_or_offline(args), mode.name)
    if template is None:
        raise CliError(
            f"No saved {mode.label} settings yet. Start {mode.label} once (from the MOTO window or with "
            f"`{PRIMARY_COMMAND} {mode.name} start --model ...`), or build settings with "
            f"`{PRIMARY_COMMAND} config init {mode.name} --model ...`.",
            EXIT_USAGE,
        )
    emit(args, template)
    return EXIT_OK


def cmd_open(args: argparse.Namespace) -> int:
    import webbrowser

    instance = resolve_instance(args)
    url = instance.frontend_url or "http://localhost:5173"
    webbrowser.open(url)
    emit(args, {"opened": url}, lambda p: f"Opened {p['opened']}")
    return EXIT_OK


def cmd_launch(args: argparse.Namespace) -> int:
    live = discover_instances(getattr(args, "data_root", None))
    if live and not args.json:
        print(f"MOTO is already running at {live[0].backend_url}.")
        return EXIT_OK
    if live:
        emit(args, {"already_running": True, "instance": live[0].describe()})
        return EXIT_OK
    launcher = REPO_ROOT / "moto_launcher.py"
    env = os.environ.copy()
    # Agents and scripts cannot answer update dialogs or "press Enter" prompts.
    if args.non_interactive or args.json or not sys.stdin.isatty():
        env["MOTO_LAUNCHER_NONINTERACTIVE"] = "1"
    if args.no_browser:
        env["MOTO_LAUNCHER_NO_BROWSER"] = "1"
    result = subprocess.run(
        [sys.executable, str(launcher)],
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL if env.get("MOTO_LAUNCHER_NONINTERACTIVE") else None,
        stdout=sys.stderr if args.json else None,
    )
    if result.returncode != 0:
        raise CliError("The MOTO launcher did not finish successfully.", result.returncode or EXIT_ERROR)
    live = discover_instances(getattr(args, "data_root", None))
    if args.json:
        emit(args, {"already_running": False, "instance": live[0].describe() if live else None})
    elif not live:
        # The launcher returns early after applying an update and relaunching itself.
        print("The launcher finished but MOTO is not answering yet (it may still be starting after an update). "
              f"Check with `{PRIMARY_COMMAND} status` in a minute.", file=sys.stderr)
    return EXIT_OK if live else EXIT_NOT_RUNNING


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_tree(pid: int) -> None:
    """Force-stop a process and its children (last resort)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


# Run in a detached helper: attach to the backend's own console and press Ctrl+C
# there, which lets uvicorn run MOTO's normal shutdown (checkpoints, locks, cleanup).
_CTRL_C_HELPER = (
    "import ctypes, sys\n"
    "k = ctypes.windll.kernel32\n"
    "k.FreeConsole()\n"
    "if not k.AttachConsole(int(sys.argv[1])): sys.exit(2)\n"
    "k.SetConsoleCtrlHandler(None, True)\n"
    "sys.exit(0 if k.GenerateConsoleCtrlEvent(0, 0) else 3)\n"
)


def _shares_our_console(pid: int) -> bool:
    """True when ``pid`` is attached to this terminal (a Ctrl+C there would also hit the user's shell)."""
    try:
        import ctypes

        buffer = (ctypes.c_ulong * 1024)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(buffer, 1024)
    except (AttributeError, OSError):
        return True
    return pid in list(buffer[: min(count, 1024)])


def request_graceful_exit(pid: int) -> bool:
    """Ask the backend to shut down normally. False when that cannot be requested safely."""
    if sys.platform != "win32":
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
        return True
    if _shares_our_console(pid):
        return False
    flags = getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CTRL_C_HELPER, str(pid)],
            capture_output=True, check=False, timeout=20, creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _wait_until(condition: Callable[[], bool], seconds: float, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if condition():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def stop_activity_for_shutdown(client: "MotoClient", wait_seconds: float) -> dict:
    """Stop the running workflow and manual proof checks, then wait for them to finish stopping."""
    workflows, _raw = workflow_statuses(client)
    stopped = [name for name, info in workflows.items() if info.get("running")]
    for name in stopped:
        client.post(f"{MODES[name].prefix}/stop")
    runs = active_proof_runs(client)
    for run in runs:
        try:
            client.post(f"/api/proofs/runs/{urllib.parse.quote(str(run['proof_run_id']), safe='')}/stop",
                        {"expected_lifecycle_generation": run.get("lifecycle_generation") or 1})
        except CliError as exc:
            if exc.code == EXIT_NOT_RUNNING:
                raise

    def idle() -> bool:
        current, _ = workflow_statuses(client)
        return not any(info.get("running") for info in current.values()) and not active_proof_runs(client)

    finished = _wait_until(idle, wait_seconds) if (stopped or runs) else True
    return {
        "stopped_workflows": stopped,
        "stopped_proof_runs": [run["proof_run_id"] for run in runs],
        "stopped_cleanly": finished,
    }


def cmd_shutdown(args: argparse.Namespace) -> int:
    instance = resolve_instance(args)
    if instance.pid is None:
        raise CliError("This instance did not record its process; close its MOTO windows instead.")
    # The backend PID comes from the control file of a backend that just accepted
    # its per-launch token, so it is this launch's process, not a reused PID.
    pid = instance.pid
    activity: dict = {"stopped_workflows": [], "stopped_proof_runs": [], "stopped_cleanly": None}
    graceful = False
    if not args.force:
        try:
            activity = stop_activity_for_shutdown(MotoClient(instance, timeout=30), args.wait)
        except CliError as exc:
            if exc.code != EXIT_NOT_RUNNING:
                raise
        if request_graceful_exit(pid):
            graceful = _wait_until(lambda: not _pid_alive(pid), 60)
    forced = False
    if _pid_alive(pid):
        _terminate_tree(pid)
        forced = True
        _wait_until(lambda: not _pid_alive(pid), 15, 0.5)
    stopped: list[int] = [pid]
    frontend_pid = launcher_frontend_pid(instance)
    if frontend_pid is not None and _pid_alive(frontend_pid):
        _terminate_tree(frontend_pid)
        stopped.append(frontend_pid)
    if instance.data_root is not None:
        control = read_control_file(instance.data_root)
        if control is not None and control.pid == pid:
            try:
                (instance.data_root / CONTROL_FILE_NAME).unlink()
            except OSError:
                pass
    payload = {
        "stopped_pids": stopped,
        "instance_id": instance.instance_id,
        "frontend_stopped": frontend_pid is not None,
        "graceful": graceful,
        "forced": forced,
        **activity,
    }

    def human(p: dict) -> str:
        text = f"MOTO instance '{p['instance_id']}' was shut down"
        text += " cleanly." if p["graceful"] else " (it had to be force-closed)."
        if p["stopped_workflows"]:
            text += " Stopped: " + ", ".join(MODES[name].label for name in p["stopped_workflows"]) + " (resumes on next start)."
        if p["stopped_cleanly"] is False:
            text += " A workflow was still stopping when MOTO closed."
        if not p["frontend_stopped"]:
            text += " Close the MOTO frontend window yourself if it is still open."
        return text

    emit(args, payload, human)
    return EXIT_OK


def launcher_frontend_pid(instance: Instance) -> Optional[int]:
    """The frontend window PID the launcher recorded for exactly this backend launch.

    Launcher state can outlive a launch and its PIDs can be reused by unrelated
    programs, so a record counts only when its backend PID is this backend's.
    """
    state = _read_json_file(LAUNCHER_STATE_PATH)
    if not isinstance(state, dict):
        return None
    for record in state.get("instances") or []:
        if (
            isinstance(record, dict)
            and str(record.get("instance_id")) == instance.instance_id
            and record.get("backend_window_pid") == instance.pid
            and isinstance(record.get("frontend_window_pid"), int)
        ):
            return int(record["frontend_window_pid"])
    return None


# ---------------------------------------------------------------------------
# Waiting
# ---------------------------------------------------------------------------


def _paper_count(client: "MotoClient") -> int:
    papers = client.get("/api/auto-research/papers")
    items = papers.get("papers") if isinstance(papers, dict) else papers
    return len(items) if isinstance(items, list) else 0


def _has_final_answer(client: "MotoClient") -> bool:
    answer = client.get("/api/auto-research/tier3/final-answer")
    return isinstance(answer, dict) and bool(answer.get("has_final_answer"))


def cmd_wait(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=30)
    started = time.monotonic()
    if args.event:
        wanted = set(args.event)
        seen: dict = {}

        def on_event(event: dict) -> bool:
            if event.get("type") in wanted:
                seen.update(event)
                return True
            return False

        if not stream_events(args, on_event, timeout=args.timeout):
            raise CliError(f"Timed out after {args.timeout:g}s waiting for {', '.join(sorted(wanted))}.")
        emit(args, {"met": True, "condition": "event", "event": seen, "elapsed_seconds": round(time.monotonic() - started, 1)},
             lambda r: f"Got {r['event'].get('type')}.")
        return EXIT_OK

    workflows, _ = workflow_statuses(client)
    mode = args.mode or next((name for name, info in workflows.items() if info.get("running")), None)
    if args.papers is not None or args.final_answer:
        mode = "autonomous"
    if args.proof_run:
        condition = f"proof check {args.proof_run} to finish"
    elif args.papers is not None:
        condition = f"{args.papers} completed papers"
    elif args.final_answer:
        condition = "the Tier 3 final answer"
    elif mode:
        condition = f"{MODES[mode].label} to stop"
    else:
        emit(args, {"met": True, "condition": "stopped", "message": "No workflow is running."}, lambda r: r["message"])
        return EXIT_OK
    deadline = started + args.timeout if args.timeout else None
    snapshot: Any = None
    while True:
        if args.proof_run:
            snapshot = client.get(f"/api/proofs/runs/{urllib.parse.quote(args.proof_run, safe='')}")
            if isinstance(snapshot, dict) and snapshot.get("status") in TERMINAL_PROOF_RUN_STATUSES:
                break
        else:
            snapshot = client.get(f"{MODES[mode].prefix}/status")
            running = _is_running(snapshot)
            if args.papers is not None and _paper_count(client) >= args.papers:
                break
            if args.final_answer and _has_final_answer(client):
                break
            if not running:
                if args.papers is None and not args.final_answer:
                    break
                terminal = snapshot.get("terminal_event") if isinstance(snapshot, dict) else None
                raise CliError(
                    f"{MODES[mode].label} stopped before reaching {condition}."
                    + (f" Reason: {_short(terminal.get('message'), 300)}" if isinstance(terminal, dict) else ""),
                    EXIT_REJECTED,
                    {"terminal_event": terminal},
                )
        if deadline is not None and time.monotonic() >= deadline:
            raise CliError(f"Timed out after {args.timeout:g}s waiting for {condition}.")
        time.sleep(args.interval if deadline is None else max(0.0, min(args.interval, deadline - time.monotonic())))
    payload = {"met": True, "condition": condition, "elapsed_seconds": round(time.monotonic() - started, 1), "status": snapshot}
    emit(args, payload, lambda r: f"Done waiting for {r['condition']}.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Deep thinking for agents (`moto think`)
# ---------------------------------------------------------------------------
#
# An agent hands MOTO one hard question, keeps working, and checks back later.
# MOTO explores the question (parallel brainstorming checked by a validator,
# plus Lean proofs or papers when asked for), then makes one model call that
# answers the question from what it explored. That answer is written when the
# exploration finishes on its own, at the maximum runtime, or on `think finish`.

THINK_PREFIX = MODES["autonomous"].prefix
THINK_TEMPLATE = "think"
MAX_RUNTIME_REASON = "max_runtime_reached"
# Endings the user or MOTO chose; they are results, not problems to repair.
QUIET_STOP_REASONS = (MAX_RUNTIME_REASON, "direct_answer_complete")

THINK_NEXT_STEPS = {
    "thinking": f"MOTO is still exploring. Check back later, block with `{PRIMARY_COMMAND} think wait --timeout SECONDS`, "
                f"or run `{PRIMARY_COMMAND} think finish` to stop exploring and get its answer now.",
    "answering": f"MOTO has stopped exploring and is writing its answer. Run `{PRIMARY_COMMAND} think wait`, then "
                 f"`{PRIMARY_COMMAND} think answer`.",
    "answered": f"MOTO's answer is ready: `{PRIMARY_COMMAND} think answer`.",
    "time_limit_reached": f"The maximum runtime ended the run. `{PRIMARY_COMMAND} think resume` keeps exploring.",
    "needs_attention": f"The run stopped on a problem (see attention). Fix it, then `{PRIMARY_COMMAND} think finish` to "
                       f"answer from what was explored, or `{PRIMARY_COMMAND} think resume` to keep exploring.",
    "stopped": f"The run was stopped without an answer. `{PRIMARY_COMMAND} think finish` answers from what was explored; "
               f"`{PRIMARY_COMMAND} think resume` keeps exploring.",
    "idle": f"No question is in progress. Start one with `{PRIMARY_COMMAND} think start --prompt-file question.txt`.",
}


def _direct_answer_status(status: Any) -> dict:
    direct = status.get("direct_answer") if isinstance(status, dict) else None
    return direct if isinstance(direct, dict) else {}


def think_state(status: Any) -> str:
    """One word for where a question stands, from the Autonomous Research status."""
    if not isinstance(status, dict):
        return "idle"
    direct = _direct_answer_status(status)
    if _is_running(status):
        return "answering" if direct.get("state") == "writing" else "thinking"
    terminal = status.get("terminal_event")
    reason = terminal.get("reason") if isinstance(terminal, dict) else None
    if reason and reason not in QUIET_STOP_REASONS:
        return "needs_attention"
    if direct.get("state") == "ready":
        return "answered"
    if reason == MAX_RUNTIME_REASON:
        return "time_limit_reached"
    return "stopped" if status.get("resume_available") else "idle"


def _think_question(client: "MotoClient") -> str:
    try:
        result = client.get(f"{THINK_PREFIX}/prompt")
    except CliError as exc:
        if exc.code == EXIT_NOT_RUNNING:
            raise
        return ""
    return str(result.get("prompt") or "") if isinstance(result, dict) else ""


def think_snapshot(client: "MotoClient", status: Any = None) -> dict:
    """Where the current question stands: state, time limit, progress, and anything needing attention."""
    status = status if isinstance(status, dict) else client.get(f"{THINK_PREFIX}/status")
    status = status if isinstance(status, dict) else {}
    state = think_state(status)
    direct = _direct_answer_status(status)
    stats = status.get("stats") if isinstance(status.get("stats"), dict) else {}
    brainstorm = status.get("current_brainstorm") if isinstance(status.get("current_brainstorm"), dict) else None
    paper = status.get("current_paper") if isinstance(status.get("current_paper"), dict) else None
    terminal = status.get("terminal_event") if isinstance(status.get("terminal_event"), dict) else None
    running = state in ("thinking", "answering")
    next_step = THINK_NEXT_STEPS[state]
    if state == "answered" and status.get("resume_available"):
        next_step += f" `{PRIMARY_COMMAND} think resume` explores further and answers again."
    snapshot: dict = {
        "state": state,
        "running": running,
        "answer_ready": direct.get("state") == "ready",
        "answer_reason": str(direct.get("reason") or "").partition(":")[0] or None,
        "can_resume": bool(status.get("resume_available")) and not running,
        "session_id": status.get("run_id"),
        "question": _think_question(client) if state != "idle" else "",
        "phase": status.get("current_tier"),
        "runtime_limit": status.get("runtime_limit"),
        "progress": {
            "ideas_accepted": stats.get("total_submissions_accepted", 0),
            "brainstorms_completed": stats.get("total_brainstorms_completed", 0),
            "papers_completed": stats.get("total_papers_completed", 0),
            "current_brainstorm": {key: brainstorm.get(key) for key in ("topic_id", "topic_prompt", "acceptance_count")}
            if brainstorm else None,
            "paper_in_progress": {key: paper.get(key) for key in ("paper_id", "title", "word_count")} if paper else None,
        },
        "stop_reason": None,
        "attention": [],
        "next": next_step,
    }
    if not running and terminal:
        snapshot["stop_reason"] = {"reason": terminal.get("reason"), "message": _short(terminal.get("message"), 400)}
    if state in ("thinking", "answering", "needs_attention"):
        snapshot["attention"] = attention_items(client, {"autonomous": status}, [])
    return snapshot


def _think_human(snapshot: dict) -> str:
    lines = [f"State: {snapshot['state']}"]
    if snapshot.get("question"):
        lines.append(f"Question: {_short(snapshot['question'], 200)}")
    limit = snapshot.get("runtime_limit")
    if isinstance(limit, dict):
        lines.append(f"Time limit: {format_duration(limit.get('max_runtime_seconds'))} "
                     f"({format_duration(limit.get('remaining_seconds'))} left)")
    progress = snapshot.get("progress") or {}
    lines.append(f"Progress: {progress.get('ideas_accepted', 0)} accepted ideas, {progress.get('papers_completed', 0)} papers"
                 + (f", phase {snapshot['phase']}" if snapshot.get("phase") else ""))
    if snapshot.get("stop_reason"):
        lines.append(f"Stopped: {snapshot['stop_reason'].get('message') or snapshot['stop_reason'].get('reason')}")
    for item in snapshot.get("attention") or []:
        lines.append(f"Needs attention: {item.get('message')}")
        if item.get("hint"):
            lines.append(f"  {item['hint']}")
    lines.append(snapshot["next"])
    return "\n".join(lines)


def _think_reject_if_busy(client: "MotoClient") -> dict:
    """Refuse to start while any workflow runs; returns the raw Autonomous Research status."""
    workflows, raw = workflow_statuses(client)
    active = next((name for name, info in workflows.items() if info.get("running")), None)
    if active:
        hint = (f"MOTO is already thinking. Check it with `{PRIMARY_COMMAND} think status`, or end it with "
                f"`{PRIMARY_COMMAND} think finish`." if active == "autonomous"
                else f"Stop it first with `{PRIMARY_COMMAND} {active} stop`, or wait for it to finish.")
        raise CliError(f"{MODES[active].label} is already running. {hint}", EXIT_REJECTED,
                       {"reason": "workflow_running", "active_workflow": active})
    status = raw.get("autonomous")
    return status if isinstance(status, dict) else {}


def think_start_request(args: argparse.Namespace, instance: Instance, client: Optional["MotoClient"],
                        resume: bool = False) -> dict:
    """An Autonomous Research start request that ends with a direct answer instead of a paper."""
    body = build_start_request(args, MODES["autonomous"], instance, client,
                               template_mode=THINK_TEMPLATE if resume else None)
    body["direct_answer_enabled"] = True
    body["tier3_enabled"] = False
    if not resume:  # a new question explores only; papers and proofs are opt-in extras
        body["allow_research_papers"] = bool(args.papers)
        body["allow_mathematical_proofs"] = bool(args.proofs)
    return body


def _think_started(args: argparse.Namespace, client: "MotoClient", body: dict, result: Any, resumed: bool) -> int:
    payload = {
        "started": True,
        "resumed": resumed,
        "max_runtime_seconds": body.get("max_runtime_seconds"),
        "message": _short(result.get("message") if isinstance(result, dict) else result),
        **think_snapshot(client),
    }
    emit(args, payload, lambda p: ("MOTO is continuing this question.\n" if p["resumed"] else "MOTO is thinking about your question.\n")
         + _think_human(p))
    return EXIT_OK


def _think_start(args: argparse.Namespace, client: "MotoClient") -> int:
    mode = MODES["autonomous"]
    status = _think_reject_if_busy(client)
    body = think_start_request(args, client.instance, client)
    resumed = False
    if status.get("resume_available"):
        paused = _think_question(client)
        if paused.strip() == str(body[mode.prompt_field]).strip():
            resumed = True  # the same question: MOTO continues its saved progress
        elif not args.replace:
            raise CliError(
                "A paused Autonomous Research session for a different question exists. Starting a new question "
                "retires it: its results stay in history, but it can no longer be resumed. Re-run with "
                f"--replace to do that, or continue the paused one with `{PRIMARY_COMMAND} think resume`.",
                EXIT_REJECTED, {"reason": "paused_session_exists", "paused_question": _short(paused, 600)},
            )
        else:
            path, query = mode.clear
            client.post(f"{mode.prefix}{path}", **query)
    result = client.post(f"{mode.prefix}/start", body)
    return _think_started(args, client, body, result, resumed)


def _think_resume_body(args: argparse.Namespace, client: "MotoClient") -> dict:
    status = _think_reject_if_busy(client)
    if not status.get("resume_available"):
        raise CliError(f"There is no paused question to continue. Start one with `{PRIMARY_COMMAND} think start`.",
                       EXIT_REJECTED, {"reason": "nothing_to_resume"})
    args.prompt = _think_question(client) or None  # MOTO keeps the paused session's own question either way
    return think_start_request(args, client.instance, client, resume=True)


def _think_resume(args: argparse.Namespace, client: "MotoClient") -> int:
    body = _think_resume_body(args, client)
    result = client.post(f"{THINK_PREFIX}/start", body)
    return _think_started(args, client, body, result, True)


def _think_finish(args: argparse.Namespace, client: "MotoClient") -> int:
    """Stop exploring and have MOTO answer now from what it has explored."""
    status = client.get(f"{THINK_PREFIX}/status")
    if _is_running(status):
        if not _direct_answer_status(status).get("enabled"):
            raise CliError(
                f"The running Autonomous Research was not started with `{PRIMARY_COMMAND} think`, so it has no direct "
                f"answer step. Stop it with `{PRIMARY_COMMAND} autonomous stop`, then run `{PRIMARY_COMMAND} think finish`.",
                EXIT_REJECTED, {"reason": "not_a_think_run"})
    else:
        # A stopped question is continued just long enough to hand it to its answer.
        client.post(f"{THINK_PREFIX}/start", _think_resume_body(args, client))
    result = client.post(f"{THINK_PREFIX}/direct-answer/finish")
    payload = {"finishing": True, "message": _short(result.get("message") if isinstance(result, dict) else result),
               **think_snapshot(client)}
    emit(args, payload, lambda p: "MOTO stopped exploring and is writing its answer.\n" + _think_human(p))
    return EXIT_OK


def _think_wait(args: argparse.Namespace, client: "MotoClient") -> int:
    started = time.monotonic()
    deadline = started + args.timeout if args.timeout else None
    while True:
        status = client.get(f"{THINK_PREFIX}/status")
        if not _is_running(status):
            timed_out = False
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(args.interval if deadline is None else max(0.0, min(args.interval, deadline - time.monotonic())))
    payload = {"timed_out": timed_out, "elapsed_seconds": round(time.monotonic() - started, 1), **think_snapshot(client, status)}
    emit(args, payload, _think_human)
    return EXIT_OK


ANSWER_REASON_NOTES = {
    "exploration_complete": "MOTO finished exploring on its own and wrote this answer.",
    "time_limit": "MOTO reached its maximum runtime and wrote this answer from what it had explored.",
    "finish_requested": "MOTO was asked to finish and wrote this answer from what it had explored.",
}


def _think_answer(args: argparse.Namespace, client: "MotoClient") -> int:
    snapshot = think_snapshot(client)
    direct = client.get(f"{THINK_PREFIX}/direct-answer")
    record = direct.get("answer") if isinstance(direct, dict) and direct.get("has_answer") else None
    record = record if isinstance(record, dict) else None
    payload: dict = {
        "state": snapshot["state"],
        "answer_ready": record is not None,
        "session_id": snapshot["session_id"],
        "question": (record or {}).get("question") or snapshot["question"],
        "answer": None,
        "papers": [],
        "proofs": [],
        "stop_reason": snapshot["stop_reason"],
        "note": "",
    }
    if record is not None:
        reason = str(record.get("reason") or "").partition(":")[0]
        payload["answer"] = {
            "content": record.get("answer"),
            "reason": reason,
            "exploration_complete": bool(record.get("exploration_complete")),
            "created_at": record.get("created_at"),
            "word_count": record.get("word_count"),
            "model": record.get("model"),
        }
        payload["note"] = ANSWER_REASON_NOTES.get(reason, "This is MOTO's answer.")
        if snapshot["running"]:
            payload["note"] += " MOTO is exploring further and will write a newer answer."
    elif snapshot["state"] == "idle":
        payload["note"] = THINK_NEXT_STEPS["idle"]
    elif snapshot["state"] == "answering":
        payload["note"] = "MOTO is writing its answer now. Check again shortly."
    else:
        payload["note"] = (f"No answer has been written yet. `{PRIMARY_COMMAND} think finish` stops exploring and has "
                           "MOTO answer from what it has explored so far.")
    if args.details and snapshot["state"] != "idle":
        papers = client.get(f"{THINK_PREFIX}/papers")
        for paper in (papers.get("papers") if isinstance(papers, dict) else None) or []:
            if not isinstance(paper, dict):
                continue
            entry = {key: paper.get(key) for key in ("paper_id", "title", "abstract", "word_count", "critique_avg")}
            if paper.get("paper_id"):
                paper_id = urllib.parse.quote(str(paper["paper_id"]), safe="")
                detail = client.get(f"{THINK_PREFIX}/paper/{paper_id}")
                entry["content"] = detail.get("content") if isinstance(detail, dict) else None
            payload["papers"].append(entry)
        proofs = client.get("/api/proofs", scope="autonomous")
        for proof in (proofs.get("proofs") if isinstance(proofs, dict) else None) or []:
            if isinstance(proof, dict):
                payload["proofs"].append({key: proof.get(key) for key in ("proof_id", "theorem_statement", "novelty_tier", "proof_kind")})

    def human(p: dict) -> str:
        lines = [str(p["answer"]["content"] or ""), "", f"({p['note']})"] if p["answer"] else [p["note"]]
        for paper in p["papers"]:
            lines.append(f"\nPaper {paper.get('paper_id')}: {paper.get('title')}\n{paper.get('content') or _short(paper.get('abstract'), 1200)}")
        for proof in p["proofs"]:
            lines.append(f"\nProof {proof.get('proof_id')} ({proof.get('novelty_tier') or 'unranked'}): "
                         f"{_short(proof.get('theorem_statement'), 400)}")
        return "\n".join(lines)

    emit(args, payload, human)
    return EXIT_OK


def cmd_think(args: argparse.Namespace) -> int:
    action = args.think_action
    if action == "start" and args.prompt is None and not args.prompt_file:
        raise CliError("Give MOTO the question to think about: pass --prompt or --prompt-file.", EXIT_USAGE)
    if action == "start" and args.dry_run:
        instance = resolve_instance_or_offline(args)
        lookup = MotoClient(instance, timeout=30) if instance.backend_url else None
        emit(args, think_start_request(args, instance, lookup))
        return EXIT_OK
    # Starting can wait on embedding/provider readiness checks before it returns.
    client = MotoClient(resolve_instance(args), timeout=300 if action in ("start", "resume", "finish") else 60)
    if action == "start":
        return _think_start(args, client)
    if action == "resume":
        return _think_resume(args, client)
    if action == "finish":
        return _think_finish(args, client)
    if action == "wait":
        return _think_wait(args, client)
    if action == "answer":
        return _think_answer(args, client)
    if action == "stop":
        client.post(f"{THINK_PREFIX}/stop")
    emit(args, think_snapshot(client), _think_human)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

# name -> (label, set route or None, clear route)
CREDENTIALS: dict[str, tuple[str, Optional[str], str]] = {
    "openrouter": ("OpenRouter API key", "/api/openrouter/set-api-key", "/api/openrouter/api-key"),
    "wolfram": ("Wolfram Alpha App ID", "/api/compiler/wolfram/set-api-key", "/api/compiler/wolfram/api-key"),
    "kimi-code": ("Kimi Code API key", "/api/cloud-access/kimi-code/api-key", "/api/cloud-access/kimi-code"),
    "sakana-fugu": ("Sakana Fugu API key", "/api/cloud-access/sakana-fugu/api-key", "/api/cloud-access/sakana-fugu"),
    "claude-api": ("Anthropic Claude API key", "/api/cloud-access/claude-api/api-key", "/api/cloud-access/claude-api"),
    "amazon-bedrock": (
        "Amazon Bedrock API key", "/api/cloud-access/amazon-bedrock/api-key", "/api/cloud-access/amazon-bedrock",
    ),
    "codex": ("OpenAI Codex sign-in", None, "/api/cloud-access/openai-codex"),
    "grok": ("xAI Grok sign-in", None, "/api/cloud-access/xai-grok"),
}
OAUTH_PROVIDERS = {"codex": ("OpenAI Codex", "/api/cloud-access/openai-codex"), "grok": ("xAI Grok", "/api/cloud-access/xai-grok")}


def read_secret(args: argparse.Namespace, label: str) -> str:
    """A secret from an environment variable, piped stdin, or a hidden terminal prompt - never argv."""
    if args.from_env:
        value = os.environ.get(args.from_env, "")
        if not value.strip():
            raise CliError(f"Environment variable {args.from_env} is empty or not set.", EXIT_USAGE)
        return value.strip()
    if not sys.stdin.isatty():
        value = sys.stdin.read()
    elif args.json:
        raise CliError("Pipe the key on stdin or use --from-env VAR (keys are never taken as arguments).", EXIT_USAGE)
    else:
        value = getpass.getpass(f"{label}: ")
    value = value.strip()
    if not value:
        raise CliError(f"No {label} was given.", EXIT_USAGE)
    return value


def cmd_keys(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=60)
    if args.keys_action == "status":
        emit(args, client.get("/api/connectivity/status"))
        return EXIT_OK
    label, set_route, clear_route = CREDENTIALS[args.provider]
    if args.keys_action == "set":
        if set_route is None:
            raise CliError(f"{label} uses a browser sign-in: run `{PRIMARY_COMMAND} login {args.provider}`.", EXIT_USAGE)
        body = {"api_key": read_secret(args, label)}
        if getattr(args, "region", None):
            body["region"] = args.region
        result = client.post(set_route, body)
        emit(args, result, lambda r: f"{label} saved. {_short(r.get('message') if isinstance(r, dict) else '')}".strip())
        return EXIT_OK
    if not args.yes:
        raise CliError(f"Removing the {label} cannot be undone; re-run with --yes to confirm.", EXIT_USAGE)
    result = client.delete(clear_route)
    emit(args, result, lambda r: f"{label} removed.")
    return EXIT_OK


VLLM_ROUTE = "/api/cloud-access/vllm"


def _describe_vllm(result: Any) -> str:
    """One-line summary of a vLLM test/save/availability result."""
    if not isinstance(result, dict):
        return ""
    chat = result.get("chat_models")
    names = [m.get("id") for m in chat if isinstance(m, dict)] if isinstance(chat, list) else result.get("models") or []
    embedding = result.get("embedding_model") or (result.get("status") or {}).get("embedding_model")
    parts = [f"chat models: {', '.join(str(n) for n in names) or 'none'}"]
    parts.append(f"embedding model: {embedding or 'none (RAG embeddings need LM Studio or OpenRouter)'}")
    return "; ".join(parts)


def cmd_vllm(args: argparse.Namespace) -> int:
    """Set up, check, or remove the user's vLLM servers."""
    client = MotoClient(resolve_instance(args), timeout=120)
    if args.vllm_action == "status":
        result = client.get(f"{VLLM_ROUTE}/availability", refresh="true")

        def human(r: Any) -> str:
            if not isinstance(r, dict) or not r.get("configured"):
                return f"vLLM is not set up. Run `{PRIMARY_COMMAND} vllm set http://127.0.0.1:8001`."
            state = "reachable" if r.get("available") else f"not answering ({_short(r.get('error') or '', 200)})"
            return f"vLLM servers {', '.join(r.get('servers') or [])}: {state}; {_describe_vllm(r)}"

        emit(args, result, human)
        return EXIT_OK
    if args.vllm_action == "clear":
        if not args.yes:
            raise CliError("Removing the vLLM setup cannot be undone; re-run with --yes to confirm.", EXIT_USAGE)
        emit(args, client.delete(VLLM_ROUTE), lambda r: "vLLM setup removed.")
        return EXIT_OK
    body: dict[str, Any] = {"servers": list(args.servers), "embedding_model": args.embedding_model or ""}
    if args.key_from_env:
        key = os.environ.get(args.key_from_env, "").strip()
        if not key:
            raise CliError(f"Environment variable {args.key_from_env} is empty or not set.", EXIT_USAGE)
        body["api_key"] = key
    if args.vllm_action == "test":
        result = client.post(f"{VLLM_ROUTE}/test", body)
        emit(args, result, lambda r: f"vLLM servers reachable (nothing saved); {_describe_vllm(r)}")
        return EXIT_OK
    body["clear_api_key"] = bool(args.clear_key)
    result = client.post(f"{VLLM_ROUTE}/config", body)
    emit(args, result, lambda r: f"vLLM setup saved; {_describe_vllm(r)}")
    return EXIT_OK


def _oauth_status(client: "MotoClient", base: str) -> dict:
    result = client.get(f"{base}/status")
    status = result.get("status") if isinstance(result, dict) else None
    return status if isinstance(status, dict) else {}


def cmd_login(args: argparse.Namespace) -> int:
    label, base = OAUTH_PROVIDERS[args.provider]
    client = MotoClient(resolve_instance(args), timeout=60)
    if args.callback_url:
        result = client.post(f"{base}/oauth/exchange", {"redirect_url": args.callback_url, "state": args.state or ""})
        emit(args, result, lambda r: f"Signed in to {label}.")
        return EXIT_OK

    def signed_in(after: int) -> bool:
        status = _oauth_status(client, base)
        return bool(status.get("configured")) and not status.get("expired") and int(status.get("updated_at") or 0) >= after

    if args.wait_only:
        if not _wait_until(lambda: signed_in(0), args.timeout, 3):
            raise CliError(f"Not signed in to {label} after {args.timeout:g}s.")
        emit(args, {"signed_in": True, "provider": args.provider}, lambda r: f"Signed in to {label}.")
        return EXIT_OK
    started_at = int(time.time()) - 1
    start = client.post(f"{base}/oauth/start", {})
    url = start.get("authorization_url") if isinstance(start, dict) else None
    if not url:
        raise CliError(f"MOTO did not return a {label} sign-in link.")
    info = {
        "authorization_url": url,
        "state": start.get("state"),
        "expires_in": start.get("expires_in"),
        "callback_available": bool(start.get("callback_available")),
    }

    def sign_in_link(r: dict) -> str:
        return f"Open this link to sign in to {label}:\n{r['authorization_url']}"

    if args.no_wait:
        emit(args, info, sign_in_link)
        return EXIT_OK
    if not args.json:
        emit(args, info, sign_in_link)
        sys.stdout.flush()
        if not args.no_browser:
            import webbrowser

            webbrowser.open(url)
    if not info["callback_available"]:
        raise CliError(
            "MOTO could not listen for the sign-in callback. After signing in, copy the address your browser "
            f"lands on and run `{PRIMARY_COMMAND} login {args.provider} --callback-url URL --state {info['state']}`.",
            EXIT_ERROR, info,
        )
    if not _wait_until(lambda: signed_in(started_at), args.timeout, 3):
        raise CliError(
            f"Not signed in to {label} after {args.timeout:g}s. If the browser showed an error page, copy its "
            f"address and run `{PRIMARY_COMMAND} login {args.provider} --callback-url URL --state {info['state']}`.",
            EXIT_ERROR, info,
        )
    emit(args, {"signed_in": True, "provider": args.provider}, lambda r: f"Signed in to {label}.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Proofs, settings, recovery, updates
# ---------------------------------------------------------------------------


def _on_off(value: Optional[str]) -> Optional[bool]:
    return None if value is None else value == "on"


def _proof_source(args: argparse.Namespace) -> tuple[str, str]:
    if args.target == "aggregator":
        return "brainstorm", "manual_aggregator"
    if args.target == "compiler":
        return "paper", "manual_compiler_current"
    if not args.source_id:
        raise CliError(f"`proofs check {args.target}` needs an ID (see `{PRIMARY_COMMAND} autonomous {args.target}s`).", EXIT_USAGE)
    return args.target, args.source_id


def cmd_proofs(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=120)
    action = args.proofs_action
    quote = lambda value: urllib.parse.quote(str(value), safe="")  # noqa: E731
    if action == "status":
        emit(args, client.get("/api/proofs/status"))
    elif action == "settings":
        current = client.get("/api/proofs/status")
        if not isinstance(current, dict):
            raise CliError("MOTO did not return its proof settings.")
        changes = {
            "lean4_lsp_enabled": _on_off(args.lsp),
            "max_parallel_candidates": args.parallel,
            "single_shot_proofs_enabled": _on_off(args.single_shot),
            "master_proofs_enabled": _on_off(args.master),
            "smt_enabled": _on_off(args.smt),
            "smt_timeout": args.smt_timeout,
        }
        if args.lean is None and args.timeout is None and all(value is None for value in changes.values()):
            emit(args, current)
            return EXIT_OK
        body = {
            "enabled": _on_off(args.lean) if args.lean is not None else bool(current.get("lean4_user_enabled")),
            "timeout": args.timeout if args.timeout is not None else int(current.get("lean4_proof_timeout") or 2000),
            **{key: value for key, value in changes.items() if value is not None},
        }
        emit(args, client.post("/api/proofs/settings", body), lambda r: "Proof settings saved.")
    elif action == "check":
        source_type, source_id = _proof_source(args)
        body = {"source_type": source_type, "source_id": source_id, "run_mode": "loop_with_pruning" if args.loop else "one_round"}
        result = client.post("/api/proofs/check", body)
        emit(args, result, lambda r: f"Proof check {r.get('proof_run_id')} queued. Follow it with "
             f"`{PRIMARY_COMMAND} proofs run {r.get('proof_run_id')}` or `{PRIMARY_COMMAND} wait --proof-run {r.get('proof_run_id')}`.")
    elif action == "runs":
        emit(args, client.get("/api/proofs/runs", limit=50, scope=args.scope))
    elif action == "run":
        path = f"/api/proofs/runs/{quote(args.proof_run_id)}"
        emit(args, client.get(f"{path}/activity") if args.activity else client.get(path))
    elif action == "stop":
        snapshot = client.get(f"/api/proofs/runs/{quote(args.proof_run_id)}")
        result = client.post(f"/api/proofs/runs/{quote(args.proof_run_id)}/stop",
                             {"expected_lifecycle_generation": int(snapshot.get("lifecycle_generation") or 1)})
        emit(args, result, lambda r: f"Proof check {args.proof_run_id} stop requested.")
    elif action == "list":
        emit(args, client.get("/api/proofs", scope=args.scope))
    elif action == "show":
        result = client.get(f"/api/proofs/{quote(args.proof_id)}", scope=args.scope)
        emit(args, result)
    elif action == "certificate":
        suffix = "certificate.lean" if args.lean else "certificate"
        if args.session:
            path = f"/api/proofs/library/{quote(args.session)}/{quote(args.proof_id)}/{suffix}"
            query = None
        else:
            path = f"/api/proofs/{quote(args.proof_id)}/{suffix}"
            query = {"scope": args.scope}
        if args.output:
            payload, content_type = client.request("GET", path, query=query, binary=True)
            Path(args.output).expanduser().write_bytes(payload)
            emit(args, {"saved": str(Path(args.output).expanduser()), "bytes": len(payload), "content_type": content_type},
                 lambda r: f"Saved {r['bytes']} bytes to {r['saved']}.")
        else:
            emit(args, client.request("GET", path, query=query, raw=args.lean) if args.lean else client.request("GET", path, query=query))
    elif action == "library":
        emit(args, client.get("/api/proofs/library"))
    elif action == "search":
        emit(args, client.post("/api/proof-search/search", {"query": args.query, "limit": args.limit}))
    return EXIT_OK


def cmd_settings(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=60)
    action = args.settings_action
    if action == "show":
        emit(args, {
            "connectivity": client.get("/api/connectivity/status"),
            "proofs": client.get("/api/proofs/status"),
            "free_models": client.get("/api/openrouter/free-model-settings"),
        })
    elif action == "free-models":
        current = client.get("/api/openrouter/free-model-settings")
        current = current if isinstance(current, dict) else {}
        current = current.get("settings") if isinstance(current.get("settings"), dict) else current
        body = {
            "looping_enabled": _on_off(args.looping) if args.looping is not None else bool(current.get("looping_enabled")),
            "auto_selector_enabled": _on_off(args.auto_selector) if args.auto_selector is not None
            else bool(current.get("auto_selector_enabled")),
        }
        emit(args, client.post("/api/openrouter/free-model-settings", body), lambda r: "Free-model settings saved.")
    elif action == "skills":
        body = {
            "syntheticlib4_enabled": _on_off(args.syntheticlib4),
            "agent_conversation_memory_enabled": _on_off(args.session_history),
            "wolfram_alpha_enabled": _on_off(args.wolfram),
        }
        body = {key: value for key, value in body.items() if value is not None}
        if not body:
            raise CliError("Choose at least one of --session-history, --wolfram, --syntheticlib4.", EXIT_USAGE)
        emit(args, client.post("/api/connectivity/toggles", body), lambda r: "Skill settings saved.")
    return EXIT_OK


def cmd_retry(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=60)
    if args.target == "openrouter":
        result = client.post("/api/openrouter/reset-exhaustion")
        emit(args, result, lambda r: "OpenRouter credit pauses were reset; paused work will retry.")
        return EXIT_OK
    path = client.get("/api/workflow/solution-path")
    if not isinstance(path, dict) or not path.get("run_id"):
        raise CliError("There is no solution path to retry.", EXIT_REJECTED)
    repairs = [item for item in path.get("repairs") or [] if isinstance(item, dict)]
    if args.proposal:
        repairs = [item for item in repairs if item.get("proposal_id") == args.proposal]
    if not repairs:
        raise CliError("No blocked solution-path update needs a retry.", EXIT_REJECTED)
    results = [
        client.post("/api/workflow/solution-path/resume", {
            "run_id": path["run_id"],
            "proposal_id": item["proposal_id"],
            "lifecycle_generation": int(path.get("lifecycle_generation") or item.get("lifecycle_generation") or 1),
        })
        for item in repairs
    ]
    emit(args, {"resumed": results}, lambda r: f"Retried {len(r['resumed'])} solution-path update(s).")
    return EXIT_OK


def cmd_alerts(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=30)
    result = client.get("/api/cloud-access/provider-notifications")
    notifications = result.get("notifications") if isinstance(result, dict) else []
    notifications = list(notifications or [])[-args.limit:] if args.limit else list(notifications or [])

    def human(items: list) -> str:
        if not items:
            return "No provider alerts."
        return "\n".join(
            f"[{item.get('timestamp') or item.get('created_at') or ''}] {_short(item.get('message') or item.get('title') or item, 300)}"
            for item in items
        )

    emit(args, notifications, human)
    return EXIT_OK


def cmd_update(args: argparse.Namespace) -> int:
    client = MotoClient(resolve_instance(args), timeout=60)
    if args.update_action == "check":
        notice = client.get("/api/update-notice")
        emit(args, notice, lambda r: "An update is available." if isinstance(r, dict) and r.get("update_available")
             else "MOTO is up to date.")
        return EXIT_OK
    if not args.yes:
        raise CliError("Updating changes MOTO's program files; re-run with --yes to confirm.", EXIT_USAGE)
    started = client.post("/api/update/pull")
    if not isinstance(started, dict) or not started.get("started"):
        raise CliError(_short(started.get("reason") if isinstance(started, dict) else started) or "The update did not start.",
                       EXIT_REJECTED, started)
    state: dict = {}

    def finished() -> bool:
        nonlocal state
        current = client.get("/api/update/pull-status")
        state = current if isinstance(current, dict) else {}
        return state.get("status") in ("done", "error")  # "idle" until the update task begins

    if not _wait_until(finished, args.timeout, 2):
        raise CliError(f"The update was still running after {args.timeout:g}s; check `{PRIMARY_COMMAND} api GET /api/update/pull-status`.")
    ok = state.get("status") == "done"
    payload = {**state, "restart_required": ok}
    if not ok:
        raise CliError("The update failed. " + _short("\n".join(state.get("output_lines") or [])[-600:]), EXIT_ERROR, payload)
    emit(args, payload, lambda r: f"Update applied. Restart MOTO to use it: `{PRIMARY_COMMAND} shutdown` then `{PRIMARY_COMMAND} launch`.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Command installation (PATH shims) and agent skill
# ---------------------------------------------------------------------------


def default_bin_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "MOTO" / "bin"
    return Path.home() / ".local" / "bin"


def preferred_python() -> str:
    venv = REPO_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return str(venv if venv.exists() else Path(sys.executable).resolve())


def shim_filename(command: str) -> str:
    return f"{command}.cmd" if sys.platform == "win32" else command


def render_shim(python: str) -> str:
    cli = str(REPO_ROOT / "moto_cli.py")
    if sys.platform == "win32":
        # Inside a batch file a literal % must be doubled.
        python, cli = python.replace("%", "%%"), cli.replace("%", "%%")
        return (
            "@echo off\r\n"
            f"rem {SHIM_MARKER} (remove with: moto uninstall-cli)\r\n"
            "setlocal\r\n"
            f'set "MOTO_PY={python}"\r\n'
            'if not exist "%MOTO_PY%" set "MOTO_PY=py"\r\n'
            f'"%MOTO_PY%" "{cli}" %*\r\n'
            "exit /b %ERRORLEVEL%\r\n"
        )
    return (
        "#!/bin/sh\n"
        f"# {SHIM_MARKER} (remove with: moto uninstall-cli)\n"
        f"MOTO_PY={shlex.quote(python)}\n"
        '[ -x "$MOTO_PY" ] || MOTO_PY=python3\n'
        f'exec "$MOTO_PY" {shlex.quote(cli)} "$@"\n'
    )


def is_moto_shim(path: Path) -> bool:
    try:
        return SHIM_MARKER in path.read_text(encoding="utf-8", errors="ignore")[:400]
    except OSError:
        return False


def find_command_conflict(command: str, bin_dir: Path, search_path: Optional[str] = None) -> Optional[str]:
    """Return another program that already answers to ``command``, if any."""
    path_value = search_path if search_path is not None else os.environ.get("PATH", "")
    entries = [entry for entry in path_value.split(os.pathsep) if entry]
    owned = os.path.normcase(str(bin_dir.resolve()))
    if sys.platform == "win32":
        extensions = [ext.lower() for ext in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if ext]
    else:
        extensions = [""]
    for entry in entries:
        try:
            directory = Path(entry).expanduser()
            if os.path.normcase(str(directory.resolve())) == owned:
                continue
        except OSError:
            continue
        for ext in extensions:
            candidate = directory / f"{command}{ext}"
            if candidate.is_file() and not is_moto_shim(candidate):
                if sys.platform != "win32" and not os.access(candidate, os.X_OK):
                    continue
                return str(candidate)
    return None


def _windows_user_path() -> tuple[str, int]:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ) as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
            return str(value), kind
        except FileNotFoundError:
            return "", winreg.REG_EXPAND_SZ


def path_with_entry(path_value: str, entry: str) -> Optional[str]:
    """Append ``entry`` (never prepend: existing commands keep priority). None when present."""
    parts = [part for part in path_value.split(os.pathsep) if part]
    target = os.path.normcase(os.path.normpath(os.path.expandvars(entry)))
    for part in parts:
        if os.path.normcase(os.path.normpath(os.path.expandvars(part))) == target:
            return None
    return os.pathsep.join(parts + [entry])


def path_without_entry(path_value: str, entry: str) -> Optional[str]:
    parts = [part for part in path_value.split(os.pathsep) if part]
    target = os.path.normcase(os.path.normpath(os.path.expandvars(entry)))
    kept = [part for part in parts if os.path.normcase(os.path.normpath(os.path.expandvars(part))) != target]
    return None if len(kept) == len(parts) else os.pathsep.join(kept)


def _set_windows_user_path(value: str, kind: int) -> None:
    import ctypes
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "Path", 0, kind, value)
    try:  # Tell Explorer so newly opened terminals see the change.
        result = ctypes.c_ulong()
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(result))
    except (AttributeError, OSError):
        pass


def install_cli(bin_dir: Optional[Path] = None, force: bool = False, update_path: bool = True) -> dict:
    """Install ``moto`` / ``moto-asi`` command shims. Never overwrites another program."""
    bin_dir = Path(bin_dir or default_bin_dir())
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = render_shim(preferred_python())
    installed: list[str] = []
    skipped: dict[str, str] = {}
    for command in (PRIMARY_COMMAND, ALIAS_COMMAND):
        target = bin_dir / shim_filename(command)
        if target.exists() and not is_moto_shim(target):
            skipped[command] = f"{target} belongs to another program"
            continue
        conflict = find_command_conflict(command, bin_dir)
        if conflict and command == PRIMARY_COMMAND and not force:
            skipped[command] = f"'{command}' already runs {conflict}"
            # A program installed after us now owns the name: withdraw our shim so
            # it cannot shadow that program where our folder comes first on PATH.
            if target.exists():
                target.unlink()
            continue
        target.write_text(shim, encoding="utf-8", newline="")
        if sys.platform != "win32":
            target.chmod(0o755)
        installed.append(command)
    path_updated = False
    on_path = path_with_entry(os.environ.get("PATH", ""), str(bin_dir)) is None
    if update_path and installed:
        if sys.platform == "win32":
            current, kind = _windows_user_path()
            updated = path_with_entry(current, str(bin_dir))
            if updated is not None:
                _set_windows_user_path(updated, kind)
                path_updated = True
            on_path = True
    return {
        "bin_dir": str(bin_dir),
        "installed": installed,
        "skipped": skipped,
        "path_updated": path_updated,
        "on_path": on_path or path_updated,
    }


def uninstall_cli(bin_dir: Optional[Path] = None, update_path: bool = True) -> dict:
    bin_dir = Path(bin_dir or default_bin_dir())
    removed: list[str] = []
    for command in (PRIMARY_COMMAND, ALIAS_COMMAND):
        target = bin_dir / shim_filename(command)
        if target.exists() and is_moto_shim(target):
            target.unlink()
            removed.append(command)
    path_updated = False
    if update_path and sys.platform == "win32" and bin_dir == default_bin_dir():
        remaining = [p for p in bin_dir.iterdir()] if bin_dir.is_dir() else []
        if not remaining:
            current, kind = _windows_user_path()
            updated = path_without_entry(current, str(bin_dir))
            if updated is not None:
                _set_windows_user_path(updated, kind)
                path_updated = True
    return {"bin_dir": str(bin_dir), "removed": removed, "path_updated": path_updated}


def cmd_install_cli(args: argparse.Namespace) -> int:
    result = install_cli(Path(args.bin_dir) if args.bin_dir else None, force=args.force, update_path=not args.no_path)

    def human(r: dict) -> str:
        lines = []
        if r["installed"]:
            lines.append(f"Installed {', '.join(r['installed'])} in {r['bin_dir']}.")
        for command, reason in r["skipped"].items():
            lines.append(f"Skipped '{command}': {reason}. Use '{ALIAS_COMMAND}' instead.")
        if r["path_updated"]:
            lines.append("Added that folder to your user PATH; open a new terminal to use it.")
        elif not r["on_path"]:
            lines.append(f"Add {r['bin_dir']} to your PATH to use it from any terminal.")
        return "\n".join(lines) or "Nothing to install."

    emit(args, result, human)
    return EXIT_OK if result["installed"] else EXIT_ERROR


def cmd_uninstall_cli(args: argparse.Namespace) -> int:
    result = uninstall_cli(Path(args.bin_dir) if args.bin_dir else None)
    emit(args, result, lambda r: f"Removed {', '.join(r['removed']) or 'nothing'} from {r['bin_dir']}.")
    return EXIT_OK


def skill_targets() -> dict[str, Path]:
    home = Path.home()
    return {"openclaw": home / ".openclaw" / "skills" / "moto", "claude": home / ".claude" / "skills" / "moto"}


def cmd_skill(args: argparse.Namespace) -> int:
    if not SKILL_SOURCE.is_file():
        raise CliError(f"The agent skill file is missing from this install ({SKILL_SOURCE.name}).")
    if args.skill_action == "show":
        print(SKILL_SOURCE.read_text(encoding="utf-8"))
        return EXIT_OK
    destinations: list[Path] = []
    for target in args.target or ["openclaw"]:
        destinations.append(skill_targets().get(target) or Path(target).expanduser() / "moto")
    written = []
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SKILL_SOURCE, destination / "SKILL.md")
        written.append(str(destination / "SKILL.md"))
    emit(args, {"installed": written}, lambda r: "Installed the MOTO agent skill:\n  " + "\n  ".join(r["installed"]))
    return EXIT_OK


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("connection and output")
    group.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Print machine-readable JSON.")
    group.add_argument("--instance", default=argparse.SUPPRESS, help="Instance ID when several MOTO instances run.")
    group.add_argument("--data-root", dest="data_root", default=argparse.SUPPRESS, help="Data root of the MOTO instance.")
    group.add_argument("--url", default=argparse.SUPPRESS, help="Backend URL (with --token or MOTO_DESKTOP_API_TOKEN).")
    group.add_argument("--token", default=argparse.SUPPRESS, help="Desktop API token for --url.")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog=PRIMARY_COMMAND,
        parents=[common],
        description="Control MOTO Autonomous ASI from a terminal or an AI agent.",
        epilog=(
            "Examples:\n"
            "  moto launch\n"
            "  moto keys set openrouter < key.txt\n"
            '  moto autonomous start --model openai/gpt-5 --prompt "Find a closed form for ..."\n'
            "  moto status --json\n"
            "  moto events --json --limit 20\n"
            "  moto wait --papers 1 --json\n"
            "  moto think start --prompt-file question.txt --max-runtime 2h --json\n"
            "  moto think wait --json\n"
            "  moto think answer --json\n"
            "  moto autonomous papers --json\n"
            "  moto routes proofs --schema\n"
            "  moto api PATCH /api/proofs/ID/live-context --data @body.json\n"
            "Exit codes: 0 ok, 1 error, 2 usage, 3 not running, 4 rejected by MOTO."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, func: Callable, help_text: str, aliases: Iterable[str] = ()) -> argparse.ArgumentParser:
        command = sub.add_parser(name, parents=[common], help=help_text, description=help_text, aliases=list(aliases))
        command.set_defaults(func=func)
        return command

    launch = add("launch", cmd_launch, "Start MOTO (runs the normal launcher) unless it is already running.")
    launch.add_argument("--no-browser", action="store_true", help="Do not open the app in a browser.")
    launch.add_argument(
        "--non-interactive", action="store_true",
        help="Never prompt (implied by --json or when input is not a terminal); updates are left for later.",
    )
    add("status", cmd_status, "Show whether MOTO runs, the active workflow, proof checks, and anything needing attention.")
    add("instances", cmd_instances, "List running MOTO instances.")
    add("open", cmd_open, "Open the MOTO app in the browser.")
    shutdown = add("shutdown", cmd_shutdown,
                   "Stop the running workflow (progress is kept), then close MOTO cleanly (backend and app windows).")
    shutdown.add_argument("--wait", type=float, default=120.0, help="Seconds to let workflows finish stopping (default 120).")
    shutdown.add_argument("--force", action="store_true", help="Close immediately without stopping workflows first.")

    def add_model_options(target: argparse.ArgumentParser, required: bool = False) -> None:
        group = target.add_argument_group("model settings (one model for every role; adjust single roles with --set)")
        group.add_argument("--model", required=required, help="Model ID, for example openai/gpt-5. Builds fresh settings.")
        group.add_argument("--provider", default="openrouter",
                           help="openrouter (default), lm_studio, vllm, codex, grok, kimi-code, sakana-fugu, claude-api, or amazon-bedrock.")
        group.add_argument("--host", help="OpenRouter host provider to pin (default: OpenRouter chooses).")
        group.add_argument("--reasoning", help="Reasoning effort: auto, xhigh, high, medium, low, minimal, none (max: Codex/Kimi/Claude/Bedrock).")
        group.add_argument("--context", type=int, help="Context window in tokens (filled in for OpenRouter and vLLM models).")
        group.add_argument("--max-tokens", dest="max_tokens", type=int, help="Max output tokens (filled in for OpenRouter and vLLM models).")
        group.add_argument("--submitters", type=int, default=3, help="Parallel brainstorm submitters, 1-10 (default 3).")

    def add_output_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--proofs", dest="proofs", action="store_true", default=None, help="Allow Lean proof output.")
        target.add_argument("--no-proofs", dest="proofs", action="store_false", help="Disallow proof output.")
        target.add_argument("--papers", dest="papers", action="store_true", default=None, help="Allow research papers.")
        target.add_argument("--no-papers", dest="papers", action="store_false", help="Disallow research papers.")

    for mode in MODES.values():
        mode_parser = add(mode.name, cmd_workflow, f"Control {mode.label}.", mode.aliases)
        mode_parser.set_defaults(mode=mode.name)
        actions = mode_parser.add_subparsers(dest="action", metavar="ACTION", required=True)
        start = actions.add_parser("start", parents=[common], help=f"Start {mode.label}.")
        start.add_argument("--prompt", help="Research prompt.")
        start.add_argument("--prompt-file", help="Read the prompt from a file ('-' for stdin).")
        start.add_argument("--config", help="Full start request JSON (default: settings last used in the app).")
        start.add_argument("--set", action="append", metavar="KEY=VALUE",
                           help="Override one field; dotted paths reach nested ones (submitter_configs.*.model_id=...).")
        start.add_argument("--dry-run", action="store_true", help="Print the request without starting.")
        add_model_options(start)
        if mode.name == "aggregator":
            start.add_argument("--file", action="append", metavar="FILE",
                               help="Upload a .txt or .lean reference file and use it in this run (repeatable).")
        if mode.allowed_outputs:
            add_output_options(start)
        if mode.name == "autonomous":
            start.add_argument("--final-answer", dest="final_answer", action="store_true", default=None,
                               help="Turn on the final answer stage (Tier 3): the run ends itself once it has an answer.")
            start.add_argument("--no-final-answer", dest="final_answer", action="store_false",
                               help="Turn the final answer stage off.")
            start.add_argument("--max-runtime", dest="max_runtime", metavar="DURATION", help=MAX_RUNTIME_HELP)
        if mode.name == "leanoj":
            start.add_argument("--lean-template", help="Lean template file ('-' for stdin).")
        actions.add_parser("stop", parents=[common], help=f"Stop {mode.label} (progress is kept).")
        actions.add_parser("status", parents=[common], help=f"Show {mode.label} status.")
        clear = actions.add_parser("clear", parents=[common], help=f"Erase {mode.label} progress (needs --yes).")
        clear.add_argument("--yes", action="store_true", help="Confirm the erase.")
        for (mode_name, action), (help_text, _handler, takes_id, extra) in EXTRA_ACTION_SPECS.items():
            if mode_name != mode.name:
                continue
            extra_parser = actions.add_parser(action, parents=[common], help=help_text)
            if takes_id:
                extra_parser.add_argument("item_id", help="Item ID (see the matching list command).")
            for flags, options in extra:
                extra_parser.add_argument(*flags, **options)

    think = add("think", cmd_think,
                "Hand MOTO one hard question to think about in depth, keep working, and check back for its answer.")
    think_actions = think.add_subparsers(dest="think_action", metavar="ACTION", required=True)
    think_start = think_actions.add_parser(
        "start", parents=[common],
        help="Start thinking about a question (uses your saved Autonomous Research model settings).")
    think_start.add_argument("--prompt", help="The question or decision to think about, with the context MOTO needs.")
    think_start.add_argument("--prompt-file", help="Read the question from a file ('-' for stdin).")
    think_start.add_argument("--max-runtime", dest="max_runtime", metavar="DURATION", help=THINK_MAX_RUNTIME_HELP)
    think_start.add_argument("--proofs", action="store_true",
                             help="Also search for Lean-verified proofs while exploring (needs Lean; much more usage).")
    think_start.add_argument("--papers", action="store_true",
                             help="Also write research papers from the exploration before answering (much longer).")
    think_start.add_argument("--replace", action="store_true",
                             help="Retire a paused session for a different question (it stays in history but cannot be resumed).")
    think_start.add_argument("--config", help="Full start request JSON (default: settings last used in the app).")
    think_start.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override one field (dotted paths allowed).")
    think_start.add_argument("--dry-run", action="store_true", help="Print the request without starting.")
    add_model_options(think_start)
    for name, help_text in (
        ("resume", "Keep exploring the paused question, optionally with a new time limit; it answers again at the end."),
        ("finish", "Stop exploring and have MOTO answer now from what it has explored."),
    ):
        continued = think_actions.add_parser(name, parents=[common], help=help_text)
        if name == "resume":
            continued.add_argument("--max-runtime", dest="max_runtime", metavar="DURATION", help=THINK_MAX_RUNTIME_HELP)
        continued.add_argument("--config", help="Full start request JSON (default: this question's saved settings).")
        continued.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override one field (dotted paths allowed).")
        continued.set_defaults(prompt=None, prompt_file=None, model=None, proofs=None, papers=None, dry_run=False,
                               max_runtime=None)
    think_actions.add_parser("status", parents=[common], help="Where the question stands: state, time left, progress, problems.")
    think_answer = think_actions.add_parser("answer", parents=[common], help="MOTO's answer to the question, once it has written one.")
    think_answer.add_argument("--details", action="store_true",
                              help="Also list the papers (full text) and verified proofs behind the answer.")
    think_wait = think_actions.add_parser("wait", parents=[common], help="Block until MOTO stops (answer written or a problem), then print the status.")
    think_wait.add_argument("--timeout", type=float, default=0,
                            help="Return after this many seconds even if MOTO is still thinking (timed_out: true).")
    think_wait.add_argument("--interval", type=float, default=15.0, help="Seconds between status checks (default 15).")
    think_actions.add_parser("stop", parents=[common], help="Stop at once without an answer (progress is kept; prefer finish).")

    events = add("events", cmd_events, "Stream live activity (one JSON object per line with --json).")
    events.add_argument("--type", action="append", help="Only events of this type (repeatable).")
    events.add_argument("--limit", type=int, default=0, help="Stop after this many events.")
    events.add_argument("--timeout", type=float, default=0, help="Stop after this many seconds.")
    events.add_argument("--no-reconnect", action="store_true", help="Exit (code 3) if the stream drops instead of reconnecting.")

    wait = add("wait", cmd_wait, "Wait until a workflow stops, papers or a final answer exist, a proof check ends, or an event arrives.")
    wait.add_argument("--mode", choices=list(MODES), help="Workflow to wait for (default: the running one).")
    wait.add_argument("--papers", type=int, help="Wait until Autonomous Research has this many completed papers.")
    wait.add_argument("--final-answer", action="store_true", help="Wait for the Tier 3 final answer.")
    wait.add_argument("--proof-run", help="Wait until this proof check finishes.")
    wait.add_argument("--event", action="append", metavar="TYPE", help="Wait for a live event of this type (repeatable).")
    wait.add_argument("--timeout", type=float, default=0, help="Give up after this many seconds (exit 1). Default: wait forever.")
    wait.add_argument("--interval", type=float, default=15.0, help="Seconds between status checks (default 15).")

    config = add("config", cmd_config, "Show saved start settings, or build new ones for a fresh install.")
    config_actions = config.add_subparsers(dest="config_action", metavar="ACTION", required=True)
    config_show = config_actions.add_parser("show", parents=[common], help="Print the settings last used to start a workflow.")
    config_show.add_argument("mode", choices=list(MODES), help="Workflow mode.")
    config_init = config_actions.add_parser("init", parents=[common], help="Build a start request that uses one model for every role.")
    config_init.add_argument("mode", choices=list(MODES), help="Workflow mode.")
    add_model_options(config_init, required=True)
    config_init.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override one field (dotted paths allowed).")
    config_init.add_argument("--output", "-o", help="Write the request to this file instead of printing it.")

    api = add("api", cmd_api, "Call any MOTO REST route directly (see `moto routes`).")
    api.add_argument("method", type=str.upper, choices=list(HTTP_METHODS))
    api.add_argument("path", help="Route path, for example api/proofs (a leading / is optional).")
    api.add_argument("--data", help="JSON body, or @file.")
    api.add_argument("--file", action="append", metavar="[FIELD=]PATH", help="Send a file upload (multipart; field defaults to 'file').")
    api.add_argument("--query", action="append", metavar="KEY=VALUE", help="Query parameter (repeatable).")
    api.add_argument("--output", "-o", help="Save the response body to this file (use for PDFs and downloads).")
    api.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")

    routes = add("routes", cmd_routes, "List MOTO's REST routes, optionally with their request shapes.")
    routes.add_argument("filter", nargs="?", help="Only routes whose method, path, or summary contains this text.")
    routes.add_argument("--method", type=str.upper, choices=list(HTTP_METHODS), help="Only this HTTP method.")
    routes.add_argument("--schema", action="store_true", help="Include parameters and the request body schema.")

    keys = add("keys", cmd_keys, "Save, remove, or check provider keys (keys are read from stdin, never arguments).")
    key_actions = keys.add_subparsers(dest="keys_action", metavar="ACTION", required=True)
    key_actions.add_parser("status", parents=[common], help="Show which providers and skills are connected.")
    key_set = key_actions.add_parser("set", parents=[common], help="Save a key (pipe it on stdin, or --from-env VAR).")
    key_set.add_argument("provider", choices=[name for name, spec in CREDENTIALS.items() if spec[1]])
    key_set.add_argument("--from-env", metavar="VAR", help="Read the key from this environment variable.")
    key_set.add_argument("--region", help="AWS region for an Amazon Bedrock key (for example us-east-1).")
    key_clear = key_actions.add_parser("clear", parents=[common], help="Remove a key or sign-in (needs --yes).")
    key_clear.add_argument("provider", choices=list(CREDENTIALS))
    key_clear.add_argument("--yes", action="store_true", help="Confirm the removal.")

    vllm = add("vllm", cmd_vllm, "Set up, check, or remove your vLLM servers (local chat models and RAG embeddings).")
    vllm_actions = vllm.add_subparsers(dest="vllm_action", metavar="ACTION", required=True)
    vllm_actions.add_parser("status", parents=[common], help="Check the saved vLLM servers and list detected models.")
    for action, text in (
        ("set", "Check the servers, then save them (models and context windows are detected)."),
        ("test", "Check the servers and list their models without saving."),
    ):
        vllm_set = vllm_actions.add_parser(action, parents=[common], help=text)
        vllm_set.add_argument("servers", nargs="+", metavar="URL", help="vLLM server address, for example http://127.0.0.1:8001.")
        vllm_set.add_argument("--embedding-model", help="Embedding model for RAG (default: the first one detected).")
        vllm_set.add_argument("--key-from-env", metavar="VAR",
                              help="Read the server's API key (--api-key) from this environment variable.")
        if action == "set":
            vllm_set.add_argument("--clear-key", action="store_true", help="Remove the saved vLLM API key.")
    vllm_clear = vllm_actions.add_parser("clear", parents=[common], help="Remove the vLLM setup (needs --yes).")
    vllm_clear.add_argument("--yes", action="store_true", help="Confirm the removal.")

    login = add("login", cmd_login, "Sign in to OpenAI Codex or xAI Grok with your subscription (browser sign-in).")
    login.add_argument("provider", choices=list(OAUTH_PROVIDERS))
    login.add_argument("--no-browser", action="store_true", help="Print the sign-in link without opening a browser.")
    login.add_argument("--no-wait", action="store_true", help="Print the link and state, then exit (finish with --wait-only).")
    login.add_argument("--wait-only", action="store_true", help="Only wait until the sign-in completes.")
    login.add_argument("--callback-url", help="Finish sign-in with the address the browser landed on.")
    login.add_argument("--state", help="The state value printed when sign-in started (with --callback-url).")
    login.add_argument("--timeout", type=float, default=600.0, help="Seconds to wait for sign-in (default 600).")

    proofs = add("proofs", cmd_proofs, "Lean 4 proof checks, proof records, certificates, search, and proof settings.")
    proof_actions = proofs.add_subparsers(dest="proofs_action", metavar="ACTION", required=True)
    proof_actions.add_parser("status", parents=[common], help="Lean/SMT availability and proof counts.")
    proof_settings = proof_actions.add_parser("settings", parents=[common], help="Show or change Lean 4 and SMT settings.")
    proof_settings.add_argument("--lean", choices=["on", "off"], help="Turn Lean 4 proof work on or off.")
    proof_settings.add_argument("--timeout", type=int, help="Lean check timeout in seconds.")
    proof_settings.add_argument("--lsp", choices=["on", "off"], help="Persistent Lean server (faster checks).")
    proof_settings.add_argument("--parallel", type=int, help="Concurrent proof attempts (0 = unlimited).")
    proof_settings.add_argument("--single-shot", dest="single_shot", choices=["on", "off"],
                                help="Single-shot proofs (one five-attempt budget per theorem).")
    proof_settings.add_argument("--master", choices=["on", "off"],
                                help="Master proofs (large theorems built as a scaffold of steps). Keep at least one type on.")
    proof_settings.add_argument("--smt", choices=["on", "off"], help="Z3/SMT tactic hints.")
    proof_settings.add_argument("--smt-timeout", dest="smt_timeout", type=int, help="SMT timeout in seconds.")
    proof_check = proof_actions.add_parser("check", parents=[common], help="Search a brainstorm or paper for new Lean proofs.")
    proof_check.add_argument("target", choices=["aggregator", "compiler", "brainstorm", "paper"],
                             help="aggregator/compiler: the manual workflow's results; brainstorm/paper: an Autonomous item.")
    proof_check.add_argument("source_id", nargs="?", help="Brainstorm topic ID or paper ID (history papers: SESSION:PAPER).")
    proof_check.add_argument("--loop", action="store_true",
                             help="Keep running rounds until 10 in a row find nothing (default: one round).")
    proof_runs = proof_actions.add_parser("runs", parents=[common], help="List recent proof checks.")
    proof_runs.add_argument("--scope", choices=["autonomous", "manual"])
    proof_run = proof_actions.add_parser("run", parents=[common], help="Show one proof check.")
    proof_run.add_argument("proof_run_id")
    proof_run.add_argument("--activity", action="store_true", help="Show its live activity log instead.")
    proof_stop = proof_actions.add_parser("stop", parents=[common], help="Stop a proof check.")
    proof_stop.add_argument("proof_run_id")
    proof_list = proof_actions.add_parser("list", parents=[common], help="List verified proofs from the current run.")
    proof_list.add_argument("--scope", choices=["autonomous", "manual"], default="autonomous")
    proof_show = proof_actions.add_parser("show", parents=[common], help="Show one proof with its Lean code.")
    proof_show.add_argument("proof_id")
    proof_show.add_argument("--scope", choices=["autonomous", "manual"], default="autonomous")
    proof_cert = proof_actions.add_parser("certificate", parents=[common], help="Export a proof certificate (JSON, or --lean source).")
    proof_cert.add_argument("proof_id")
    proof_cert.add_argument("--lean", action="store_true", help="The .lean source instead of the JSON certificate.")
    proof_cert.add_argument("--scope", choices=["autonomous", "manual"], default="autonomous")
    proof_cert.add_argument("--session", help="Export from the proof library (past runs) with this session ID.")
    proof_cert.add_argument("--output", "-o", help="Save to this file.")
    proof_actions.add_parser("library", parents=[common], help="List proofs from past runs.")
    proof_search = proof_actions.add_parser("search", parents=[common], help="Search verified proofs.")
    proof_search.add_argument("query")
    proof_search.add_argument("--limit", type=int, default=7, help="Results (1-7).")

    settings = add("settings", cmd_settings, "Show or change MOTO settings that are not part of a workflow start.")
    settings_actions = settings.add_subparsers(dest="settings_action", metavar="ACTION", required=True)
    settings_actions.add_parser("show", parents=[common], help="Connectivity, proof, and free-model settings.")
    free_models = settings_actions.add_parser("free-models", parents=[common], help="OpenRouter free-model fallbacks.")
    free_models.add_argument("--looping", choices=["on", "off"], help="Rotate to another free model when one is rate limited.")
    free_models.add_argument("--auto-selector", dest="auto_selector", choices=["on", "off"],
                             help="Fall back to openrouter/free when every free model is busy.")
    skills = settings_actions.add_parser("skills", parents=[common], help="Optional skills (stop the workflow first).")
    skills.add_argument("--session-history", dest="session_history", choices=["on", "off"], help="Session History Memory.")
    skills.add_argument("--wolfram", choices=["on", "off"], help="Wolfram Alpha checks while writing papers.")
    skills.add_argument("--syntheticlib4", choices=["on", "off"], help="SyntheticLib4 proof corpus.")

    retry = add("retry", cmd_retry, "Resume work that paused for a fixable reason.")
    retry.add_argument("target", choices=["openrouter", "solution-path"],
                       help="openrouter: after adding credits; solution-path: a blocked solution-path update.")
    retry.add_argument("--proposal", help="Only this solution-path proposal ID.")

    alerts = add("alerts", cmd_alerts, "Show recent provider alerts (failed sign-ins, credit and usage limits).")
    alerts.add_argument("--limit", type=int, default=20, help="Most recent alerts to show (0 = all).")

    update = add("update", cmd_update, "Check for or apply a MOTO update.")
    update_actions = update.add_subparsers(dest="update_action", metavar="ACTION", required=True)
    update_actions.add_parser("check", parents=[common], help="Is an update available?")
    update_apply = update_actions.add_parser("apply", parents=[common], help="Download and apply the update (needs --yes).")
    update_apply.add_argument("--yes", action="store_true", help="Confirm the update.")
    update_apply.add_argument("--timeout", type=float, default=900.0, help="Seconds to wait for it (default 900).")

    install = add("install-cli", cmd_install_cli, "Put the moto and moto-asi commands on your PATH.")
    install.add_argument("--bin-dir", help="Folder for the command files.")
    install.add_argument("--force", action="store_true", help="Install 'moto' even if another program uses that name.")
    install.add_argument("--no-path", action="store_true", help="Do not change PATH.")
    uninstall = add("uninstall-cli", cmd_uninstall_cli, "Remove the moto commands from your PATH.")
    uninstall.add_argument("--bin-dir", help="Folder holding the command files.")

    skill = add("skill", cmd_skill, "Show or install the MOTO skill for AI agents (OpenClaw, Claude Code).")
    skill.add_argument("skill_action", choices=["show", "install"], help="Print the skill or install it.")
    skill.add_argument("--target", action="append", help="openclaw (default), claude, or a skills folder path.")
    return parser


def _apply_common_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for name, default in (("json", False), ("instance", None), ("data_root", None), ("url", None), ("token", None)):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # A legacy console code page must never crash a command over one character.
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else EXIT_USAGE
    args = _apply_common_defaults(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    try:
        return args.func(args)
    except CliError as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "exit_code": exc.code, "detail": exc.detail}, indent=2, default=str))
        else:
            print(f"moto: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return EXIT_ERROR
    except Exception as exc:  # Never dump a traceback on an agent; MOTO_CLI_DEBUG=1 shows it.
        if os.environ.get("MOTO_CLI_DEBUG"):
            raise
        message = f"Unexpected {type(exc).__name__}: {exc}"
        if args.json:
            print(json.dumps({"error": message, "exit_code": EXIT_ERROR, "detail": None}, indent=2))
        else:
            print(f"moto: {message}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
