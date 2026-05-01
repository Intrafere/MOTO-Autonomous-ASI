"""
FastAPI main application.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from backend.api.middleware import setup_middleware
from backend.api.routes import (
    aggregator,
    websocket,
    compiler,
    autonomous,
    boost,
    workflow,
    openrouter,
    download,
    features,
    health,
    proofs,
)
from backend.shared.build_info import get_build_info
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.config import rag_config, system_config
from backend.shared.lean4_client import clear_lean4_client, close_lean4_client, initialize_lean4_client
from backend.aggregator.core.coordinator import coordinator
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator

# Setup logging with millisecond precision for log correlation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP client logs (keep only WARNING/ERROR level)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

build_info = get_build_info()

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag_enabled(name: str) -> bool:
    """Return True when an environment flag is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _apply_generic_mode_from_env() -> None:
    """Enable generic mode from the explicit hosted deployment env var."""
    if _env_flag_enabled("MOTO_GENERIC_MODE"):
        system_config.generic_mode = True


def _validate_generic_mode_startup_env() -> None:
    """Fail closed when hosted runtime auth inputs are missing."""
    if not system_config.generic_mode:
        return

    missing: list[str] = []

    if not any(os.environ.get(name, "").strip() for name in ("MOTO_INSTANCE_ID", "INSTANCE_ID")):
        missing.append("MOTO_INSTANCE_ID")
    if not system_config.internal_proxy_secret:
        missing.append("MOTO_INTERNAL_PROXY_SECRET")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Generic mode requires the following environment variables before startup: {joined}."
        )


def _apply_generic_mode_openrouter_env(api_client_manager) -> None:
    """Load the hosted OpenRouter key from env without using the desktop keyring."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "Generic mode started without OPENROUTER_API_KEY; OpenRouter can be configured later via proxied API routes."
        )
        return

    rag_config.openrouter_api_key = api_key
    rag_config.openrouter_enabled = True
    api_client_manager.set_openrouter_api_key(api_key)
    logger.info("Loaded OpenRouter API key from OPENROUTER_API_KEY for generic-mode startup")


def _restore_desktop_provider_credentials(api_client_manager) -> None:
    """Restore persisted desktop credentials from the OS-backed keyring."""
    from backend.shared.secret_store import (
        SecretStoreError,
        load_openrouter_api_key,
        load_wolfram_api_key,
    )
    from backend.shared.wolfram_alpha_client import initialize_wolfram_client

    # NOTE: We intentionally do NOT log `get_active_service_name()` or
    # `system_config.secret_namespace` here. Both values are purely diagnostic
    # identifiers (they contain no credential material), but CodeQL's
    # "clear-text logging of sensitive information" query treats any field
    # whose name starts with `secret_` as tainted, and any string derived
    # from it — including the OS-keyring service name — as sensitive. Logging
    # a boolean flag instead gives operators the diagnostic signal they need
    # (namespaced vs. default instance) without tripping the static analyzer.
    logger.info(
        "Secret store active: namespaced_instance=%s",
        bool(system_config.secret_namespace),
    )

    try:
        openrouter_api_key = load_openrouter_api_key()
        if openrouter_api_key:
            rag_config.openrouter_api_key = openrouter_api_key
            rag_config.openrouter_enabled = True
            api_client_manager.set_openrouter_api_key(openrouter_api_key)
            logger.info("Restored OpenRouter API key from secure backend storage")
        else:
            logger.info(
                "No OpenRouter API key found in secure backend storage for this namespace"
            )

        wolfram_api_key = load_wolfram_api_key()
        if wolfram_api_key:
            initialize_wolfram_client(wolfram_api_key)
            system_config.wolfram_alpha_api_key = wolfram_api_key
            system_config.wolfram_alpha_enabled = True
            logger.info("Restored Wolfram Alpha API key from secure backend storage")
        else:
            logger.info(
                "No Wolfram Alpha API key found in secure backend storage for this namespace"
            )
    except SecretStoreError as exc:
        logger.warning("Secure credential storage unavailable on startup: %s", exc)
    except Exception as exc:
        logger.warning("Failed to restore provider credentials on startup: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app."""
    _apply_generic_mode_from_env()
    _validate_generic_mode_startup_env()

    # Startup
    logger.info(
        "Starting ASI Aggregator System instance '%s' (data_dir=%s, logs_dir=%s)",
        system_config.instance_id,
        system_config.data_dir,
        system_config.logs_dir,
    )

    # Ensure per-instance mutable directories exist before subsystems touch them.
    Path(system_config.data_dir).mkdir(parents=True, exist_ok=True)
    Path(system_config.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(system_config.user_uploads_dir).mkdir(parents=True, exist_ok=True)

    from backend.shared.api_client_manager import api_client_manager

    if system_config.generic_mode:
        logger.info("Generic mode enabled - LM Studio and OS keyring restore are bypassed")
        _apply_generic_mode_openrouter_env(api_client_manager)
    else:
        # Restore securely persisted provider credentials before the UI checks status.
        _restore_desktop_provider_credentials(api_client_manager)

        # Test LM Studio connection (non-blocking - system works without it)
        connected = await lm_studio_client.test_connection()
        if not connected:
            logger.warning("LM Studio not available. System will default to OpenRouter when configured.")
    
    # CRITICAL: Restore session context on startup to display existing data
    # This ensures brainstorms and papers are loaded from the correct session directory
    # without requiring the user to click "Start" first
    try:
        from backend.autonomous.memory.session_manager import session_manager
        from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
        from backend.autonomous.memory.paper_library import paper_library
        from backend.autonomous.memory.research_metadata import research_metadata
        from backend.autonomous.memory.final_answer_memory import final_answer_memory
        from backend.autonomous.memory.proof_database import proof_database
        
        # Check for a resumable session
        interrupted_session = await session_manager.find_interrupted_session(system_config.auto_sessions_base_dir)
        if interrupted_session:
            session_id = interrupted_session["session_id"]
            logger.info(f"Found resumable session on startup: {session_id}")
            
            # Resume the session to set the correct path context
            await session_manager.resume_session(session_id, system_config.auto_sessions_base_dir)
            
            # Set session manager on all memory modules so they use session paths
            brainstorm_memory.set_session_manager(session_manager)
            paper_library.set_session_manager(session_manager)
            research_metadata.set_session_manager(session_manager)
            final_answer_memory.set_session_manager(session_manager)
            proof_database.set_session_manager(session_manager)
            
            logger.info(f"Session context restored - brainstorms and papers will load from session: {session_id}")
        else:
            logger.info("No resumable session found - using legacy paths")
            proof_database.set_session_manager(None)

        await proof_database.initialize()
    except Exception as e:
        logger.warning(f"Failed to restore session context on startup: {e}")
        # Non-fatal - continue with legacy paths
    
    # Set WebSocket broadcaster
    coordinator.set_websocket_broadcaster(websocket.broadcast_event)
    compiler_coordinator.set_websocket_broadcaster(websocket.broadcast_event)
    autonomous_coordinator.set_broadcast_callback(websocket.broadcast_event)
    
    # Set boost manager broadcaster
    from backend.shared.boost_manager import boost_manager
    boost_manager.set_broadcast_callback(websocket.broadcast_event)
    
    # Set API client manager broadcaster (token tracking, rate limits, fallbacks)
    api_client_manager.set_broadcast_callback(websocket.broadcast_event)

    # Lean 4 warm start must NEVER block the FastAPI lifespan. A cold Mathlib
    # workspace can spend many minutes inside `lake update` / `lake exe cache
    # get`, during which the backend would otherwise refuse every HTTP request
    # (including `/api/openrouter/api-key-status`). Users then see the UI
    # report "no OpenRouter key" even though the key is persisted in the OS
    # keyring, until they happen to poll again after the bootstrap finishes.
    # We fire-and-forget the warm start on a background task so the rest of
    # the API is reachable the moment uvicorn is ready to accept connections.
    lean4_warm_start_task: Optional[asyncio.Task] = None
    if system_config.lean4_enabled:
        try:
            lean4_client = initialize_lean4_client()
        except Exception as exc:
            logger.warning("Lean 4 client initialization failed: %s", exc)
        else:
            async def _warm_start_lean4() -> None:
                try:
                    await lean4_client.warm_start()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Lean 4 client warm start failed: %s", exc)

            lean4_warm_start_task = asyncio.create_task(_warm_start_lean4())

    logger.info("ASI Aggregator System ready")

    yield
    
    # Shutdown
    logger.info("Shutting down ASI Aggregator System...")
    if lean4_warm_start_task is not None and not lean4_warm_start_task.done():
        lean4_warm_start_task.cancel()
        try:
            await lean4_warm_start_task
        except (asyncio.CancelledError, Exception):
            pass
    await coordinator.stop()
    await compiler_coordinator.stop()
    await autonomous_coordinator.stop()
    await close_lean4_client()
    clear_lean4_client()
    await lm_studio_client.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ASI Aggregator System",
    description="AI-powered aggregator with RAG and multi-agent validation",
    version=build_info.version,
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Include routers
app.include_router(aggregator.router)
app.include_router(compiler.router)
app.include_router(autonomous.router)
app.include_router(boost.router)
app.include_router(workflow.router)
app.include_router(features.router)
app.include_router(health.router)
app.include_router(proofs.router)
app.include_router(openrouter.router)
app.include_router(download.router)
app.include_router(websocket.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ASI Aggregator System",
        "version": build_info.version,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=system_config.backend_host,
        port=system_config.backend_port,
        access_log=False,
    )

