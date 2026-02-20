"""
FastAPI main application.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from backend.api.middleware import setup_middleware
from backend.api.routes import aggregator, websocket, compiler, autonomous, boost, workflow, openrouter
from backend.shared.lm_studio_client import lm_studio_client
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app."""
    # Startup
    logger.info("Starting ASI Aggregator System...")
    
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
        
        # Check for a resumable session
        interrupted_session = await session_manager.find_interrupted_session()
        if interrupted_session:
            session_id = interrupted_session["session_id"]
            logger.info(f"Found resumable session on startup: {session_id}")
            
            # Resume the session to set the correct path context
            await session_manager.resume_session(session_id)
            
            # Set session manager on all memory modules so they use session paths
            brainstorm_memory.set_session_manager(session_manager)
            paper_library.set_session_manager(session_manager)
            research_metadata.set_session_manager(session_manager)
            final_answer_memory.set_session_manager(session_manager)
            
            logger.info(f"Session context restored - brainstorms and papers will load from session: {session_id}")
        else:
            logger.info("No resumable session found - using legacy paths")
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
    
    logger.info("ASI Aggregator System ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASI Aggregator System...")
    await coordinator.stop()
    await compiler_coordinator.stop()
    await autonomous_coordinator.stop()
    await lm_studio_client.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ASI Aggregator System",
    description="AI-powered aggregator with RAG and multi-agent validation",
    version="1.0.2",
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
app.include_router(openrouter.router)
app.include_router(websocket.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ASI Aggregator System",
        "version": "1.0.2",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)

