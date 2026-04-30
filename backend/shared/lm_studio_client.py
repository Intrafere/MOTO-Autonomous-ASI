"""
LM Studio HTTP API client for generating completions and embeddings.

TEMPERATURE POLICY: All completions use temperature=0.0 (deterministic generation) by default.
The system's evolving context provides sufficient diversity through:
- Growing aggregator/compiler databases
- Rejection feedback loops
- Cyclic RAG chunk sizes (256→512→768→1024 for submitters)
- Completion reviews and cleanup cycles

Deterministic generation ensures reproducible results, consistent JSON formatting,
and stable validation decisions across long research sessions.
"""
import httpx
import asyncio
import time
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.shared.config import rag_config, system_config
import logging

logger = logging.getLogger(__name__)

# Ensure instance-scoped logs directory exists
Path(system_config.logs_dir).mkdir(parents=True, exist_ok=True)


class LMStudioClient:
    """Client for LM Studio API."""
    
    # Embedding performance settings
    EMBEDDING_BATCH_SIZE = 100  # Process embeddings in batches of 100
    EMBEDDING_TIMEOUT = None  # No timeout - continuous runtime
    MAX_RETRIES = 1  # Fail fast when LM Studio unavailable (OpenRouter fallback)
    RETRY_DELAY = 0.5  # seconds (not used when MAX_RETRIES=1)
    
    # Rate limiting semaphores
    _embedding_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent embedding requests
    _model_semaphores: Dict[str, asyncio.Semaphore] = {}  # Per-model semaphores for chat completions
    _semaphore_lock = asyncio.Lock()  # Thread-safe dictionary access
    
    # Model configuration cache
    _model_configs: Dict[str, Dict[str, Any]] = {}
    _config_lock = asyncio.Lock()
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or rag_config.lm_studio_base_url
        # Optimized HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=None,  # No timeout - continuous runtime
            limits=httpx.Limits(
                max_keepalive_connections=20,  # Connection pool
                max_connections=50,
                keepalive_expiry=30.0
            )
        )
    
    async def _get_model_semaphore(self, model: str) -> asyncio.Semaphore:
        """
        Get or create semaphore for a specific model.
        Each model gets its own semaphore (limit=1) to prevent concurrent requests.
        Different models can run in parallel.
        
        Args:
            model: Model name/identifier
            
        Returns:
            Semaphore for this specific model
        """
        async with self._semaphore_lock:
            if model not in self._model_semaphores:
                self._model_semaphores[model] = asyncio.Semaphore(1)
                logger.debug(f"Created semaphore for model: {model}")
            return self._model_semaphores[model]
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from LM Studio."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def get_loaded_models(self) -> List[str]:
        """
        Get list of currently LOADED models with their instance IDs.
        
        Returns model IDs as they appear at runtime (e.g., 'openai/gpt-oss-20b:2').
        This is different from list_models() which returns downloaded model names.
        
        Uses 'lms ps' command to get accurate loaded model list with instance IDs.
        
        Returns:
            List of loaded model IDs with instance suffixes
        """
        try:
            # Use 'lms ps' to get loaded models with instance IDs (NON-BLOCKING)
            process = await asyncio.create_subprocess_exec(
                "lms", "ps",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                logger.warning("'lms ps' timed out after 10s")
                process.kill()
                await process.wait()
                return []
            
            result_stdout = stdout.decode() if stdout else ""
            result_stderr = stderr.decode(errors="replace").strip() if stderr else ""
            result_returncode = process.returncode

            if result_returncode == 0:
                # Exit code 0 means success. Empty stdout simply means no models
                # are loaded — that is a normal state, not an error.
                models: List[str] = []
                if result_stdout:
                    for line in result_stdout.strip().split('\n'):
                        # Skip headers, separators, and empty lines
                        if not line or line.startswith('-') or line.startswith('ID') or line.startswith('Model'):
                            continue
                        # Extract model ID (first column)
                        parts = line.strip().split()
                        if parts:
                            models.append(parts[0])
                logger.debug(f"Loaded models from 'lms ps': {models}")
                return models
            else:
                if result_stderr:
                    logger.warning(
                        f"'lms ps' returned non-zero code {result_returncode}: {result_stderr}"
                    )
                else:
                    logger.warning(f"'lms ps' returned non-zero code {result_returncode}")
                return []
                
        except FileNotFoundError:
            logger.error("'lms' command not found in PATH")
            return []
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
            return []
    
    async def generate_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,  # Default to deterministic generation - evolving context provides diversity
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        skip_semaphore: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate a completion using LM Studio API with validation and retry.
        
        Args:
            skip_semaphore: If True, skips model semaphore acquisition (for non-blocking operations)
            tools: Optional OpenAI-compatible tool schemas (LM Studio 0.3+).
            tool_choice: Optional tool-choice directive.
        """
        # Get model-specific semaphore (allows different models to run in parallel)
        if skip_semaphore:
            # Direct execution without semaphore
            return await self._execute_completion_request(
                model, messages, temperature, max_tokens, response_format,
                tools=tools, tool_choice=tool_choice,
            )
        
        model_semaphore = await self._get_model_semaphore(model)
        
        # ACQUIRE THIS MODEL'S SEMAPHORE to prevent concurrent requests to same model
        async with model_semaphore:
            return await self._execute_completion_request(
                model, messages, temperature, max_tokens, response_format,
                tools=tools, tool_choice=tool_choice,
            )
    
    async def _execute_completion_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the actual completion request (extracted for semaphore bypass)."""
        # Calculate approximate token count for logging
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        approx_tokens = total_chars // 4  # Rough approximation
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # ALWAYS set max_tokens to prevent mid-generation context overflow
        # If not explicitly provided, use a generous default for reasoning models
        if max_tokens is None:
            max_tokens = 25000  # Increased to 25K to accommodate reasoning models with extensive thinking
            logger.debug(f"Auto-limiting max_tokens to {max_tokens} (25K for reasoning model support)")
        
        payload["max_tokens"] = max_tokens
        
        if response_format:
            payload["response_format"] = response_format
        
        # OpenAI-compatible tool calling (LM Studio 0.3+). We pass the tool
        # list straight through; LM Studio's OpenAI-compatible server either
        # surfaces tool_calls on the message or simply returns a normal
        # completion if the loaded model ignores tool schemas. Callers
        # detect the latter and fall back to single-shot.
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        
        # NOTE: Stop sequences were removed because they caused premature truncation
        # with certain models (e.g., Grok 4.1). Models will now generate until max_tokens
        # or natural completion. The json_parser handles any trailing garbage/padding.
        
        # Retry logic for transient errors
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
                    logger.error(
                        f"LM Studio 400 Bad Request (attempt {attempt + 1}/{max_retries + 1}): "
                        f"model={model}, approx_tokens={approx_tokens}, "
                        f"messages_count={len(messages)}, error={error_detail}"
                    )
                    
                    # Check error type
                    is_model_crash = "has crashed" in error_detail.lower() or "exit code:" in error_detail.lower()
                    is_regex_error = "failed to process regex" in error_detail.lower()
                    is_input_overflow = ("prompt" in error_detail.lower() and "too" in error_detail.lower()) or \
                                        ("input" in error_detail.lower() and "exceeds" in error_detail.lower()) or \
                                        ("prompt exceeds" in error_detail.lower())
                    is_mid_generation_overflow = "mid-generation" in error_detail.lower() or \
                                                 ("context length" in error_detail.lower() and "does not support" in error_detail.lower())
                    
                    if is_model_crash:
                        # Model crashed - LM Studio has unloaded it
                        logger.critical(
                            f"Model '{model}' CRASHED! Error: {error_detail}. "
                            f"Please reload the model in LM Studio."
                        )
                        raise ValueError(f"Model '{model}' crashed. Please reload it in LM Studio.")
                    
                    elif is_regex_error:
                        # LM Studio's internal regex engine failed
                        logger.error(
                            f"LM Studio regex processing failed! This is an LM Studio internal error. "
                            f"Prompt: ~{approx_tokens} tokens."
                        )
                        raise ValueError(
                            f"LM Studio regex engine failed. This may be transient. "
                            f"Submitter will retry on next iteration."
                        )
                    
                    elif is_mid_generation_overflow:
                        logger.error(
                            f"Mid-generation context overflow! This indicates max_tokens was not set correctly. "
                            f"Prompt: ~{approx_tokens} tokens, model context limit unknown."
                        )
                        raise ValueError(
                            f"Model ran out of context space during generation. "
                            f"This is a bug - max_tokens should prevent this."
                        )
                    
                    elif is_input_overflow:
                        import re
                        limit_match = re.search(r'context.*?(\d+)', error_detail.lower())
                        context_limit = int(limit_match.group(1)) if limit_match else "unknown"
                        
                        logger.error(
                            f"Input prompt too large! Prompt: ~{approx_tokens} tokens, "
                            f"Model context limit: {context_limit} tokens."
                        )
                        raise ValueError(
                            f"Prompt ({approx_tokens} tokens) exceeds model's context window ({context_limit} tokens). "
                            f"In LM Studio: Increase 'Context Length (n_ctx)' to at least {approx_tokens + 5000} tokens."
                        )
                    
                    # Retry on transient 400 errors
                    if attempt < max_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        logger.info(f"Retrying after 400 error...")
                        continue
                    
                    raise
                    
                elif e.response.status_code == 404:
                    logger.error(f"Model '{model}' not found (404). Please ensure it is loaded in LM Studio.")
                    raise
                else:
                    logger.error(f"HTTP {e.response.status_code} error: {e}")
                    raise
                    
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                logger.error(f"Connection error for model '{model}': {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise
                    
            except Exception as e:
                logger.error(f"Failed to generate completion: {e}")
                raise
        
        raise RuntimeError("Completion generation failed after all retries")
    
    async def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings using LM Studio API with rate limiting.
        Optimized with batching, retry logic, and performance metrics.
        """
        if not texts:
            return []
        
        # ACQUIRE SEMAPHORE for rate limiting
        async with self._embedding_semaphore:
            embedding_model = model or rag_config.embedding_model
            start_time = time.time()
            
            try:
                # Process in batches to avoid timeouts and improve throughput
                all_embeddings = []
                total_batches = (len(texts) + self.EMBEDDING_BATCH_SIZE - 1) // self.EMBEDDING_BATCH_SIZE
                
                for batch_idx in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
                    batch_texts = texts[batch_idx:batch_idx + self.EMBEDDING_BATCH_SIZE]
                    batch_num = batch_idx // self.EMBEDDING_BATCH_SIZE + 1
                    
                    logger.debug(
                        f"Embedding batch {batch_num}/{total_batches} "
                        f"({len(batch_texts)} texts)"
                    )
                    
                    # Retry logic for transient failures
                    batch_embeddings = await self._get_embeddings_with_retry(
                        batch_texts, 
                        embedding_model
                    )
                    all_embeddings.extend(batch_embeddings)
                
                elapsed = time.time() - start_time
                texts_per_sec = len(texts) / elapsed if elapsed > 0 else 0
                
                logger.debug(
                    f"Embeddings complete: {len(texts)} texts in {elapsed:.2f}s "
                    f"({texts_per_sec:.1f} texts/sec, {total_batches} batches)"
                )
                
                return all_embeddings
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Failed to get embeddings after {elapsed:.2f}s "
                    f"({len(texts)} texts): {e}"
                )
                raise
    
    async def _get_embeddings_with_retry(
        self, 
        texts: List[str], 
        model: str
    ) -> List[List[float]]:
        """Get embeddings with retry logic for transient failures."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                payload = {
                    "model": model,
                    "input": texts
                }
                
                response = await self.client.post(
                    f"{self.base_url}/v1/embeddings",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings in order
                embeddings = [
                    item["embedding"] 
                    for item in sorted(data["data"], key=lambda x: x["index"])
                ]
                return embeddings
                
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                if attempt < self.MAX_RETRIES:
                    logger.warning(
                        f"Embedding attempt {attempt}/{self.MAX_RETRIES} failed: {e}. "
                        f"Retrying in {self.RETRY_DELAY}s..."
                    )
                    await asyncio.sleep(self.RETRY_DELAY)
                else:
                    logger.error(
                        f"Embedding failed after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise
            except Exception as e:
                logger.error(f"Embedding failed with unexpected error: {e}")
                raise
    
    async def test_connection(self) -> bool:
        """Test connection to LM Studio (bounded, never blocks startup)."""
        try:
            # Hard cap the startup probe so a LM Studio process that bound the
            # port but never responds cannot stall the FastAPI lifespan.
            models = await asyncio.wait_for(self.list_models(), timeout=5.0)
            logger.info(f"Successfully connected to LM Studio. Found {len(models)} models.")
            return True
        except asyncio.TimeoutError:
            logger.warning("LM Studio startup probe timed out after 5s; treating as unavailable.")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {e}")
            return False
    
    async def check_availability(self) -> Dict[str, Any]:
        """
        Check if LM Studio server is reachable and has models loaded.
        
        Returns:
            Dict with:
            - available: bool - True if LM Studio is running and has models
            - has_models: bool - True if at least one model is loaded
            - model_count: int - Number of loaded models
            - models: List[str] - List of loaded model IDs
            - error: Optional[str] - Error message if unavailable
        """
        result = {
            "available": False,
            "has_models": False,
            "model_count": 0,
            "models": [],
            "error": None
        }
        
        try:
            # First check if server is reachable
            response = await self.client.get(f"{self.base_url}/v1/models", timeout=5.0)
            response.raise_for_status()

            # Server is reachable
            result["available"] = True

            # Extract models from the /v1/models response as a reliable fallback.
            # The `lms ps` CLI is preferred (it returns instance IDs), but the CLI
            # may be missing from PATH or slow/timing out during startup while
            # nomic is still loading. In either case we must NOT downgrade a
            # successful /v1/models response to "no models" — that produces a
            # phantom "LM Studio Offline" state even though embedding calls
            # are succeeding.
            http_models: List[str] = []
            try:
                data = response.json()
                for entry in data.get("data", []) or []:
                    if isinstance(entry, dict):
                        model_id = entry.get("id")
                        if isinstance(model_id, str) and model_id:
                            http_models.append(model_id)
            except Exception as parse_err:
                logger.debug(f"Could not parse /v1/models response body: {parse_err}")

            cli_models = await self.get_loaded_models()

            if cli_models:
                models = cli_models
                source = "lms ps"
            else:
                models = http_models
                source = "/v1/models"

            result["models"] = models
            result["model_count"] = len(models)
            result["has_models"] = len(models) > 0

            logger.debug(
                f"LM Studio availability check: {len(models)} models loaded (source: {source})"
            )
            return result
            
        except httpx.ConnectError:
            result["error"] = "Cannot connect to LM Studio server. Please ensure LM Studio is running."
            logger.warning(f"LM Studio availability check failed: {result['error']}")
            return result
        except httpx.TimeoutException:
            result["error"] = "Connection to LM Studio timed out."
            logger.warning(f"LM Studio availability check failed: {result['error']}")
            return result
        except Exception as e:
            result["error"] = f"Error checking LM Studio availability: {str(e)}"
            logger.warning(f"LM Studio availability check failed: {result['error']}")
            return result
    
    async def test_model_compatibility(self, model_name: str) -> tuple[bool, str, dict]:
        """
        Test if a model is compatible with the ASI system.
        
        Tests that the model can:
        - Generate non-empty responses
        - Produce valid JSON (REQUIRED for ASI system)
        - Return more than minimal tokens
        
        Args:
            model_name: Name of model to test
            
        Returns:
            Tuple of (is_compatible, error_message, details)
        """
        try:
            test_prompt = 'Output JSON: {"status": "ok", "test": "Model is compatible"}'
            
            response = await self.generate_completion(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.0,  # Deterministic generation for model health checks
                max_tokens=None
            )
            
            # Extract response details
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            
            details = {
                "model_name": model_name,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "content_length": len(content),
                "content_preview": content[:100] if content else "(empty)"
            }
            
            # Check 1: Empty or whitespace-only response
            if not content or not content.strip():
                error = f"Model '{model_name}' returned empty response (completion_tokens={completion_tokens})"
                logger.error(f"Compatibility test failed: {error}")
                logger.error(f"Details: {details}")
                return (False, error, details)
            
            # Check 2: Anomalously short response (< 5 tokens)
            if completion_tokens < 5:
                error = f"Model '{model_name}' returned too few tokens (completion_tokens={completion_tokens})"
                logger.warning(f"Compatibility test failed: {error}")
                logger.warning(f"Details: {details}")
                return (False, error, details)
            
            # Check 3: MUST parse as JSON (CRITICAL for ASI system)
            try:
                import json
                from backend.shared.json_parser import sanitize_json_response
                
                sanitized_content = sanitize_json_response(content)
                parsed_json = json.loads(sanitized_content)
                logger.info(f"Model '{model_name}' produced valid JSON: {parsed_json}")
            except json.JSONDecodeError as json_err:
                error = f"Model '{model_name}' FAILED to produce valid JSON: {json_err}"
                logger.error(f"Compatibility test FAILED: {error}")
                logger.error(f"Response content: {content}")
                logger.error(f"Details: {details}")
                return (False, error, details)
            
            # SUCCESS - Cache model config
            estimated_context = max(prompt_tokens * 4, 32768)
            
            model_config = {
                "model_path": model_name,
                "context_length": estimated_context,
                "compatibility_test_passed": True,
                "tested_at": datetime.now().isoformat()
            }
            
            await self.cache_model_load_config(model_name, model_config)
            
            logger.info(
                f"Model '{model_name}' passed compatibility test "
                f"(tokens={completion_tokens})"
            )
            return (True, "", details)
            
        except Exception as e:
            error = f"Model '{model_name}' compatibility test failed with exception: {str(e)}"
            logger.error(error)
            details = {
                "model_name": model_name,
                "exception": str(e),
                "exception_type": type(e).__name__
            }
            return (False, error, details)
    
    async def cache_model_load_config(self, model_id: str, config: Dict[str, Any]):
        """
        Cache model load configuration.
        
        Args:
            model_id: Model identifier
            config: Configuration dict (context_length, etc.)
        """
        async with self._config_lock:
            self._model_configs[model_id] = {
                "model_id": model_id,
                "cached_at": datetime.now().isoformat(),
                **config
            }
            logger.debug(f"Cached config for model '{model_id}'")
    
    async def get_cached_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached configuration for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Cached config dict or None
        """
        async with self._config_lock:
            return self._model_configs.get(model_id)
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        try:
            await self.client.aclose()
            logger.info("LM Studio client closed successfully")
        except Exception as e:
            logger.error(f"Error closing LM Studio client: {e}")


# Global client instance
lm_studio_client = LMStudioClient()
