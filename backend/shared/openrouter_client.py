"""
OpenRouter HTTP API client for generating completions.

TEMPERATURE POLICY: All completions use temperature=0.0 (deterministic generation) by default.
The system's evolving context provides sufficient diversity through growing databases and feedback loops.
"""
import httpx
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds
    RATE_LIMIT_COOLDOWN = 3600.0  # 1 hour in seconds
    
    # Per-model semaphores for rate limiting
    _model_semaphores: Dict[str, asyncio.Semaphore] = {}
    _semaphore_lock = asyncio.Lock()
    
    # Track rate-limited free models
    # Maps model_id -> timestamp when rate limit was hit
    _rate_limited_models: Dict[str, float] = {}
    _rate_limit_lock = asyncio.Lock()
    
    # App attribution for OpenRouter leaderboards
    # See: https://openrouter.ai/docs/app-attribution
    APP_URL = "https://intrafere.com/moto-autonomous-home-ai/"
    APP_TITLE = "MOTO Deep Research Harness"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=None,  # No timeout - continuous runtime
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            )
        )
    
    async def _get_model_semaphore(self, model: str) -> asyncio.Semaphore:
        """
        Get or create semaphore for a specific model.
        Each model gets its own semaphore (limit=1) to prevent concurrent requests.
        
        Args:
            model: Model name/identifier
            
        Returns:
            Semaphore for this specific model
        """
        async with self._semaphore_lock:
            if model not in self._model_semaphores:
                self._model_semaphores[model] = asyncio.Semaphore(1)
                logger.debug(f"Created semaphore for OpenRouter model: {model}")
            return self._model_semaphores[model]
    
    def _is_free_model(self, model: str) -> bool:
        """
        Check if a model is a free model (subject to rate limits).
        
        Free models are identified by the ':free' suffix in their model ID.
        
        Args:
            model: Model identifier
            
        Returns:
            True if model is a free model, False otherwise
        """
        return ":free" in model.lower()
    
    async def _is_model_rate_limited(self, model: str) -> tuple[bool, Optional[float]]:
        """
        Check if a model is currently rate-limited.
        
        Args:
            model: Model identifier
            
        Returns:
            Tuple of (is_rate_limited, retry_after_timestamp)
            - is_rate_limited: True if model is rate-limited and cooldown hasn't elapsed
            - retry_after_timestamp: Timestamp when model can be retried (None if not rate-limited)
        """
        async with self._rate_limit_lock:
            if model not in self._rate_limited_models:
                return False, None
            
            rate_limit_time = self._rate_limited_models[model]
            current_time = time.time()
            elapsed = current_time - rate_limit_time
            
            # If 1 hour has passed, remove from rate limit tracking
            if elapsed >= self.RATE_LIMIT_COOLDOWN:
                del self._rate_limited_models[model]
                logger.info(f"OpenRouter rate limit cooldown elapsed for model: {model}")
                return False, None
            
            # Still rate-limited
            retry_after = rate_limit_time + self.RATE_LIMIT_COOLDOWN
            return True, retry_after
    
    async def _record_rate_limit(self, model: str) -> float:
        """
        Record that a model hit rate limit.
        
        Args:
            model: Model identifier
            
        Returns:
            Timestamp when model can be retried (current_time + 1 hour)
        """
        current_time = time.time()
        async with self._rate_limit_lock:
            self._rate_limited_models[model] = current_time
        
        retry_after = current_time + self.RATE_LIMIT_COOLDOWN
        logger.warning(
            f"OpenRouter free model rate limit hit: {model}. "
            f"Pausing requests for 1 hour (until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))})."
        )
        return retry_after
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get standard headers including app attribution for OpenRouter leaderboards.
        
        Returns:
            Dict with Authorization, Content-Type, HTTP-Referer, and X-Title headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.APP_URL,
            "X-Title": self.APP_TITLE
        }
    
    async def list_models(self, free_only: bool = False) -> List[Dict[str, Any]]:
        """
        List available models from OpenRouter.
        
        Args:
            free_only: If True, only return models with $0 pricing (both prompt and completion)
        
        Returns:
            List of model objects, each containing:
            - id: Model identifier
            - name: Display name
            - providers: List of available providers for the model (if available)
            - pricing: Pricing information
            - context_length: Max context window
            - etc.
        """
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/models",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            
            # Filter for free models if requested
            if free_only:
                filtered_models = []
                for model in models:
                    pricing = model.get("pricing", {})
                    prompt_price = pricing.get("prompt", "0")
                    completion_price = pricing.get("completion", "0")
                    
                    # Convert to float for comparison (handle string values)
                    try:
                        is_free = float(prompt_price) == 0.0 and float(completion_price) == 0.0
                        if is_free:
                            filtered_models.append(model)
                    except (ValueError, TypeError):
                        # Skip models with invalid pricing data
                        continue
                
                logger.info(f"Filtered {len(filtered_models)} free models out of {len(models)} total")
                return sorted(filtered_models, key=lambda m: m.get("name", m.get("id", "")))
            
            # Return all models sorted by name
            return sorted(models, key=lambda m: m.get("name", m.get("id", "")))
        except Exception as e:
            logger.error(f"Failed to list OpenRouter models: {e}")
            return []
    
    async def get_model_providers(self, model_id: str) -> List[str]:
        """
        Get available providers/endpoints for a specific model using OpenRouter's
        dedicated endpoints API.
        
        Args:
            model_id: The OpenRouter model identifier (e.g., "anthropic/claude-3.5-sonnet")
            
        Returns:
            List of provider names that offer this model.
            Returns empty list if model not found or no providers available.
            
        Note:
            Uses OpenRouter's /api/v1/models/:author/:slug/endpoints endpoint
            to get the actual list of available providers for a model.
        """
        try:
            # Model ID format is "author/slug" (e.g., "anthropic/claude-3.5-sonnet")
            if "/" not in model_id:
                logger.warning(f"Invalid model ID format (expected 'author/slug'): {model_id}")
                return []
            
            parts = model_id.split("/", 1)
            if len(parts) != 2:
                logger.warning(f"Could not parse model ID: {model_id}")
                return []
            
            author, slug = parts
            
            # Call the dedicated endpoints API
            url = f"{self.BASE_URL}/models/{author}/{slug}/endpoints"
            logger.debug(f"Fetching providers from: {url}")
            
            response = await self.client.get(
                url,
                headers=self._get_headers()
            )
            
            if response.status_code == 404:
                logger.warning(f"Model {model_id} not found in OpenRouter")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            # Cache the response but don't spam logs with the full data
            logger.debug(f"OpenRouter endpoints API response for {model_id} (cached)")
            
            providers = []
            
            # The response should contain endpoint data with provider info
            # Expected structure: {"data": {"endpoints": [{"provider_name": "...", ...}, ...]}}
            # or similar variations
            
            if isinstance(data, dict):
                # Check for 'data' wrapper
                endpoints_data = data.get("data", data)
                
                # Check for 'endpoints' array
                endpoints = None
                if isinstance(endpoints_data, dict):
                    endpoints = endpoints_data.get("endpoints", [])
                elif isinstance(endpoints_data, list):
                    endpoints = endpoints_data
                
                if endpoints and isinstance(endpoints, list):
                    for endpoint in endpoints:
                        if isinstance(endpoint, dict):
                            # Check if provider is available (status == 0 means available)
                            status = endpoint.get("status", -1)
                            if status < 0:
                                # Skip unavailable providers
                                provider_name = endpoint.get("provider_name", "unknown")
                                logger.debug(f"Filtering out unavailable provider {provider_name} (status={status})")
                                continue
                            
                            # Try various field names for provider
                            provider = (
                                endpoint.get("provider_name") or
                                endpoint.get("provider") or
                                endpoint.get("name") or
                                endpoint.get("id")
                            )
                            if provider and isinstance(provider, str):
                                providers.append(provider)
                
                # Also check top-level for provider info
                if not providers:
                    if "provider_name" in endpoints_data:
                        providers.append(endpoints_data["provider_name"])
                    elif "provider" in endpoints_data:
                        providers.append(endpoints_data["provider"])
            
            # Deduplicate and sort (caching silently works behind the scenes)
            unique_providers = sorted(list(set(providers)))
            logger.debug(f"Available providers for {model_id}: {unique_providers}")
            return unique_providers
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching providers for {model_id}: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Failed to get providers for model {model_id}: {e}")
            return []
    
    async def generate_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion using OpenRouter API with validation and retry.
        
        Args:
            model: OpenRouter model identifier
            messages: Chat messages
            temperature: Sampling temperature (default 0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format constraints
            provider: Optional specific provider to use (None lets OpenRouter choose)
            
        Returns:
            API response dict
            
        Raises:
            CreditExhaustionError: When credits are exhausted (402 error)
            ValueError: For other API errors
        """
        model_semaphore = await self._get_model_semaphore(model)
        
        # ACQUIRE THIS MODEL'S SEMAPHORE to prevent concurrent requests
        async with model_semaphore:
            return await self._execute_completion_request(
                model, messages, temperature, max_tokens, response_format, provider
            )
    
    def _is_reasoning_model_without_temperature(self, model: str) -> bool:
        """
        Check if the model is a reasoning model that doesn't support temperature parameter.
        
        OpenAI's o-series reasoning models (o1, o1-mini, o1-preview, o4-mini, etc.)
        do not support the temperature parameter.
        
        Args:
            model: OpenRouter model identifier
            
        Returns:
            True if model doesn't support temperature, False otherwise
        """
        model_lower = model.lower()
        
        # OpenAI o-series reasoning models
        reasoning_model_patterns = [
            "/o1-",           # Matches openai/o1-mini, openai/o1-preview
            "/o1:",           # Matches openai/o1:version
            "/o4-mini",       # Matches openai/o4-mini-*
            "/o4:",           # Matches openai/o4:version
        ]
        
        return any(pattern in model_lower for pattern in reasoning_model_patterns)
    
    async def _execute_completion_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the actual completion request."""
        # Check if this model is currently rate-limited (for free models)
        if self._is_free_model(model):
            is_limited, retry_after = await self._is_model_rate_limited(model)
            if is_limited:
                retry_after_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))
                raise RateLimitError(
                    f"OpenRouter free model '{model}' is rate-limited. "
                    f"Retry after {retry_after_str} (1-hour cooldown).",
                    model=model,
                    retry_after=retry_after
                )
        
        # Calculate approximate token count for logging
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        approx_tokens = total_chars // 4
        
        payload = {
            "model": model,
            "messages": messages,
        }
        
        # Only add temperature if the model supports it
        # OpenAI o-series reasoning models (o1, o4-mini, etc.) do not support temperature
        if not self._is_reasoning_model_without_temperature(model):
            payload["temperature"] = temperature
        else:
            logger.debug(f"Skipping temperature parameter for reasoning model: {model}")
        
        # Set max_tokens if provided
        if max_tokens is None:
            max_tokens = 25000  # Default for reasoning models
            logger.debug(f"Auto-limiting max_tokens to {max_tokens}")
        
        payload["max_tokens"] = max_tokens
        
        if response_format:
            payload["response_format"] = response_format
        
        # Add provider routing if specified
        if provider:
            payload["provider"] = {"order": [provider]}
            logger.debug(f"Using specific provider: {provider}")
        
        # NOTE: Stop sequences were removed because they caused premature truncation
        # with certain models (e.g., Grok 4.1). Models will now generate until max_tokens
        # or natural completion. The json_parser handles any trailing garbage/padding.
        
        # Retry logic for transient errors (but NOT credit exhaustion)
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                    headers=self._get_headers()
                )
                
                # Check for credit exhaustion (402 Payment Required)
                if response.status_code == 402:
                    error_text = response.text
                    logger.error(
                        f"OpenRouter credit exhaustion detected (402): {error_text}"
                    )
                    raise CreditExhaustionError(
                        f"OpenRouter credits exhausted for model '{model}'. "
                        f"Falling back to LM Studio."
                    )
                
                response.raise_for_status()
                return response.json()
                
            except CreditExhaustionError:
                # Don't retry credit exhaustion - propagate immediately
                raise
            
            except OpenRouterPrivacyPolicyError:
                # Don't retry privacy policy errors - propagate immediately
                raise
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
                
                # Check for rate limit (429 Too Many Requests)
                if e.response.status_code == 429:
                    if self._is_free_model(model):
                        retry_after = await self._record_rate_limit(model)
                        retry_after_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))
                        logger.error(
                            f"OpenRouter rate limit detected (429) for free model '{model}': {error_detail}"
                        )
                        raise RateLimitError(
                            f"OpenRouter free model '{model}' hit rate limit. "
                            f"Pausing for 1 hour. Retry after {retry_after_str}.",
                            model=model,
                            retry_after=retry_after
                        )
                    else:
                        # Paid model hit rate limit - treat as transient error
                        logger.warning(f"OpenRouter rate limit (429) for paid model '{model}': {error_detail}")
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                            continue
                        raise ValueError(f"OpenRouter rate limit: {error_detail}")
                
                # Check for rate limit keywords in error message
                if any(keyword in error_detail.lower() for keyword in ["rate limit", "too many requests", "rate_limit"]):
                    if self._is_free_model(model):
                        retry_after = await self._record_rate_limit(model)
                        retry_after_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))
                        logger.error(
                            f"OpenRouter rate limit detected in error message for free model '{model}': {error_detail}"
                        )
                        raise RateLimitError(
                            f"OpenRouter free model '{model}' hit rate limit. "
                            f"Pausing for 1 hour. Retry after {retry_after_str}.",
                            model=model,
                            retry_after=retry_after
                        )
                
                # Check for privacy policy error (404 with specific message)
                # This occurs when user's OpenRouter privacy settings block free models
                if e.response.status_code == 404 and "data policy" in error_detail.lower():
                    logger.error(
                        f"OpenRouter privacy policy error detected (404): {error_detail}"
                    )
                    raise OpenRouterPrivacyPolicyError(
                        f"OpenRouter privacy settings are blocking access to free models. "
                        f"Please visit https://openrouter.ai/settings/privacy and enable "
                        f"the option to allow your data to be used for model training. "
                        f"Free models on OpenRouter require this setting to be enabled."
                    )
                
                # Check for credit-related errors in message
                if any(keyword in error_detail.lower() for keyword in ["credit", "insufficient", "balance", "quota"]):
                    logger.error(f"OpenRouter credit exhaustion detected in error message: {error_detail}")
                    raise CreditExhaustionError(
                        f"OpenRouter credits exhausted for model '{model}'. "
                        f"Falling back to LM Studio."
                    )
                
                logger.error(
                    f"OpenRouter HTTP {e.response.status_code} error (attempt {attempt + 1}/{self.MAX_RETRIES}): "
                    f"model={model}, approx_tokens={approx_tokens}, error={error_detail}"
                )
                
                # Retry on transient errors
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                
                raise ValueError(f"OpenRouter API error: {error_detail}")
                
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                # Use repr(e) to get detailed exception info even if str(e) is empty
                error_type = type(e).__name__
                error_detail = repr(e) if not str(e) else str(e)
                logger.warning(
                    f"OpenRouter connection error for model '{model}' (attempt {attempt + 1}/{self.MAX_RETRIES}): "
                    f"[{error_type}] {error_detail}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                raise ValueError(f"OpenRouter connection failed after {self.MAX_RETRIES} attempts: [{error_type}] {error_detail}")
                
            except Exception as e:
                logger.error(f"OpenRouter unexpected error: {e}")
                raise
        
        raise RuntimeError("OpenRouter completion generation failed after all retries")
    
    async def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings using OpenRouter API.
        
        Args:
            texts: List of texts to embed
            model: Embedding model (default: openai/text-embedding-3-small)
        
        Returns:
            List of embedding vectors
            
        Raises:
            CreditExhaustionError: When credits are exhausted (402 error)
            RateLimitError: When free model hits rate limit
            ValueError: For other API errors
        """
        if not texts:
            return []
        
        embedding_model = model or "openai/text-embedding-3-small"
        
        # Check if embedding model is rate-limited (for free models)
        if self._is_free_model(embedding_model):
            is_limited, retry_after = await self._is_model_rate_limited(embedding_model)
            if is_limited:
                retry_after_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))
                raise RateLimitError(
                    f"OpenRouter free embedding model '{embedding_model}' is rate-limited. "
                    f"Retry after {retry_after_str} (1-hour cooldown).",
                    model=embedding_model,
                    retry_after=retry_after
                )
        
        payload = {
            "model": embedding_model,
            "input": texts
        }
        
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/embeddings",
                json=payload,
                headers=self._get_headers()
            )
            
            # Check for credit exhaustion (402 Payment Required)
            if response.status_code == 402:
                error_text = response.text
                logger.error(f"OpenRouter credit exhaustion for embeddings (402): {error_text}")
                raise CreditExhaustionError("OpenRouter credits exhausted for embeddings")
            
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings in order
            embeddings = [
                item["embedding"]
                for item in sorted(data["data"], key=lambda x: x["index"])
            ]
            
            logger.debug(f"OpenRouter embeddings: {len(texts)} texts embedded with {embedding_model}")
            return embeddings
            
        except CreditExhaustionError:
            raise
        
        except OpenRouterPrivacyPolicyError:
            # Don't retry privacy policy errors - propagate immediately
            raise
        
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
            
            # Check for rate limit (429 Too Many Requests)
            if e.response.status_code == 429:
                if self._is_free_model(embedding_model):
                    retry_after = await self._record_rate_limit(embedding_model)
                    retry_after_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))
                    logger.error(
                        f"OpenRouter rate limit detected (429) for free embedding model '{embedding_model}': {error_detail}"
                    )
                    raise RateLimitError(
                        f"OpenRouter free embedding model '{embedding_model}' hit rate limit. "
                        f"Pausing for 1 hour. Retry after {retry_after_str}.",
                        model=embedding_model,
                        retry_after=retry_after
                    )
            
            # Check for rate limit keywords in error message
            if any(keyword in error_detail.lower() for keyword in ["rate limit", "too many requests", "rate_limit"]):
                if self._is_free_model(embedding_model):
                    retry_after = await self._record_rate_limit(embedding_model)
                    retry_after_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_after))
                    logger.error(
                        f"OpenRouter rate limit detected in error message for free embedding model '{embedding_model}': {error_detail}"
                    )
                    raise RateLimitError(
                        f"OpenRouter free embedding model '{embedding_model}' hit rate limit. "
                        f"Pausing for 1 hour. Retry after {retry_after_str}.",
                        model=embedding_model,
                        retry_after=retry_after
                    )
            
            # Check for privacy policy error (404 with specific message)
            if e.response.status_code == 404 and "data policy" in error_detail.lower():
                logger.error(f"OpenRouter privacy policy error for embeddings (404): {error_detail}")
                raise OpenRouterPrivacyPolicyError(
                    f"OpenRouter privacy settings are blocking access to free embedding models. "
                    f"Please visit https://openrouter.ai/settings/privacy and enable "
                    f"the option to allow your data to be used for model training."
                )
            
            # Check for credit-related errors in message
            if any(keyword in error_detail.lower() for keyword in ["credit", "insufficient", "balance", "quota"]):
                logger.error(f"OpenRouter credit exhaustion in embedding error: {error_detail}")
                raise CreditExhaustionError("OpenRouter credits exhausted for embeddings")
            
            logger.error(f"OpenRouter embedding HTTP error: {error_detail}")
            raise ValueError(f"OpenRouter embedding API error: {error_detail}")
        except Exception as e:
            logger.error(f"OpenRouter embedding error: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        try:
            await self.client.aclose()
            logger.info("OpenRouter client closed successfully")
        except Exception as e:
            logger.error(f"Error closing OpenRouter client: {e}")


class CreditExhaustionError(Exception):
    """Raised when OpenRouter credits are exhausted."""
    pass


class OpenRouterPrivacyPolicyError(Exception):
    """
    Raised when OpenRouter returns 404 due to privacy policy restrictions.
    
    This occurs when the user's OpenRouter account has privacy settings that
    block access to free models that require data sharing for training.
    
    Solution: User must go to https://openrouter.ai/settings/privacy and
    enable the option to allow data to be used for model training.
    """
    pass


class RateLimitError(Exception):
    """
    Raised when OpenRouter free model hits rate limit (429 or rate limit message).
    
    Free models have usage limits and require a cooldown period (typically 1 hour)
    before requests can resume. This error indicates the model should not be
    called again until the cooldown period elapses.
    
    Attributes:
        model: Model identifier that hit rate limit
        retry_after: Timestamp (Unix epoch) when model can be retried
    """
    def __init__(self, message: str, model: str, retry_after: float):
        super().__init__(message)
        self.model = model
        self.retry_after = retry_after

