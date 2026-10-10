"""
Wolfram Alpha API client for mathematical verification.

Provides async HTTP client for Wolfram Alpha Simple API with singleton pattern.
Documentation: https://products.wolframalpha.com/simple-api/documentation
"""
import httpx
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULT = "no_result"
    RETRYABLE_FAILURE = "retryable_failure"
    RATE_LIMITED = "rate_limited"
    REPAIR_REQUIRED = "repair_required"


@dataclass(frozen=True)
class WolframToolResult:
    status: ToolResultStatus
    content: Optional[str] = None
    safe_error: str = ""
    status_code: Optional[int] = None
    retry_after_seconds: Optional[int] = None


class WolframAlphaClient:
    """
    Client for Wolfram Alpha Simple API.
    
    Uses the Simple API which returns plain text answers to natural language queries.
    Graceful error handling - returns None on failures, never raises exceptions.
    """
    
    BASE_URL = "https://api.wolframalpha.com/v1/result"
    
    def __init__(self, api_key: str):
        """
        Initialize Wolfram Alpha client.
        
        Args:
            api_key: Wolfram Alpha App ID from developer portal
        """
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Wolfram Alpha client initialized")
    
    async def query(self, question: str) -> Optional[str]:
        """
        Query Wolfram Alpha with natural language question.
        
        Args:
            question: Natural language math question (e.g., "Is pi algebraic?")
        
        Returns:
            Answer string on success (200 status), None on failure
        """
        result = await self.query_result(question)
        return result.content if result.status == ToolResultStatus.SUCCESS else None

    async def query_result(self, question: str) -> WolframToolResult:
        """Query Wolfram Alpha while preserving failure versus no-result semantics."""
        try:
            params = {
                "appid": self.api_key,
                "i": question
            }
            
            logger.info("Querying Wolfram Alpha (query_len=%d)", len(question or ""))
            response = await self.client.get(self.BASE_URL, params=params)
            
            if response.status_code == 200:
                result = response.text.strip()
                logger.info("Wolfram Alpha success (result_len=%d)", len(result))
                return WolframToolResult(ToolResultStatus.SUCCESS, content=result)
            elif response.status_code == 401:
                logger.warning("Wolfram Alpha: Invalid API key (401)")
                return WolframToolResult(
                    ToolResultStatus.REPAIR_REQUIRED,
                    safe_error="Invalid Wolfram Alpha API key.",
                    status_code=401,
                )
            elif response.status_code == 403:
                retry_after = response.headers.get("retry-after")
                retry_after_seconds = (
                    int(retry_after) if str(retry_after or "").isdigit() else None
                )
                if retry_after_seconds is not None:
                    logger.warning("Wolfram Alpha: Rate limited (403)")
                    return WolframToolResult(
                        ToolResultStatus.RATE_LIMITED,
                        safe_error="Wolfram Alpha rate limit reached.",
                        status_code=403,
                        retry_after_seconds=retry_after_seconds,
                    )
                logger.warning("Wolfram Alpha: API key forbidden (403)")
                return WolframToolResult(
                    ToolResultStatus.REPAIR_REQUIRED,
                    safe_error="Wolfram Alpha access is forbidden. Check the App ID.",
                    status_code=403,
                )
            elif response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                logger.warning("Wolfram Alpha: Rate limited (429)")
                return WolframToolResult(
                    ToolResultStatus.RATE_LIMITED,
                    safe_error="Wolfram Alpha rate limit reached.",
                    status_code=429,
                    retry_after_seconds=int(retry_after) if str(retry_after or "").isdigit() else None,
                )
            elif response.status_code == 501:
                logger.warning("Wolfram Alpha: Could not interpret query (501; query_len=%d)", len(question or ""))
                return WolframToolResult(
                    ToolResultStatus.NO_RESULT,
                    safe_error="Wolfram Alpha could not interpret the query.",
                    status_code=501,
                )
            else:
                logger.warning(f"Wolfram Alpha query failed: status {response.status_code}")
                status = (
                    ToolResultStatus.RETRYABLE_FAILURE
                    if response.status_code >= 500
                    else ToolResultStatus.REPAIR_REQUIRED
                )
                return WolframToolResult(
                    status,
                    safe_error=f"Wolfram Alpha request failed with status {response.status_code}.",
                    status_code=response.status_code,
                )
                
        except httpx.TimeoutException:
            logger.warning("Wolfram Alpha query timeout after 30s (query_len=%d)", len(question or ""))
            return WolframToolResult(
                ToolResultStatus.RETRYABLE_FAILURE,
                safe_error="Wolfram Alpha request timed out.",
            )
        except Exception as e:
            logger.error(f"Wolfram Alpha API error: {e}", exc_info=True)
            return WolframToolResult(
                ToolResultStatus.RETRYABLE_FAILURE,
                safe_error="Wolfram Alpha request failed.",
            )
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
        logger.info("Wolfram Alpha client closed")


# Singleton instance
_wolfram_alpha_client: Optional[WolframAlphaClient] = None


async def replace_wolfram_client(api_key: str) -> None:
    """Replace the singleton and close the superseded HTTP client."""
    global _wolfram_alpha_client
    previous = _wolfram_alpha_client
    _wolfram_alpha_client = WolframAlphaClient(api_key)
    if previous is not None:
        await previous.close()
    logger.info("Wolfram Alpha singleton client initialized")


def initialize_wolfram_client(api_key: str) -> None:
    """
    Initialize Wolfram Alpha client with API key.
    
    Args:
        api_key: Wolfram Alpha App ID
    """
    global _wolfram_alpha_client
    _wolfram_alpha_client = WolframAlphaClient(api_key)
    logger.info("Wolfram Alpha singleton client initialized")


def get_wolfram_client() -> Optional[WolframAlphaClient]:
    """
    Get the Wolfram Alpha client instance.
    
    Returns:
        WolframAlphaClient instance or None if not initialized
    """
    return _wolfram_alpha_client


async def clear_wolfram_client() -> None:
    """Clear and close the Wolfram Alpha client singleton."""
    global _wolfram_alpha_client
    previous = _wolfram_alpha_client
    _wolfram_alpha_client = None
    if previous is not None:
        await previous.close()
    logger.info("Wolfram Alpha client cleared")

