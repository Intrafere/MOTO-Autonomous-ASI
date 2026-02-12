"""
Wolfram Alpha API client for mathematical verification.

Provides async HTTP client for Wolfram Alpha Simple API with singleton pattern.
Documentation: https://products.wolframalpha.com/simple-api/documentation
"""
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
        try:
            params = {
                "appid": self.api_key,
                "i": question
            }
            
            logger.info(f"Querying Wolfram Alpha: {question[:100]}")
            response = await self.client.get(self.BASE_URL, params=params)
            
            if response.status_code == 200:
                result = response.text.strip()
                logger.info(f"Wolfram Alpha success: {result[:200]}")
                return result
            elif response.status_code == 401:
                logger.warning("Wolfram Alpha: Invalid API key (401)")
                return None
            elif response.status_code == 403:
                logger.warning("Wolfram Alpha: API key forbidden or rate limited (403)")
                return None
            elif response.status_code == 501:
                logger.warning(f"Wolfram Alpha: Could not interpret query (501): {question}")
                return None
            else:
                logger.warning(f"Wolfram Alpha query failed: status {response.status_code}")
                return None
                
        except httpx.TimeoutException:
            logger.warning(f"Wolfram Alpha query timeout after 30s: {question[:100]}")
            return None
        except Exception as e:
            logger.error(f"Wolfram Alpha API error: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
        logger.info("Wolfram Alpha client closed")


# Singleton instance
_wolfram_alpha_client: Optional[WolframAlphaClient] = None


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


def clear_wolfram_client() -> None:
    """Clear the Wolfram Alpha client singleton."""
    global _wolfram_alpha_client
    _wolfram_alpha_client = None
    logger.info("Wolfram Alpha client cleared")

