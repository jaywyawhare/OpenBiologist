"""
Async Python client for the OpenBiologist Protein Folding API
Simplified client for direct protein structure prediction
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProteinFoldingClient:
    """Client for the OpenBiologist Protein Folding API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the protein folding service"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "device": "unknown",
                "error": str(e)
            }
    
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict protein structure from sequence"""
        try:
            payload = {"sequence": sequence}
            async with self.session.post(f"{self.base_url}/predict", json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return {"models": []}

# Synchronous wrapper functions for convenience
def predict_structure_sync(sequence: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Synchronous wrapper for protein structure prediction"""
    async def _predict():
        async with ProteinFoldingClient(base_url) as client:
            return await client.predict_structure(sequence)
    
    return asyncio.run(_predict())

def health_check_sync(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Synchronous wrapper for health check"""
    async def _health():
        async with ProteinFoldingClient(base_url) as client:
            return await client.health_check()
    
    return asyncio.run(_health())
