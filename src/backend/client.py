"""
Client library for the OpenBiologist Protein Folding Backend
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

class ProteinFoldingClient:
    """Client for interacting with the protein folding backend API"""
    
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
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the backend"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager or call connect()")
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(method, url, **kwargs) as response:
            if response.status >= 400:
                error_text = await response.text()
                try:
                    error_data = await response.json()
                    error_detail = error_data.get('detail', error_text)
                except:
                    error_detail = error_text
                
                raise HTTPError(f"HTTP {response.status}: {error_detail}")
            
            if response.status == 204:  # No content
                return {}
            
            return await response.json()
    
    async def connect(self):
        """Initialize the client session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def disconnect(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the backend service"""
        return await self._request("GET", "/health")
    
    async def submit_prediction(
        self, 
        sequence: str, 
        job_name: str, 
        model_type: str = "esmfold"
    ) -> Dict[str, Any]:
        """Submit a protein structure prediction job"""
        data = {
            "sequence": sequence,
            "job_name": job_name,
            "model_type": model_type
        }
        return await self._request("POST", "/predict", json=data)
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a prediction job"""
        return await self._request("GET", f"/status/{job_id}")
    
    async def predict_sync(
        self, 
        sequence: str, 
        job_name: str, 
        model_type: str = "esmfold"
    ) -> Dict[str, Any]:
        """Submit a prediction and wait for completion"""
        data = {
            "sequence": sequence,
            "job_name": job_name,
            "model_type": model_type
        }
        return await self._request("POST", "/predict/sync", json=data)
    
    async def get_job_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get job history"""
        return await self._request("GET", f"/jobs?limit={limit}")
    
    async def download_pdb(self, job_id: str, output_path: str) -> bool:
        """Download PDB file for a completed job"""
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        url = f"{self.base_url}/download/{job_id}"
        
        async with self.session.get(url) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise HTTPError(f"HTTP {response.status}: {error_text}")
            
            # Save the file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
            
            return True
    
    async def delete_job(self, job_id: str) -> Dict[str, Any]:
        """Delete a job and its associated files"""
        return await self._request("DELETE", f"/jobs/{job_id}")
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return await self._request("GET", "/models")
    
    async def wait_for_completion(
        self, 
        job_id: str, 
        check_interval: int = 5,
        timeout: int = 600
    ) -> Dict[str, Any]:
        """Wait for a job to complete"""
        start_time = datetime.now()
        
        while True:
            status = await self.get_job_status(job_id)
            
            if status["status"] in ["successful", "crashed", "error"]:
                return status
            
            # Check timeout
            elapsed = (datetime.now() - start_time).seconds
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
            
            # Wait before checking again
            await asyncio.sleep(check_interval)

class HTTPError(Exception):
    """HTTP error from the backend API"""
    pass

# Convenience functions for synchronous usage
def submit_prediction_sync(
    sequence: str, 
    job_name: str, 
    model_type: str = "esmfold",
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Synchronous wrapper for submitting a prediction"""
    async def _submit():
        async with ProteinFoldingClient(base_url) as client:
            return await client.submit_prediction(sequence, job_name, model_type)
    
    return asyncio.run(_submit())

def predict_sync(
    sequence: str, 
    job_name: str, 
    model_type: str = "esmfold",
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Synchronous wrapper for synchronous prediction"""
    async def _predict():
        async with ProteinFoldingClient(base_url) as client:
            return await client.predict_sync(sequence, job_name, model_type)
    
    return asyncio.run(_predict())

def health_check_sync(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Synchronous wrapper for health check"""
    async def _health():
        async with ProteinFoldingClient(base_url) as client:
            return await client.health_check()
    
    return asyncio.run(_health())
