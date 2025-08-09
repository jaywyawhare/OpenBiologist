import asyncio
import httpx
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from src.utils.models import RichToolDescription
import os

class ProteinPredictionRequest(BaseModel):
    sequence: str = Field(description="Protein amino acid sequence")
    job_name: str = Field(description="Name for the prediction job")
    model_type: str = Field(default="esmfold", description="Model to use for prediction")

class ProteinPredictionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None

class ProteinPredictionStatus(BaseModel):
    job_id: str
    status: str
    pdb_content: Optional[str] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None

class ProteinStructureService:
    def __init__(self, api_base_url: Optional[str] = None):
        # Read from env with localhost fallback
        self.api_base_url = api_base_url or os.getenv("PROTEIN_FOLDING_ENDPOINT", "http://localhost:7114")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def submit_prediction(self, sequence: str, job_name: str, model_type: str = "esmfold") -> ProteinPredictionResponse:
        job_id = f"openbio_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "job_id": job_id,
            "job_name": job_name,
            "sequence": sequence,
            "model": model_type,
            "user_id": "openbiologist_user"
        }
        
        try:
            response = await self.client.post(
                f"{self.api_base_url}/predict",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return ProteinPredictionResponse(
                job_id=result["job_id"],
                status=result["status"],
                message=result["message"],
                estimated_time=300  # 5 minutes estimate
            )
        except Exception as e:
            return ProteinPredictionResponse(
                job_id=job_id,
                status="error",
                message=f"Failed to submit prediction: {str(e)}"
            )
    
    async def check_status(self, job_id: str) -> ProteinPredictionStatus:
        """Check the status of a prediction job"""
        try:
            response = await self.client.get(f"{self.api_base_url}/status/{job_id}")
            response.raise_for_status()
            
            data = response.json()
            return ProteinPredictionStatus(
                job_id=data["job_id"],
                status=data["status"],
                pdb_content=data.get("pdb_content"),
                confidence_score=data.get("plddt_score"),
                error_message=data.get("error_message")
            )
        except Exception as e:
            return ProteinPredictionStatus(
                job_id=job_id,
                status="error",
                error_message=f"Failed to check status: {str(e)}"
            )
    
    async def wait_for_completion(self, job_id: str, max_wait_time: int = 600) -> ProteinPredictionStatus:
        """Wait for a prediction job to complete"""
        start_time = datetime.now()
        
        while True:
            status = await self.check_status(job_id)
            
            if status.status in ["successful", "crashed", "error"]:
                return status
            
            # Check if we've exceeded max wait time
            elapsed = (datetime.now() - start_time).seconds
            if elapsed > max_wait_time:
                return ProteinPredictionStatus(
                    job_id=job_id,
                    status="timeout",
                    error_message="Prediction timed out"
                )
            
            # Wait before checking again
            await asyncio.sleep(10)
    
    async def validate_sequence(self, sequence: str) -> Dict[str, Any]:
        """Validate a protein sequence"""
        # Basic validation
        valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequence = sequence.upper().strip()
        
        if not sequence:
            return {"valid": False, "error": "Empty sequence"}
        
        if len(sequence) < 10:
            return {"valid": False, "error": "Sequence too short (minimum 10 amino acids)"}
        
        if len(sequence) > 2000:
            return {"valid": False, "error": "Sequence too long (maximum 2000 amino acids)"}
        
        invalid_chars = [char for char in sequence if char not in valid_amino_acids]
        if invalid_chars:
            return {"valid": False, "error": f"Invalid amino acids: {set(invalid_chars)}"}
        
        return {
            "valid": True,
            "length": len(sequence),
            "amino_acid_count": {aa: sequence.count(aa) for aa in valid_amino_acids if sequence.count(aa) > 0}
        }

# Global service instance
protein_service = ProteinStructureService() 