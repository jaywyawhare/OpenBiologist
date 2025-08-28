"""
Local Protein Predictor Service
Uses simplified ESMFold implementation for local protein structure prediction
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
from pathlib import Path
import json
import torch

from src.fold_models import ESMFold, ESMFoldConfig

logger = logging.getLogger(__name__)

class LocalProteinPredictionRequest(BaseModel):
    sequence: str = Field(description="Protein amino acid sequence")
    job_name: str = Field(description="Name for the prediction job")
    model_type: str = Field(default="esmfold", description="Model to use for prediction")

class LocalProteinPredictionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None

class LocalProteinPredictionStatus(BaseModel):
    job_id: str
    status: str
    pdb_content: Optional[str] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None

class LocalProteinStructureService:
    """Local protein structure prediction service using ESMFold"""
    
    def __init__(self):
        self.model: Optional[ESMFold] = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results_dir = Path("protein_folding_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ESMFold model"""
        try:
            logger.info("Initializing ESMFold model...")
            config = ESMFoldConfig()
            self.model = ESMFold(config)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("ESMFold model moved to GPU")
            else:
                logger.info("ESMFold model running on CPU")
                
            logger.info("ESMFold model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ESMFold model: {e}")
            self.model = None
    
    async def submit_prediction(self, sequence: str, job_name: str, model_type: str = "esmfold") -> LocalProteinPredictionResponse:
        """Submit a protein structure prediction job"""
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        
        # Validate sequence
        validation = await self.validate_sequence(sequence)
        if not validation["valid"]:
            return LocalProteinPredictionResponse(
                job_id=job_id,
                status="error",
                message=f"Sequence validation failed: {validation['error']}"
            )
        
        # Check if model is available
        if self.model is None:
            return LocalProteinPredictionResponse(
                job_id=job_id,
                status="error",
                message="ESMFold model not available"
            )
        
        # Create job record
        job_info = {
            "job_id": job_id,
            "job_name": job_name,
            "sequence": sequence,
            "model_type": model_type,
            "status": "pending",
            "created_at": datetime.now(),
            "user_id": "local_user"
        }
        
        self.jobs[job_id] = job_info
        
        # Start processing in background
        asyncio.create_task(self._process_job(job_id))
        
        return LocalProteinPredictionResponse(
            job_id=job_id,
            status="pending",
            message="Job submitted successfully to local ESMFold model",
            estimated_time=180  # 3 minutes estimate for local processing
        )
    
    async def _process_job(self, job_id: str):
        """Process a protein folding job"""
        try:
            job_info = self.jobs[job_id]
            sequence = job_info["sequence"]
            
            logger.info(f"Processing job {job_id} with sequence length {len(sequence)}")
            
            # Update status
            job_info["status"] = "processing"
            
            # Run prediction
            pdb_content = await self._run_prediction(sequence)
            
            # Save results
            result_file = self.results_dir / f"{job_id}.pdb"
            with open(result_file, "w") as f:
                f.write(pdb_content)
            
            # Update job info
            job_info["status"] = "successful"
            job_info["completed_at"] = datetime.now()
            job_info["result_path"] = str(result_file)
            job_info["pdb_content"] = pdb_content
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            job_info = self.jobs.get(job_id, {})
            job_info["status"] = "crashed"
            job_info["completed_at"] = datetime.now()
            job_info["error_message"] = str(e)
    
    async def _run_prediction(self, sequence: str) -> str:
        """Run protein structure prediction using ESMFold model"""
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            pdb_content = await loop.run_in_executor(
                None, 
                self._predict_structure, 
                sequence
            )
            return pdb_content
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _predict_structure(self, sequence: str) -> str:
        """Predict protein structure using ESMFold"""
        try:
            # Use ESMFold's infer_pdb method
            pdb_content = self.model.infer_pdb(sequence)
            return pdb_content
        except Exception as e:
            logger.error(f"ESMFold prediction failed: {e}")
            raise
    
    async def check_status(self, job_id: str) -> LocalProteinPredictionStatus:
        """Check the status of a prediction job"""
        job_info = self.jobs.get(job_id)
        
        if not job_info:
            return LocalProteinPredictionStatus(
                job_id=job_id,
                status="not_found",
                error_message="Job not found"
            )
        
        return LocalProteinPredictionStatus(
            job_id=job_info["job_id"],
            status=job_info["status"],
            pdb_content=job_info.get("pdb_content"),
            confidence_score=job_info.get("confidence_score"),
            error_message=job_info.get("error_message")
        )
    
    async def wait_for_completion(self, job_id: str, max_wait_time: int = 600) -> LocalProteinPredictionStatus:
        """Wait for a prediction job to complete"""
        start_time = datetime.now()
        
        while True:
            status = await self.check_status(job_id)
            
            if status.status in ["successful", "crashed", "error"]:
                return status
            
            # Check if we've exceeded max wait time
            elapsed = (datetime.now() - start_time).seconds
            if elapsed > max_wait_time:
                return LocalProteinPredictionStatus(
                    job_id=job_id,
                    status="timeout",
                    error_message="Prediction timed out on local service"
                )
            
            # Wait before checking again
            await asyncio.sleep(5)
    
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the local protein folding service"""
        try:
            if self.model is None:
                return {
                    "status": "unhealthy",
                    "model_loaded": False,
                    "device": "unknown",
                    "active_jobs": 0,
                    "total_jobs": 0,
                    "error": "ESMFold model not initialized"
                }
            
            # Check if model is on GPU
            device_info = "GPU" if self.model.device.type == "cuda" else "CPU"
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "device": device_info,
                "active_jobs": len([j for j in self.jobs.values() if j["status"] in ["pending", "processing"]]),
                "total_jobs": len(self.jobs)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "device": "unknown",
                "active_jobs": 0,
                "total_jobs": 0,
                "error": f"Health check failed: {str(e)}"
            }
    
    def get_job_history(self, limit: int = 10) -> list:
        """Get recent job history"""
        sorted_jobs = sorted(
            self.jobs.values(), 
            key=lambda x: x.get("created_at", datetime.min), 
            reverse=True
        )
        return sorted_jobs[:limit]

# Global service instance
local_protein_service = LocalProteinStructureService()
