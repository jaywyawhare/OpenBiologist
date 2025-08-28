"""
FastAPI Backend for OpenBiologist Protein Folding Service
Provides REST API endpoints for protein structure prediction
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch

import sys

from src.fold_models import ESMFold, ESMFoldConfig
from src.services.local_protein_predictor import LocalProteinStructureService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenBiologist Protein Folding API",
    description="REST API for protein structure prediction using ESMFold",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ProteinPredictionRequest(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence", min_length=10, max_length=2000)
    job_name: str = Field(..., description="Name for the prediction job")
    model_type: str = Field(default="esmfold", description="Model to use for prediction")
    wait_for_completion: bool = Field(default=False, description="Wait for prediction to complete")

class ProteinPredictionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None
    created_at: datetime

class JobStatusResponse(BaseModel):
    job_id: str
    job_name: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    pdb_content: Optional[str] = None
    confidence_score: Optional[float] = None
    result_path: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    active_jobs: int
    total_jobs: int
    error: Optional[str] = None

class JobHistoryResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total_count: int

# Global service instance
protein_service = LocalProteinStructureService()

# Dependency to check if service is healthy
async def get_healthy_service():
    """Dependency to ensure the protein folding service is healthy"""
    health = await protein_service.health_check()
    if health["status"] != "healthy":
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {health.get('error', 'Unknown error')}"
        )
    return protein_service

@app.on_event("startup")
async def startup_event():
    """Initialize the protein folding service on startup"""
    logger.info("Starting OpenBiologist Protein Folding API...")
    
    # Initialize the protein folding service
    try:
        # The service is already initialized in the constructor
        health = await protein_service.health_check()
        if health["status"] == "healthy":
            logger.info("Protein folding service initialized successfully")
        else:
            logger.warning(f"Protein folding service has issues: {health.get('error')}")
    except Exception as e:
        logger.error(f"Failed to initialize protein folding service: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OpenBiologist Protein Folding API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the protein folding service"""
    try:
        health = await protein_service.health_check()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            active_jobs=0,
            total_jobs=0,
            error=str(e)
        )

@app.post("/predict", response_model=ProteinPredictionResponse)
async def predict_protein_structure(
    request: ProteinPredictionRequest,
    background_tasks: BackgroundTasks,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Submit a protein structure prediction job"""
    try:
        # Submit prediction
        response = await service.submit_prediction(
            sequence=request.sequence,
            job_name=request.job_name,
            model_type=request.model_type
        )
        
        # Return response with timestamp
        return ProteinPredictionResponse(
            job_id=response.job_id,
            status=response.status,
            message=response.message,
            estimated_time=response.estimated_time,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to submit prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Get the status of a protein prediction job"""
    try:
        status = await service.check_status(job_id)
        
        if status.status == "not_found":
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Convert to response model
        return JobStatusResponse(
            job_id=status.job_id,
            job_name=service.jobs.get(job_id, {}).get("job_name", "Unknown"),
            status=status.status,
            created_at=service.jobs.get(job_id, {}).get("created_at", datetime.now()),
            completed_at=service.jobs.get(job_id, {}).get("completed_at"),
            error_message=status.error_message,
            pdb_content=status.pdb_content,
            confidence_score=status.confidence_score,
            result_path=service.jobs.get(job_id, {}).get("result_path")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/sync", response_model=JobStatusResponse)
async def predict_protein_structure_sync(
    request: ProteinPredictionRequest,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Submit a protein structure prediction and wait for completion"""
    try:
        # Submit prediction
        response = await service.submit_prediction(
            sequence=request.sequence,
            job_name=request.job_name,
            model_type=request.model_type
        )
        
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        
        # Wait for completion
        status = await service.wait_for_completion(response.job_id, max_wait_time=600)
        
        # Convert to response model
        return JobStatusResponse(
            job_id=status.job_id,
            job_name=request.job_name,
            status=status.status,
            created_at=datetime.now(),
            completed_at=datetime.now() if status.status in ["successful", "crashed"] else None,
            error_message=status.error_message,
            pdb_content=status.pdb_content,
            confidence_score=status.confidence_score,
            result_path=service.jobs.get(status.job_id, {}).get("result_path")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=JobHistoryResponse)
async def get_job_history(
    limit: int = Query(default=10, ge=1, le=100, description="Number of jobs to return"),
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Get job history"""
    try:
        jobs = service.get_job_history(limit=limit)
        
        # Convert to response models
        job_responses = []
        for job in jobs:
            status = await service.check_status(job["job_id"])
            job_responses.append(JobStatusResponse(
                job_id=job["job_id"],
                job_name=job["job_name"],
                status=status.status,
                created_at=job["created_at"],
                completed_at=job.get("completed_at"),
                error_message=status.error_message,
                pdb_content=status.pdb_content,
                confidence_score=status.confidence_score,
                result_path=job.get("result_path")
            ))
        
        return JobHistoryResponse(
            jobs=job_responses,
            total_count=len(job_responses)
        )
        
    except Exception as e:
        logger.error(f"Failed to get job history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{job_id}")
async def download_pdb_file(
    job_id: str,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Download the PDB file for a completed job"""
    try:
        # Check job status
        status = await service.check_status(job_id)
        
        if status.status != "successful":
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} is not completed (status: {status.status})"
            )
        
        # Get result path
        job_info = service.jobs.get(job_id, {})
        result_path = job_info.get("result_path")
        
        if not result_path or not Path(result_path).exists():
            raise HTTPException(status_code=404, detail="PDB file not found")
        
        # Return file
        return FileResponse(
            path=result_path,
            filename=f"{job_id}.pdb",
            media_type="chemical/x-pdb"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download PDB file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Delete a job and its associated files"""
    try:
        # Check if job exists
        if job_id not in service.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get job info
        job_info = service.jobs[job_id]
        result_path = job_info.get("result_path")
        
        # Delete result file if it exists
        if result_path and Path(result_path).exists():
            Path(result_path).unlink()
            logger.info(f"Deleted result file: {result_path}")
        
        # Remove job from service
        del service.jobs[job_id]
        
        return {"message": f"Job {job_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get information about available protein folding models"""
    return {
        "models": [
            {
                "name": "ESMFold",
                "type": "esmfold",
                "description": "ESMFold protein structure prediction model",
                "supported_sequences": "10-2000 amino acids",
                "estimated_time": "3-10 minutes"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server for OpenBiologist Protein Folding API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
