"""
FastAPI Backend for OpenBiologist Protein Folding Service
Provides REST API endpoints for protein structure prediction
"""

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    description="REST API for protein structure prediction using Boltz",
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

class ProteinPredictionResponse(BaseModel):
    status: str
    pdb_content: str
    sequence_length: int
    message: str

class ErrorResponse(BaseModel):
    status: str
    error: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    message: Optional[str] = None
    error: Optional[str] = None

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
    logger.info("Starting OpenBiologist Protein Folding API")
    try:
        # Check service health
        health = await protein_service.health_check()
        if health["status"] == "healthy":
            logger.info(f"Protein folding service ready on {health['device']}")
        else:
            logger.error(f"Protein folding service unhealthy: {health.get('error')}")
    except Exception as e:
        logger.error(f"Failed to initialize protein folding service: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenBiologist Protein Folding API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/predict", response_model=ProteinPredictionResponse)
async def predict_protein_structure(
    request: ProteinPredictionRequest,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Predict protein structure from amino acid sequence"""
    try:
        logger.info(f"Received prediction request for sequence length {len(request.sequence)}")
        
        # Run prediction
        result = await service.predict_structure(request.sequence)
        
        if result["status"] == "success":
            return ProteinPredictionResponse(
                status="success",
                pdb_content=result["pdb_content"],
                sequence_length=result["sequence_length"],
                message=result["message"]
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check(
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Check the health of the protein folding service"""
    try:
        health = await service.health_check()
        return HealthResponse(
            status=health["status"],
            model_loaded=health["model_loaded"],
            device=health["device"],
            message=health.get("message"),
            error=health.get("error")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            error=str(e)
        )

@app.get("/models")
async def get_available_models():
    """Get information about available protein folding models"""
    return {
        "models": [
            {
                "name": "Boltz",
                "type": "boltz",
                "description": "Boltz protein structure prediction (no MSA by default)",
                "supported_sequences": "10-2000 amino acids",
                "estimated_time": "3-10 minutes"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server for OpenBiologist Protein Folding API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
