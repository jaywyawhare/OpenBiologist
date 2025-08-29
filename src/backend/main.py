"""
FastAPI Backend for OpenBiologist Protein Folding Service
Provides REST API endpoints for protein structure prediction
"""

import logging
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

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
class ProteinChain(BaseModel):
    chain_id: str = Field(..., description="Unique identifier for the chain")
    entity_type: str = Field(..., description="Type: protein, dna, rna, smiles, ccd")
    sequence: str = Field(..., description="Sequence data (amino acids, nucleotides, SMILES, or CCD code)")
    msa_path: Optional[str] = Field(None, description="Path to MSA file (.a3m or .csv) for proteins")
    
    @validator('entity_type')
    def validate_entity_type(cls, v):
        valid_types = ['protein', 'dna', 'rna', 'smiles', 'ccd']
        if v not in valid_types:
            raise ValueError(f'entity_type must be one of {valid_types}')
        return v

class YAMLInputRequest(BaseModel):
    chains: List[ProteinChain] = Field(..., description="List of protein chains and ligands")
    use_msa_server: bool = Field(True, description="Use MSA server for proteins without custom MSA")
    
    @validator('chains')
    def validate_chains(cls, v):
        if not v:
            raise ValueError('At least one chain must be provided')
        return v

class FASTAInputRequest(BaseModel):
    fasta_content: str = Field(..., description="FASTA format content with chain headers")
    use_msa_server: bool = Field(True, description="Use MSA server for proteins without custom MSA")

class SimpleSequenceRequest(BaseModel):
    sequence: str = Field(..., description="Simple protein amino acid sequence", min_length=10, max_length=2000)

class ProteinPredictionResponse(BaseModel):
    status: str
    pdb_content: str
    sequence_length: int
    message: str
    input_format: str
    chains_processed: int
    warnings: Optional[List[str]] = None

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
    request: Union[YAMLInputRequest, FASTAInputRequest, SimpleSequenceRequest] = None,
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Predict protein structure from various input formats"""
    try:
        if isinstance(request, SimpleSequenceRequest):
            # Simple sequence input (backward compatibility)
            logger.info(f"Received simple sequence request for length {len(request.sequence)}")
            result = await service.predict_structure(request.sequence)
            input_format = "simple_sequence"
            chains_processed = 1
            
        elif isinstance(request, FASTAInputRequest):
            # FASTA format input
            logger.info(f"Received FASTA format request")
            result = await service.predict_structure_fasta(request.fasta_content, request.use_msa_server)
            input_format = "fasta"
            chains_processed = result.get("chains_processed", 1)
            
        elif isinstance(request, YAMLInputRequest):
            # YAML format input
            logger.info(f"Received YAML format request with {len(request.chains)} chains")
            result = await service.predict_structure_yaml(request.chains, request.use_msa_server)
            input_format = "yaml"
            chains_processed = len(request.chains)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
        
        if result["status"] == "success":
            return ProteinPredictionResponse(
                status="success",
                pdb_content=result["pdb_content"],
                sequence_length=result["sequence_length"],
                message=result["message"],
                input_format=input_format,
                chains_processed=chains_processed,
                warnings=result.get("warnings")
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

@app.post("/predict/fasta")
async def predict_from_fasta_file(
    fasta_file: UploadFile = File(...),
    use_msa_server: bool = Form(True),
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Predict protein structure from uploaded FASTA file"""
    try:
        if not fasta_file.filename.endswith(('.fasta', '.fa', '.txt')):
            raise HTTPException(status_code=400, detail="File must be .fasta, .fa, or .txt")
        
        content = await fasta_file.read()
        fasta_content = content.decode('utf-8')
        
        logger.info(f"Received FASTA file upload: {fasta_file.filename}")
        result = await service.predict_structure_fasta(fasta_content, use_msa_server)
        
        if result["status"] == "success":
            return ProteinPredictionResponse(
                status="success",
                pdb_content=result["pdb_content"],
                sequence_length=result["sequence_length"],
                message=result["message"],
                input_format="fasta_file",
                chains_processed=result.get("chains_processed", 1),
                warnings=result.get("warnings")
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"FASTA file prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/yaml")
async def predict_from_yaml_file(
    yaml_file: UploadFile = File(...),
    use_msa_server: bool = Form(True),
    service: LocalProteinStructureService = Depends(get_healthy_service)
):
    """Predict protein structure from uploaded YAML file"""
    try:
        if not yaml_file.filename.endswith(('.yaml', '.yml')):
            raise HTTPException(status_code=400, detail="File must be .yaml or .yml")
        
        content = await yaml_file.read()
        yaml_content = content.decode('utf-8')
        
        # Parse YAML
        try:
            data = yaml.safe_load(yaml_content)
            chains_data = data.get('chains', [])
            chains = [ProteinChain(**chain) for chain in chains_data]
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML format: {e}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid chain data: {e}")
        
        logger.info(f"Received YAML file upload: {yaml_file.filename} with {len(chains)} chains")
        result = await service.predict_structure_yaml(chains, use_msa_server)
        
        if result["status"] == "success":
            return ProteinPredictionResponse(
                status="success",
                pdb_content=result["pdb_content"],
                sequence_length=result["sequence_length"],
                message=result["message"],
                input_format="yaml_file",
                chains_processed=len(chains),
                warnings=result.get("warnings")
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"YAML file prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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
                "supported_formats": ["simple_sequence", "fasta", "yaml"],
                "estimated_time": "3-10 minutes"
            }
        ]
    }

@app.get("/formats")
async def get_supported_formats():
    """Get information about supported input formats"""
    return {
        "formats": {
            "simple_sequence": {
                "description": "Plain amino acid sequence",
                "example": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"
            },
            "fasta": {
                "description": "FASTA format with chain headers",
                "example": ">A|protein|empty\nMVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"
            },
            "yaml": {
                "description": "YAML format with structured chain definitions",
                "example": {
                    "chains": [
                        {
                            "chain_id": "A",
                            "entity_type": "protein",
                            "sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
                            "msa_path": null
                        }
                    ],
                    "use_msa_server": true
                }
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server for OpenBiologist Protein Folding API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
