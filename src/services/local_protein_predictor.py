"""
Local Protein Predictor Service
Uses simplified ESMFold implementation for local protein structure prediction
"""

import asyncio
from typing import Dict, Any
import logging
import torch

from src.fold_models import ESMFold, ESMFoldConfig

logger = logging.getLogger(__name__)

class LocalProteinStructureService:
    """Local protein structure prediction service using ESMFold"""
    
    def __init__(self):
        self.model = None
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
    
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict protein structure directly from sequence"""
        try:
            # Validate sequence
            validation = await self.validate_sequence(sequence)
            if not validation["valid"]:
                return {
                    "status": "error",
                    "error": validation["error"]
                }
            
            # Check if model is available
            if self.model is None:
                return {
                    "status": "error",
                    "error": "ESMFold model not available"
                }
            
            logger.info(f"Predicting structure for sequence length {len(sequence)}")
            
            # Run prediction
            pdb_content = await self._run_prediction(sequence)
            
            return {
                "status": "success",
                "pdb_content": pdb_content,
                "sequence_length": len(sequence),
                "message": "Structure predicted successfully"
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "status": "error",
                "error": f"Prediction failed: {str(e)}"
            }
    
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
                    "error": "ESMFold model not initialized"
                }
            
            # Check if model is on GPU
            device_info = "GPU" if self.model.device.type == "cuda" else "CPU"
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "device": device_info,
                "message": "Ready for protein structure prediction"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "device": "unknown",
                "error": f"Health check failed: {str(e)}"
            }

# Global service instance
local_protein_service = LocalProteinStructureService()
