"""
Local Protein Predictor Service
Supports Boltz backend for local protein structure prediction.
"""

import asyncio
from typing import Dict, Any
import logging
import torch
import os

from src.services.boltz_adapter import BoltzPredictor
from src.utils.fasta import parse_and_fix_fasta, format_fasta, format_boltz_fasta
from typing import List


logger = logging.getLogger(__name__)

class LocalProteinStructureService:
    """Local protein structure prediction service using Boltz"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Boltz predictor"""
        try:
            logger.info("Initializing Boltz predictor...")
            self.model = BoltzPredictor()
            logger.info(
                "Boltz predictor initialized on %s",
                "GPU" if torch.cuda.is_available() else "CPU",
            )
            logger.info("Boltz predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Boltz predictor: {e}")
            self.model = None
    
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict protein structure directly from sequence using Boltz"""
        try:
            # Validate sequence
            validation = await self.validate_sequence(sequence)
            if not validation["valid"]:
                return {
                    "status": "error",
                    "error": validation["error"]
                }
            
            if self.model is None:
                return {
                    "status": "error",
                    "error": "Boltz predictor not available"
                }
            
            logger.info(f"Predicting structure (Boltz) for sequence length {len(sequence)}")
            
            # Fix/normalize FASTA: accept raw sequence or FASTA, sanitize chars
            header, fixed_seq, warns = parse_and_fix_fasta(sequence)
            if not fixed_seq:
                return {"status": "error", "error": "Invalid or empty sequence after sanitization"}

            # Use Boltz FASTA format for better compatibility
            fixed_fasta = format_boltz_fasta(header, fixed_seq)

            # Run prediction on cleaned sequence
            pdb_content = await self._run_prediction(fixed_seq)
            
            return {
                "status": "success",
                "pdb_content": pdb_content,
                "sequence_length": len(sequence),
                "message": "Structure predicted successfully (Boltz)"
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "status": "error",
                "error": f"Prediction failed: {str(e)}"
            }
    
    async def _run_prediction(self, sequence: str) -> str:
        """Run protein structure prediction using Boltz"""
        try:
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
        """Predict protein structure using Boltz"""
        try:
            self.model.eval()
            return self.model.infer_pdb(sequence)
        except Exception as e:
            logger.error(f"Boltz prediction failed: {e}")
            raise
    
    async def predict_structure_fasta(self, fasta_content: str, use_msa_server: bool = True) -> Dict[str, Any]:
        """Predict protein structure from FASTA format content"""
        try:
            # Parse FASTA content
            header, sequence, warnings = parse_and_fix_fasta(fasta_content)
            if not sequence:
                return {"status": "error", "error": "Invalid or empty sequence after parsing FASTA"}
            
            # Validate sequence
            validation = await self.validate_sequence(sequence)
            if not validation["valid"]:
                return {
                    "status": "error",
                    "error": validation["error"]
                }
            
            if self.model is None:
                return {
                    "status": "error",
                    "error": "Boltz predictor not available"
                }
            
            logger.info(f"Predicting structure from FASTA (Boltz) for sequence length {len(sequence)}")
            
            # Create Boltz-compatible FASTA format
            boltz_fasta = format_boltz_fasta(header, sequence)
            logger.debug(f"Generated Boltz FASTA: {boltz_fasta}")
            
            # Run prediction
            pdb_content = await self._run_prediction(sequence)
            
            return {
                "status": "success",
                "pdb_content": pdb_content,
                "sequence_length": len(sequence),
                "message": "Structure predicted successfully from FASTA (Boltz)",
                "chains_processed": 1,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"FASTA prediction failed: {e}")
            return {
                "status": "error",
                "error": f"FASTA prediction failed: {str(e)}"
            }

    async def predict_structure_yaml(self, chains: List[Dict[str, Any]], use_msa_server: bool = True) -> Dict[str, Any]:
        """Predict protein structure from YAML chain definitions"""
        try:
            if not chains:
                return {"status": "error", "error": "No chains provided"}
            
            # For now, we'll process the first protein chain
            # TODO: Implement multi-chain and ligand support
            protein_chains = [c for c in chains if c.get("entity_type") == "protein"]
            if not protein_chains:
                return {"status": "error", "error": "No protein chains found in YAML input"}
            
            # Use the first protein chain for now
            chain = protein_chains[0]
            sequence = chain.get("sequence", "")
            
            if not sequence:
                return {"status": "error", "error": "Empty sequence in protein chain"}
            
            # Validate sequence
            validation = await self.validate_sequence(sequence)
            if not validation["valid"]:
                return {
                    "status": "error",
                    "error": validation["error"]
                }
            
            if self.model is None:
                return {
                    "status": "error",
                    "error": "Boltz predictor not available"
                }
            
            logger.info(f"Predicting structure from YAML (Boltz) for sequence length {len(sequence)}")
            
            # Run prediction
            pdb_content = await self._run_prediction(sequence)
            
            warnings = []
            if len(protein_chains) > 1:
                warnings.append(f"Only processed first protein chain (A), {len(protein_chains)-1} additional chains ignored")
            
            return {
                "status": "success",
                "pdb_content": pdb_content,
                "sequence_length": len(sequence),
                "message": "Structure predicted successfully from YAML (Boltz)",
                "chains_processed": len(chains),
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"YAML prediction failed: {e}")
            return {
                "status": "error",
                "error": f"YAML prediction failed: {str(e)}"
            }

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
                    "error": "Boltz predictor not initialized"
                }
            
            device_info = "GPU" if torch.cuda.is_available() else "CPU"
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "device": device_info,
                "message": "Ready for protein structure prediction (Boltz)"
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
