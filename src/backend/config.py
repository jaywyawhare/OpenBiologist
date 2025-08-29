"""
Configuration for the OpenBiologist Protein Folding Backend
"""

import os
from typing import Optional
from pydantic import BaseSettings

class BackendConfig(BaseSettings):
    """Configuration for the protein folding backend"""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # API configuration
    title: str = "OpenBiologist Protein Folding API"
    description: str = "REST API for protein structure prediction using Boltz"
    version: str = "1.0.0"
    
    # CORS configuration
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Protein folding service configuration
    max_sequence_length: int = 2000
    min_sequence_length: int = 10
    max_wait_time: int = 600  # 10 minutes
    results_directory: str = "protein_folding_results"
    
    # Model configuration
    default_model_type: str = "boltz"
    gpu_enabled: bool = True
    model_download_path: Optional[str] = None
    
    class Config:
        env_prefix = "OPENBIOLOGIST_"
        case_sensitive = False

# Global configuration instance
config = BackendConfig()
