from typing import Annotated
from pydantic import Field
from src.utils.models import RichToolDescription
from src.services.protein_predictor import protein_service

ProteinStructurePredictionDescription = RichToolDescription(
    description="Predict 3D protein structure from amino acid sequence using advanced AI models",
    use_when="Use this when user provides a protein sequence and wants to predict its 3D structure",
    side_effects="Submits prediction job to external service and may take several minutes to complete",
)

async def predict_protein_structure(
    sequence: Annotated[str, Field(description="Protein amino acid sequence (e.g., MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH)")],
    job_name: Annotated[str, Field(description="Name for this prediction job")] = "Protein Structure Prediction",
    wait_for_completion: Annotated[bool, Field(description="Wait for prediction to complete (may take 5-10 minutes)")] = False,
) -> str:
    """
    Predict 3D protein structure from amino acid sequence.
    
    This tool uses state-of-the-art AI models to predict the 3D structure of proteins
    from their amino acid sequences. The prediction includes:
    - 3D coordinates in PDB format
    - Confidence scores (pLDDT)
    - Structural analysis
    
    Args:
        sequence: Protein amino acid sequence (20 standard amino acids)
        job_name: Human-readable name for the prediction
        wait_for_completion: Whether to wait for results (default: False for quick response)
    
    Returns:
        Prediction results or job status information
    """
    
    # Validate sequence
    validation = await protein_service.validate_sequence(sequence)
    if not validation["valid"]:
        return f"❌ **Sequence Validation Error**: {validation['error']}"
    
    # Submit prediction
    response = await protein_service.submit_prediction(sequence, job_name)
    
    if response.status == "error":
        return f"❌ **Prediction Error**: {response.message}"
    
    # Format initial response
    result = f"""🧬 **Protein Structure Prediction Submitted**

**Job ID**: `{response.job_id}`
**Job Name**: {job_name}
**Sequence Length**: {validation['length']} amino acids
**Status**: {response.status}
**Estimated Time**: {response.estimated_time} seconds

**Sequence Analysis**:
- **Length**: {validation['length']} residues
- **Amino Acid Composition**: {', '.join([f'{aa}: {count}' for aa, count in validation['amino_acid_count'].items()][:5])}...
"""
    
    if not wait_for_completion:
        result += f"""

⏳ **Next Steps**:
- Use the job ID `{response.job_id}` to check status later
- Prediction typically takes 5-10 minutes
- You can check status with: "Check protein prediction status {response.job_id}"
"""
        return result
    
    # Wait for completion
    result += "\n⏳ **Waiting for prediction to complete...**\n"
    
    status = await protein_service.wait_for_completion(response.job_id)
    
    if status.status == "successful":
        result += f"""
✅ **Prediction Complete!**

**Job ID**: `{status.job_id}`
**Confidence Score**: {status.confidence_score:.2f} (pLDDT)

**Results**:
- 3D structure predicted successfully
- PDB file generated with {len(status.pdb_content.split('\\n'))} lines
- Structure ready for visualization

**PDB Preview** (first 10 lines):
```
{status.pdb_content.split('\\n')[:10] if status.pdb_content else 'No PDB content available'}
```

💡 **Next Steps**:
- Use the PDB content for 3D visualization
- Analyze structure with molecular modeling tools
- Compare with known structures
"""
    elif status.status == "timeout":
        result += f"""
⏰ **Prediction Timed Out**

**Job ID**: `{status.job_id}`
**Status**: {status.status}

The prediction is still running but took longer than expected.
You can check the status later with the job ID.
"""
    else:
        result += f"""
❌ **Prediction Failed**

**Job ID**: `{status.job_id}`
**Status**: {status.status}
**Error**: {status.error_message or 'Unknown error'}
"""
    
    return result

async def check_protein_prediction_status(
    job_id: Annotated[str, Field(description="Job ID from protein structure prediction")],
) -> str:
    """
    Check the status of a protein structure prediction job.
    
    Args:
        job_id: The job ID returned from a previous prediction request
    
    Returns:
        Current status and results if available
    """
    
    status = await protein_service.check_status(job_id)
    
    if status.status == "pending":
        return f"""⏳ **Prediction Status: Pending**

**Job ID**: `{job_id}`
**Status**: {status.status}

The prediction is queued and waiting to be processed.
Please check again in a few minutes."""
    
    elif status.status == "processing":
        return f"""🔄 **Prediction Status: Processing**

**Job ID**: `{job_id}`
**Status**: {status.status}

The prediction is currently running. This typically takes 5-10 minutes.
Please check again in a few minutes."""
    
    elif status.status == "successful":
        return f"""✅ **Prediction Complete!**

**Job ID**: `{job_id}`
**Status**: {status.status}
**Confidence Score**: {status.confidence_score:.2f} (pLDDT)

**Results Available**:
- 3D structure predicted successfully
- PDB file generated
- Structure ready for analysis

**PDB Content** (first 20 lines):
```
{status.pdb_content.split('\\n')[:20] if status.pdb_content else 'No PDB content available'}
```

💡 **Analysis Tips**:
- High pLDDT scores (>70) indicate reliable predictions
- Use molecular visualization software to view the structure
- Compare with experimental structures if available"""
    
    elif status.status == "crashed":
        return f"""❌ **Prediction Failed**

**Job ID**: `{job_id}`
**Status**: {status.status}
**Error**: {status.error_message or 'Unknown error occurred'}

The prediction encountered an error during processing.
Please try again with a different sequence or contact support."""
    
    else:
        return f"""❓ **Unknown Status**

**Job ID**: `{job_id}`
**Status**: {status.status}
**Error**: {status.error_message or 'Unable to determine status'}

Please try again or contact support if the issue persists.""" 