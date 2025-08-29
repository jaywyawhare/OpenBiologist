from src.tools.job_finder import job_finder, JobFinderDescription
from src.tools.image_processor import make_img_black_and_white, MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION
from src.utils.models import RichToolDescription
from src.backend.client import ProteinFoldingClient
from src.utils.pdb_visualizer import pdb_visualizer
from typing import Annotated
from pydantic import Field
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, TextContent, ImageContent
import logging
import base64

# Protein Folding Tool Descriptions
ProteinFoldingDescription = RichToolDescription(
    description="Predict protein 3D structure from amino acid sequence using Boltz.",
    use_when="Use to predict protein 3D structure from amino acid sequence using Boltz backend.",
    side_effects="Generates protein structure prediction and returns PDB content directly.",
)

ProteinFoldingHealthDescription = RichToolDescription(
    description="Check the health and status of the protein folding service.",
    use_when="Use to verify the service is running and check model availability.",
    side_effects="No side effects - read-only operation.",
)

ProteinFoldingModelsDescription = RichToolDescription(
    description="Get information about available protein folding models.",
    use_when="Use to see what models are available and their capabilities.",
    side_effects="No side effects - read-only operation.",
)

ProteinVisualizationDescription = RichToolDescription(
    description="Generate 3D molecular structure images from PDB content with different visualization styles.",
    use_when="Use to create custom visualizations of protein structures with different styles and color schemes.",
    side_effects="Generates PNG images for molecular visualization.",
)

def register_tools(mcp):
    """Register all MCP tools with the provided MCP server instance."""

    @mcp.tool(description=JobFinderDescription.model_dump_json())
    async def job_finder_tool(
        user_goal: str,
        job_description: str | None = None,
        job_url: str | None = None,
        raw: bool = False,
    ) -> str:
        return await job_finder(user_goal, job_description, job_url, raw)

    @mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
    async def make_img_black_and_white_tool(
        puch_image_data: str = None,
    ):
        return await make_img_black_and_white(puch_image_data)



    # Protein Folding Tools
    @mcp.tool(description=ProteinFoldingDescription.model_dump_json())
    async def predict_protein_structure_tool(
        sequence: Annotated[str, Field(description="Protein amino acid sequence (10-2000 amino acids)")],
    ) -> list[TextContent | ImageContent]:
        """Predict protein structure directly from sequence."""
        logger.info(f"predict_protein_structure_tool called with sequence length: {len(sequence)}")
        try:
            async with ProteinFoldingClient() as client:
                result = await client.predict_structure(sequence)
                
                if result['status'] == 'success':
                    # Generate 3D structure image
                    image_bytes = pdb_visualizer.create_structure_image(
                        result['pdb_content'],
                        width=800,
                        height=600,
                        style="cartoon",
                        color_scheme="chain"
                    )
                    
                    if image_bytes:
                        result_text = f"""
**🧬 Protein Structure Prediction Completed!**

📋 **Prediction Details:**
• Sequence Length: {result['sequence_length']} amino acids
• Status: **{result['status'].upper()}**
• Model: Boltz

📊 **Results:**
• PDB Content Length: {len(result.get('pdb_content', ''))} characters
• Structure: Ready for analysis
• 3D Visualization: Generated below

💾 **PDB Content Preview:**
```
{result.get('pdb_content', '')[:300]}{'...' if len(result.get('pdb_content', '')) > 300 else ''}
```

🎉 **Success:** Protein structure has been predicted and visualized!

🔬 **Next Steps:**
• View the 3D structure image above
• Copy the PDB content to a .pdb file
• Open in molecular visualization software (PyMOL, VMD, ChimeraX)
• Analyze the 3D structure
• Compare with known structures
"""
                        
                        # Create response with both text and image
                        return [
                            TextContent(type="text", text=result_text.strip()),
                            ImageContent(type="image", mimeType="image/png", data=base64.b64encode(image_bytes).decode('utf-8'))
                        ]
                    else:
                        # Fallback to text-only if image generation fails
                        result_text = f"""
**🧬 Protein Structure Prediction Completed!**

📋 **Prediction Details:**
• Sequence Length: {result['sequence_length']} amino acids
• Status: **{result['status'].upper()}**
• Model: Boltz

📊 **Results:**
• PDB Content Length: {len(result.get('pdb_content', ''))} characters
• Structure: Ready for analysis

💾 **PDB Content:**
```
{result.get('pdb_content', '')[:500]}{'...' if len(result.get('pdb_content', '')) > 500 else ''}
```

🎉 **Success:** Protein structure has been predicted!

🔬 **Next Steps:**
• Copy the PDB content to a .pdb file
• Open in molecular visualization software (PyMOL, VMD, ChimeraX)
• Analyze the 3D structure
• Compare with known structures
"""
                        return [TextContent(type="text", text=result_text.strip())]
                    
                else:
                    result_text = f"""
**❌ Protein Structure Prediction Failed**

🚨 **Error:** {result.get('error', 'Unknown error')}

💡 **Troubleshooting:**
• Check if the sequence is valid (10-2000 amino acids)
• Ensure only standard amino acids are used
• Verify the service is running
"""
                    return [TextContent(type="text", text=result_text.strip())]
                
        except Exception as e:
            logger.error(f"Error in predict_protein_structure_tool: {e}")
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Error in protein structure prediction: {str(e)}")
            )

    @mcp.tool(description=ProteinFoldingHealthDescription.model_dump_json())
    async def check_protein_folding_health_tool() -> list[TextContent]:
        """Check the health of the protein folding service."""
        logger.info("check_protein_folding_health_tool called")
        try:
            async with ProteinFoldingClient() as client:
                health = await client.health_check()
                
                status_emoji = "✅" if health['status'] == 'healthy' else "❌"
                
                result_text = f"""
**🏥 Protein Folding Service Health Check**

{status_emoji} **Status:** {health['status'].upper()}
🤖 **Model Loaded:** {'Yes' if health.get('model_loaded') else 'No'}
💻 **Device:** {health.get('device', 'Unknown')}
"""
                
                if health.get('error'):
                    result_text += f"\n🚨 **Error:** {health['error']}"
                
                if health['status'] == 'healthy':
                    result_text += f"""

✅ **Service is healthy and ready for protein folding predictions!**

🧬 **Available Models:** ESMFold (ESM-2 based protein structure prediction)
⏱️ **Typical Prediction Time:** 3-10 minutes
📏 **Sequence Length Support:** 10-2000 amino acids
"""
                else:
                    result_text += f"""

⚠️ **Service has issues and may not function properly**
💡 **Check:** Ensure the backend is running and models are loaded
"""
                
                return [TextContent(type="text", text=result_text.strip())]
                
        except Exception as e:
            logger.error(f"Error in check_protein_folding_health_tool: {e}")
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Error checking protein folding health: {str(e)}")
            )

    @mcp.tool(description=ProteinFoldingModelsDescription.model_dump_json())
    async def get_protein_folding_models_tool() -> list[TextContent]:
        """Get information about available protein folding models."""
        logger.info("get_protein_folding_models_tool called")
        try:
            async with ProteinFoldingClient() as client:
                models = await client.get_available_models()
                
                result_parts = ["**🧬 Available Protein Folding Models**\n"]
                
                for model in models.get('models', []):
                    result_parts.append(f"""
📊 **{model['name']}**
• **Type:** {model['type']}
• **Description:** {model['description']}
• **Sequence Support:** {model['supported_sequences']}
• **Estimated Time:** {model['estimated_time']}
""")
                
                result_parts.append("""
💡 **Usage:**
• Predict structures with `predict_protein_structure(sequence)`
• Generate visualizations with `generate_protein_visualization(pdb_content, style, color_scheme)`
• Check service health with `check_protein_folding_health()`

🔬 **Model Details:**
• ESMFold uses ESM-2 language models for protein structure prediction
• Predicts 3D coordinates for all atoms in the protein
• Outputs standard PDB format for visualization and analysis
• Includes automatic 3D image generation for immediate visualization
""")
                
                return [TextContent(type="text", text="".join(result_parts).strip())]
                
        except Exception as e:
            logger.error(f"Error in get_protein_folding_models_tool: {e}")
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Error retrieving model information: {str(e)}")
            )

    @mcp.tool(description=ProteinVisualizationDescription.model_dump_json())
    async def generate_protein_visualization_tool(
        pdb_content: Annotated[str, Field(description="PDB content string")],
        style: Annotated[str, Field(description="Visualization style", default="cartoon")] = "cartoon",
        color_scheme: Annotated[str, Field(description="Color scheme", default="chain")] = "chain",
        width: Annotated[int, Field(description="Image width in pixels", default=800, ge=400, le=1200)] = 800,
        height: Annotated[int, Field(description="Image height in pixels", default=600, ge=300, le=900)] = 600,
        multi_view: Annotated[bool, Field(description="Generate multi-view image", default=False)] = False,
    ) -> list[TextContent | ImageContent]:
        """Generate 3D molecular structure visualization from PDB content."""
        logger.info(f"generate_protein_visualization_tool called with style: {style}, color: {color_scheme}, size: {width}x{height}")
        try:
            # Generate image based on parameters
            if multi_view:
                image_bytes = pdb_visualizer.create_multiple_views(
                    pdb_content,
                    width=width//2,
                    height=height//2
                )
                image_type = "Multi-view"
            else:
                image_bytes = pdb_visualizer.create_structure_image(
                    pdb_content,
                    width=width,
                    height=height,
                    style=style,
                    color_scheme=color_scheme
                )
                image_type = "Single view"
            
            if image_bytes:
                result_text = f"""
**🎨 Protein Structure Visualization Generated!**

📋 **Visualization Details:**
• Type: {image_type}
• Style: {style}
• Color Scheme: {color_scheme}
• Dimensions: {width}x{height} pixels
• Format: PNG

🔬 **Visualization Features:**
• 3D molecular structure representation
• Interactive-style rendering
• Professional molecular graphics
• Ready for analysis and presentation

💡 **Usage:**
• View the structure image above
• Save the image for reports/presentations
• Use for structural analysis
• Share with colleagues
"""
                
                return [
                    TextContent(type="text", text=result_text.strip()),
                    ImageContent(type="image", mimeType="image/png", data=base64.b64encode(image_bytes).decode('utf-8'))
                ]
            else:
                result_text = f"""
**❌ Visualization Generation Failed**

🚨 **Error:** Failed to generate structure image

💡 **Troubleshooting:**
• Check if PDB content is valid
• Ensure PDB content is not empty
• Try different style/color combinations
• Verify the service is running
"""
                return [TextContent(type="text", text=result_text.strip())]
                
        except Exception as e:
            logger.error(f"Error in generate_protein_visualization_tool: {e}")
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Error generating protein visualization: {str(e)}")
            )
