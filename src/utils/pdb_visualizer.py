"""
PDB Visualization Utility
Generates 3D molecular structure images from PDB content
"""

import py3Dmol
import base64
import io
from PIL import Image
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class PDBVisualizer:
    """Generate 3D molecular structure images from PDB content"""
    
    def __init__(self):
        self.viewer = None
    
    def create_structure_image(
        self, 
        pdb_content: str, 
        width: int = 800, 
        height: int = 600,
        style: str = "cartoon",
        color_scheme: str = "chain",
        background_color: str = "white"
    ) -> Optional[bytes]:
        """
        Create a 3D molecular structure image from PDB content
        
        Args:
            pdb_content: PDB format string
            width: Image width in pixels
            height: Image height in pixels
            style: Molecular representation style ('cartoon', 'stick', 'sphere', 'line')
            color_scheme: Coloring scheme ('chain', 'residue', 'element', 'rainbow')
            background_color: Background color
            
        Returns:
            PNG image bytes or None if failed
        """
        try:
            # Create py3Dmol viewer
            viewer = py3Dmol.view(width=width, height=height)
            
            # Set background color
            viewer.setBackgroundColor(background_color)
            
            # Add PDB structure
            viewer.addModel(pdb_content, "pdb")
            
            # Apply molecular style
            if style == "cartoon":
                viewer.setStyle({}, {"cartoon": {"color": "spectrum"}})
            elif style == "stick":
                viewer.setStyle({}, {"stick": {}})
            elif style == "sphere":
                viewer.setStyle({}, {"sphere": {"radius": 0.5}})
            elif style == "line":
                viewer.setStyle({}, {"line": {}})
            
            # Apply color scheme
            if color_scheme == "chain":
                viewer.setStyle({}, {"cartoon": {"colorscheme": "chain"}})
            elif color_scheme == "residue":
                viewer.setStyle({}, {"cartoon": {"colorscheme": "amino"}})
            elif color_scheme == "element":
                viewer.setStyle({}, {"stick": {"colorscheme": "element"}})
            elif color_scheme == "rainbow":
                viewer.setStyle({}, {"cartoon": {"colorscheme": "rainbow"}})
            
            # Zoom to fit structure
            viewer.zoomTo()
            
            # Generate image
            image_data = viewer.png()
            
            # Convert to PIL Image for potential post-processing
            image = Image.open(io.BytesIO(image_data))
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            
            logger.info(f"Successfully generated structure image: {width}x{height}")
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate structure image: {e}")
            return None
    
    def create_multiple_views(
        self, 
        pdb_content: str, 
        width: int = 400, 
        height: int = 300
    ) -> Optional[bytes]:
        """
        Create a multi-view image showing different perspectives
        
        Args:
            pdb_content: PDB format string
            width: Individual view width
            height: Individual view height
            
        Returns:
            Combined PNG image bytes or None if failed
        """
        try:
            # Create a larger canvas for multiple views
            total_width = width * 2
            total_height = height * 2
            
            # Create main viewer
            viewer = py3Dmol.view(width=total_width, height=total_height)
            viewer.setBackgroundColor("white")
            
            # Add PDB structure
            viewer.addModel(pdb_content, "pdb")
            
            # Create different views
            views = [
                {"style": "cartoon", "color": "spectrum", "x": 0, "y": 0},
                {"style": "stick", "color": "element", "x": width, "y": 0},
                {"style": "sphere", "color": "rainbow", "x": 0, "y": height},
                {"style": "line", "color": "chain", "x": width, "y": height}
            ]
            
            for i, view_config in enumerate(views):
                # Create sub-viewer for each perspective
                sub_viewer = py3Dmol.view(width=width, height=height)
                sub_viewer.setBackgroundColor("white")
                sub_viewer.addModel(pdb_content, "pdb")
                
                # Apply style
                if view_config["style"] == "cartoon":
                    sub_viewer.setStyle({}, {"cartoon": {"color": view_config["color"]}})
                elif view_config["style"] == "stick":
                    sub_viewer.setStyle({}, {"stick": {"colorscheme": view_config["color"]}})
                elif view_config["style"] == "sphere":
                    sub_viewer.setStyle({}, {"sphere": {"radius": 0.5, "colorscheme": view_config["color"]}})
                elif view_config["style"] == "line":
                    sub_viewer.setStyle({}, {"line": {"colorscheme": view_config["color"]}})
                
                sub_viewer.zoomTo()
                
                # Position sub-view in main canvas
                viewer.addSubView(sub_viewer, view_config["x"], view_config["y"])
            
            # Generate combined image
            image_data = viewer.png()
            
            # Convert to PIL Image and back to bytes
            image = Image.open(io.BytesIO(image_data))
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            
            logger.info(f"Successfully generated multi-view structure image: {total_width}x{total_height}")
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate multi-view structure image: {e}")
            return None

# Global instance
pdb_visualizer = PDBVisualizer()
