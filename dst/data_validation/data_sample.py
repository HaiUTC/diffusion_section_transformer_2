"""
Data Sample Definition

Core data structure for representing preprocessed samples in the pipeline.
Pure data container without any file I/O operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch


@dataclass
class DataSample:
    """
    Represents a single data sample with all required components.
    
    This is a pure data container that holds both raw data and processed outputs
    without any file operations.
    """
    sample_id: str
    screenshot_path: str
    html_structure: Dict[str, Any]
    section_layout: Dict[str, Any] 
    used_elements: List[str]
    category: str
    image_tensor: Optional[torch.Tensor] = None
    image_patches: Optional[torch.Tensor] = None
    processing_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate data sample after initialization."""
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not self.screenshot_path:
            raise ValueError("screenshot_path cannot be empty")
        if not self.html_structure:
            raise ValueError("html_structure cannot be empty")
        if not self.section_layout:
            raise ValueError("section_layout cannot be empty")
        if not self.used_elements:
            raise ValueError("used_elements cannot be empty")
        if not self.category:
            raise ValueError("category cannot be empty")
    
    def get_element_count(self) -> int:
        """Get the number of semantic elements used."""
        return len(self.used_elements)
    
    def has_image_tensor(self) -> bool:
        """Check if image tensor has been processed."""
        return self.image_tensor is not None
    
    def has_image_patches(self) -> bool:
        """Check if image patches have been extracted."""
        return self.image_patches is not None
    
    def get_tensor_shape(self) -> Optional[tuple]:
        """Get the shape of the image tensor if available."""
        return tuple(self.image_tensor.shape) if self.has_image_tensor() else None
    
    def get_patches_shape(self) -> Optional[tuple]:
        """Get the shape of the image patches if available."""
        return tuple(self.image_patches.shape) if self.has_image_patches() else None 