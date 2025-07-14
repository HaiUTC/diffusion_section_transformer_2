"""
CSS Class Parser

Parses custom compressed CSS format into semantic components.
Handles responsive prefixes, pseudo-classes, attributes, values, and units.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CSSComponent:
    """Represents a parsed CSS class component."""
    device_context: Optional[str] = None  # t, m, d (tablet, mobile, desktop)
    pseudo_context: Optional[str] = None  # h, f, a (hover, focus, active)
    attribute: Optional[str] = None       # pt, fs, c, etc.
    value: Optional[str] = None          # 7, 18, auto, etc.
    unit: Optional[str] = None           # r, x, p (rem, px, percent)
    raw_class: str = ""                  # Original class string
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'device_context': self.device_context,
            'pseudo_context': self.pseudo_context,
            'attribute': self.attribute,
            'value': self.value,
            'unit': self.unit,
            'raw_class': self.raw_class
        }


class CSSClassParser:
    """
    Parses custom CSS class format into semantic components.
    
    Handles format: [device:]?[pseudo:]?attribute[_value][unit]?
    Examples: pt_7r, t:fs_18x, h:c_blue, m:h:pt_auto
    """
    
    def __init__(self):
        # Device context mappings - Updated to match test expectations
        self.device_contexts = {
            't': 'tablet',
            'm': 'mobile', 
            'md': 'medium',
            'lg': 'large',
            'xl': 'extra-large',
            'sm': 'small'
        }
        
        # Pseudo-class context mappings - Updated to match test expectations
        self.pseudo_contexts = {
            'h': 'hover',
            'f': 'focus',
            'a': 'active',
            'v': 'visited',
            'd': 'disabled'
        }
        
        # Unit mappings
        self.units = {
          "x": "px",
          "r": "rem",
          "m": "em",
          "p": "%",
          "w": "vw",
          "h": "vh",
          "n": "vmin",
          "xv": "vmax",
          "c": "ch",
          "e": "ex",
          "cm": "cm",
          "mm": "mm",
          "i": "in",
          "pt": "pt",
          "pc": "pc",
          "f": "fr",
          "d": "deg",
          "ra": "rad",
          "t": "turn",
          "s": "s",
          "ms": "ms",
          "l": "lh",
          "dp": "dpi",
          "dx": "dppx",
        }
        
        # Common CSS attributes
        self.attributes = {
          # Typography
            "c": "color",
            "fs": "font-size",
            "lh": "line-height",
            "ls": "letter-spacing",
            "fw": "font-weight",
            "ta": "text-align",
            "tt": "text-transform",
            "td": "text-decoration",
            "fst": "font-style",
            "ts": "text-shadow",
            "text": "text",
        
          # Background
            "bg": "background",
            "bc": "background-color",
            "bi": "background-image",
            "bp": "background-position",
            "br": "background-repeat",
            "bs": "background-size",

          # Spacing
            "p": "padding",
            "m": "margin",
            "pt": "padding-top",
            "pr": "padding-right",
            "pb": "padding-bottom",
            "pl": "padding-left",
            "mt": "margin-top",
            "mr": "margin-right",
            "mb": "margin-bottom",
            "ml": "margin-left",

          # Border
            "b": "border",
            "border": "border",
            "bs": "border-style",
            "bts": "border-top-style",
            "brs": "border-right-style",
            "bbs": "border-bottom-style",
            "bls": "border-left-style",
            "bc": "border-color",
            "btc": "border-top-color",
            "brc": "border-right-color",
            "bbc": "border-bottom-color",
            "blc": "border-left-color",
            "bw": "border-width",
            "btw": "border-top-width",
            "brw": "border-right-width",
            "bbw": "border-bottom-width",
            "blw": "border-left-width",
            "br": "border-radius",
            "btl": "border-top-left-radius",
            "btr": "border-top-right-radius",
            "bbr": "border-bottom-right-radius",
            "bbl": "border-bottom-left-radius",

          # Effects         
            "o": "opacity",
            "tr": "transition",
            "tf": "transform",
            "f": "filter",
            "bsd": "box-shadow",
            "ov": "overflow",
            
          # Dimensions
            "w": "width",
            "h": "height",
            "mw": "min-width",
            "xw": "max-width",
            "mh": "min-height",
            "xh": "max-height",
            
          # Display
            "d": "display",
            "fd": "flex-direction",
            "fw": "flex-wrap",
            "fg": "flex-grow",
            "fsr": "flex-shrink",
            "fb": "flex-basis",
            "jc": "justify-content",
            "ai": "align-items",
            "ac": "align-content",
            "as": "align-self",
            "g": "gap",
            "rg": "row-gap",
            "cg": "column-gap",
            "gg": "grid-gap",
            "gr": "grid-row-gap",
            "gc": "grid-column-gap",
            "gtc": "grid-template-columns",
            "gtr": "grid-template-rows",
            "ps": "position",
            "t": "top", 
            "r": "right",
            "b": "bottom",
            "l": "left",
            "i": "inset",
            "z": "z-index",
            "of": "object-fit",
            "ar": "aspect-ratio",
        }
        
        # Compile regex pattern for parsing
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient parsing."""
        # Pattern: [device:]?[pseudo:]?attribute[_value[unit]]?
        # Use non-capturing groups for alternatives, capturing groups for the parts we want
        device_pattern = f"(?:{'|'.join(self.device_contexts.keys())})"
        pseudo_pattern = f"(?:{'|'.join(self.pseudo_contexts.keys())})"
        attr_pattern = f"(?:{'|'.join(self.attributes.keys())})"
        
        # Main pattern with exactly 6 capturing groups:
        # 1. device context (optional)
        # 2. pseudo context (optional) 
        # 3. attribute
        # 4. numeric value (optional) - for cases like "18" in "fs_18x" or "4" in "t:pt_4"
        # 5. unit (optional) - for cases like "x" in "fs_18x"
        # 6. text value (optional) - for cases like "gradient" in "bg_gradient"
        # Pattern handles three cases:
        # - Numeric values with units: fs_18x, pt_7r, w_100p
        # - Numeric values without units: t:pt_4, z_10
        # - Text values without units: bg_gradient, m_auto, c_blue
        self.css_pattern = re.compile(
            rf"^(?:({device_pattern}):)?(?:({pseudo_pattern}):)?({attr_pattern})(?:_(?:([0-9\.\-#%]+)([a-zA-Z]+)?|([a-zA-Z][a-zA-Z0-9\.\-#%]*)))?$"
        )
        
        # Alternative pattern for special cases (values without underscores)
        # This handles cases like 'bg_gradient' which should be parsed as attr='bg', value='gradient'
        self.special_pattern = re.compile(rf'^({attr_pattern})(?:_([a-zA-Z0-9\.\-#%]+))?$')
    
    def parse_class(self, css_class: str) -> CSSComponent:
        """
        Parse a single CSS class into components.
        
        Args:
            css_class: CSS class string (e.g., "t:h:pt_7r")
            
        Returns:
            CSSComponent with parsed information
        """
        if not css_class:
            return CSSComponent(raw_class=css_class, attribute=css_class)
        
        # Try main pattern first
        match = self.css_pattern.match(css_class)
        if match:
            device, pseudo, attribute, numeric_value, unit, text_value = match.groups()
            
            # Determine final value and unit
            if numeric_value is not None:
                # Numeric value with optional unit (e.g., "18x", "7r", "10")
                value = numeric_value
                final_unit = unit
            elif text_value is not None:
                # Text value without unit (e.g., "gradient", "auto", "blue")
                value = text_value
                final_unit = None
            else:
                # No value
                value = None
                final_unit = None
            
            return CSSComponent(
                device_context=device,
                pseudo_context=pseudo,
                attribute=attribute,
                value=value,
                unit=final_unit,
                raw_class=css_class
            )
        
        # Try special pattern for edge cases
        special_match = self.special_pattern.match(css_class)
        if special_match:
            attribute, value = special_match.groups()
            return CSSComponent(
                attribute=attribute,
                value=value,
                raw_class=css_class
            )
        
        # Return raw if unparseable - but still set an attribute for edge cases
        logger.warning(f"Could not parse CSS class: {css_class}")
        return CSSComponent(
            attribute=css_class if css_class else None,  # Treat whole thing as attribute
            raw_class=css_class
        )
    
    def parse_class_list(self, class_string: str) -> List[CSSComponent]:
        """
        Parse a class string with @ separators into components.
        
        Args:
            class_string: String like "mr_auto@ml_auto@pt_7r"
            
        Returns:
            List of parsed CSSComponent objects
        """
        if not class_string:
            return []
        
        classes = class_string.split('@')
        return [self.parse_class(cls.strip()) for cls in classes if cls.strip()]
    
    def parse_element_key(self, element_key: str) -> Tuple[str, List[CSSComponent]]:
        """
        Parse an element key into tag and CSS components.
        
        Args:
            element_key: Key like "div@mr_auto@ml_auto" or "h1@t:fs_18x"
            
        Returns:
            Tuple of (tag, list_of_css_components)
        """
        if '@' in element_key:
            parts = element_key.split('@')
            tag = parts[0]
            class_components = [self.parse_class(cls) for cls in parts[1:] if cls]
        else:
            tag = element_key
            class_components = []
        
        return tag, class_components
    
    def decompose_structure(self, html_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose entire HTML structure into parsed components.
        
        Args:
            html_structure: Nested HTML structure dictionary
            
        Returns:
            List of decomposed element information
        """
        decomposed = []
        self._decompose_recursive(html_structure, decomposed, depth=0, parent_path="")
        return decomposed
    
    def _decompose_recursive(self, structure: Dict[str, Any], decomposed: List[Dict[str, Any]], 
                           depth: int, parent_path: str):
        """Recursively decompose nested structure."""
        for key, value in structure.items():
            # Skip content keys
            if key in ['text', 'src', 'svg']:
                continue
            
            # Parse element key
            tag, css_components = self.parse_element_key(key)
            
            current_path = f"{parent_path}/{key}" if parent_path else key
            
            # Extract content if present
            content = {}
            if isinstance(value, dict):
                for content_key in ['text', 'src', 'svg']:
                    if content_key in value:
                        content[content_key] = value[content_key]
            
            # Create decomposed element
            element_info = {
                'path': current_path,
                'tag': tag,
                'css_components': [comp.to_dict() for comp in css_components],
                'depth': depth,
                'parent_path': parent_path or None,
                'content': content,
                'has_children': self._has_structural_children(value)
            }
            
            decomposed.append(element_info)
            
            # Process children
            if isinstance(value, dict):
                self._decompose_recursive(value, decomposed, depth + 1, current_path)
    
    def _has_structural_children(self, value: Any) -> bool:
        """Check if element has structural (non-content) children."""
        if not isinstance(value, dict):
            return False
        return any(key not in ['text', 'src', 'svg'] for key in value.keys())
    
    def get_attribute_vocabulary(self, structures: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Build attribute vocabulary from multiple structures.
        
        Args:
            structures: List of HTML structure dictionaries
            
        Returns:
            Dictionary mapping attributes to frequency counts
        """
        attribute_counts = {}
        
        for structure in structures:
            decomposed = self.decompose_structure(structure)
            for element in decomposed:
                for css_comp in element['css_components']:
                    attr = css_comp.get('attribute')
                    if attr:
                        attribute_counts[attr] = attribute_counts.get(attr, 0) + 1
        
        return attribute_counts
    
    def get_parsing_statistics(self, structures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get comprehensive parsing statistics.
        
        Args:
            structures: List of HTML structure dictionaries
            
        Returns:
            Statistics about parsing coverage and patterns
        """
        total_classes = 0
        parsed_classes = 0
        device_usage = {}
        pseudo_usage = {}
        attribute_usage = {}
        unit_usage = {}
        
        for structure in structures:
            decomposed = self.decompose_structure(structure)
            for element in decomposed:
                for css_comp in element['css_components']:
                    total_classes += 1
                    
                    # Check if successfully parsed
                    if css_comp.get('attribute'):
                        parsed_classes += 1
                    
                    # Count usage patterns
                    for context_type, usage_dict in [
                        ('device_context', device_usage),
                        ('pseudo_context', pseudo_usage),
                        ('attribute', attribute_usage),
                        ('unit', unit_usage)
                    ]:
                        context_value = css_comp.get(context_type)
                        if context_value:
                            usage_dict[context_value] = usage_dict.get(context_value, 0) + 1
        
        return {
            'total_classes': total_classes,
            'parsed_classes': parsed_classes,
            'parsing_success_rate': parsed_classes / max(total_classes, 1),
            'device_contexts': device_usage,
            'pseudo_contexts': pseudo_usage,
            'attributes': attribute_usage,
            'units': unit_usage,
            'unique_attributes': len(attribute_usage),
            'unique_devices': len(device_usage),
            'unique_pseudos': len(pseudo_usage),
            'unique_units': len(unit_usage)
        }


if __name__ == "__main__":
    # Example usage
    parser = CSSClassParser()
    
    # Test individual class parsing
    test_classes = ["pt_7r", "t:fs_18x", "h:c_blue", "m:h:pt_auto", "mr_auto"]
    
    print("=== CSS Class Parsing Examples ===")
    for css_class in test_classes:
        component = parser.parse_class(css_class)
        print(f"{css_class} -> {component}")
    
    # Test structure decomposition
    test_structure = {
        "div@mr_auto@ml_auto": {
            "h1@t:fs_18x@h:c_blue": {"text": "Welcome"},
            "p@pt_7r": {"text": "Description"}
        }
    }
    
    print(f"\n=== Structure Decomposition ===")
    decomposed = parser.decompose_structure(test_structure)
    for element in decomposed:
        print(f"Path: {element['path']}")
        print(f"Tag: {element['tag']}")
        print(f"CSS Components: {element['css_components']}")
        print("---") 