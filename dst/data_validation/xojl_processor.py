"""
XOJL Semantic Layout Processor

Handles XOJL validation, semantic element extraction, and layout analysis.
Pure processing logic without any file I/O operations.
"""

import json
import logging
from typing import Dict, List, Any, Set, Optional

logger = logging.getLogger(__name__)


class XOJLProcessor:
    """
    Processes XOJL semantic layout structures.
    
    Validates layout descriptions, extracts semantic elements, and ensures
    consistency between layout and HTML structures. No file operations.
    """
    
    def __init__(self):
        self.valid_elements = {
            'hero', 'navigation', 'sidebar', 'content', 'footer', 'header',
            'button', 'form', 'image', 'text', 'card', 'list', 'menu',
            'section', 'article', 'aside', 'main', 'banner', 'search',
            'gallery', 'table', 'modal', 'tooltip', 'dropdown', 'tab',
            'accordion', 'carousel', 'breadcrumb', 'pagination'
        }
        self.required_fields = {'type', 'elements'}
        self.semantic_categories = {
            'layout': ['hero', 'navigation', 'sidebar', 'content', 'footer', 'header'],
            'components': ['button', 'form', 'image', 'text', 'card', 'list', 'menu'],
            'structural': ['section', 'article', 'aside', 'main', 'banner'],
            'interactive': ['search', 'modal', 'tooltip', 'dropdown', 'tab'],
            'containers': ['gallery', 'table', 'accordion', 'carousel']
        }
    
    def parse_layout(self, xojl_data: Any) -> Dict[str, Any]:
        """
        Parse XOJL layout from various input formats.
        
        Args:
            xojl_data: XOJL layout as JSON string or dictionary
            
        Returns:
            Parsed XOJL layout as nested dictionary
        """
        if isinstance(xojl_data, str):
            try:
                return json.loads(xojl_data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in XOJL layout: {e}")
                raise ValueError(f"Failed to parse XOJL layout: {e}")
        elif isinstance(xojl_data, dict):
            return xojl_data
        else:
            raise TypeError(f"Unsupported XOJL data type: {type(xojl_data)}")
    
    def validate_structure(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate XOJL structure completeness and correctness.
        
        Args:
            layout: Parsed XOJL layout dictionary
            
        Returns:
            Validation result with status and details
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required fields
        missing_fields = self.required_fields - set(layout.keys())
        if missing_fields:
            validation_result['is_valid'] = False
            validation_result['errors'].append(
                f"Missing required fields: {missing_fields}"
            )
        
        # Validate layout type
        if 'type' in layout:
            if not isinstance(layout['type'], str):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Layout type must be string")
        
        # Validate elements structure
        if 'elements' in layout:
            elements_validation = self._validate_elements(layout['elements'])
            validation_result['errors'].extend(elements_validation['errors'])
            validation_result['warnings'].extend(elements_validation['warnings'])
            validation_result['statistics'].update(elements_validation['statistics'])
            
            if elements_validation['errors']:
                validation_result['is_valid'] = False
        
        return validation_result
    
    def _validate_elements(self, elements: Any) -> Dict[str, Any]:
        """Validate elements structure and content."""
        result = {
            'errors': [],
            'warnings': [],
            'statistics': {
                'total_elements': 0,
                'valid_elements': 0,
                'element_types': [],
                'categories_used': set()
            }
        }
        
        if not isinstance(elements, list):
            result['errors'].append("Elements must be a list")
            return result
        
        result['statistics']['total_elements'] = len(elements)
        
        for i, element in enumerate(elements):
            element_validation = self._validate_single_element(element, i)
            result['errors'].extend(element_validation['errors'])
            result['warnings'].extend(element_validation['warnings'])
            
            if not element_validation['errors']:
                result['statistics']['valid_elements'] += 1
                element_type = element.get('type', 'unknown')
                result['statistics']['element_types'].append(element_type)
                
                # Determine category
                category = self._get_element_category(element_type)
                if category:
                    result['statistics']['categories_used'].add(category)
        
        # Convert set to list for JSON serialization
        result['statistics']['categories_used'] = list(result['statistics']['categories_used'])
        
        return result
    
    def _validate_single_element(self, element: Any, index: int) -> Dict[str, Any]:
        """Validate a single semantic element."""
        result = {'errors': [], 'warnings': []}
        
        if not isinstance(element, dict):
            result['errors'].append(f"Element {index} must be a dictionary")
            return result
        
        # Check required element fields
        if 'type' not in element:
            result['errors'].append(f"Element {index} missing 'type' field")
        else:
            element_type = element['type']
            if element_type not in self.valid_elements:
                result['warnings'].append(
                    f"Element {index} has unknown type '{element_type}'"
                )
        
        # Validate optional fields
        if 'id' in element and not isinstance(element['id'], str):
            result['errors'].append(f"Element {index} 'id' must be string")
        
        if 'attributes' in element and not isinstance(element['attributes'], dict):
            result['errors'].append(f"Element {index} 'attributes' must be dictionary")
        
        if 'children' in element:
            if isinstance(element['children'], list):
                for j, child in enumerate(element['children']):
                    child_validation = self._validate_single_element(child, f"{index}.{j}")
                    result['errors'].extend(child_validation['errors'])
                    result['warnings'].extend(child_validation['warnings'])
            else:
                result['errors'].append(f"Element {index} 'children' must be list")
        
        return result
    
    def _get_element_category(self, element_type: str) -> Optional[str]:
        """Get the category of a semantic element."""
        for category, elements in self.semantic_categories.items():
            if element_type in elements:
                return category
        return None
    
    def extract_used_elements(self, layout: Dict[str, Any]) -> List[str]:
        """
        Extract all semantic elements used in the layout.
        
        Args:
            layout: Parsed XOJL layout dictionary
            
        Returns:
            List of unique semantic element types used
        """
        elements = set()
        self._extract_elements_recursive(layout.get('elements', []), elements)
        return sorted(list(elements))
    
    def _extract_elements_recursive(self, elements_list: List[Any], elements_set: Set[str]):
        """Recursively extract element types from nested structure."""
        for element in elements_list:
            if isinstance(element, dict) and 'type' in element:
                elements_set.add(element['type'])
                
                # Process children
                if 'children' in element and isinstance(element['children'], list):
                    self._extract_elements_recursive(element['children'], elements_set)
    
    def analyze_layout_complexity(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze layout complexity metrics.
        
        Args:
            layout: Parsed XOJL layout dictionary
            
        Returns:
            Dictionary containing complexity metrics
        """
        metrics = {
            'total_elements': 0,
            'max_depth': 0,
            'element_distribution': {},
            'category_distribution': {},
            'has_nested_structure': False,
            'complexity_score': 0
        }
        
        elements = layout.get('elements', [])
        if not elements:
            return metrics
        
        # Calculate metrics
        self._calculate_complexity_recursive(elements, metrics, depth=0)
        
        # Calculate complexity score
        metrics['complexity_score'] = self._calculate_complexity_score(metrics)
        
        return metrics
    
    def _calculate_complexity_recursive(self, elements: List[Any], metrics: Dict[str, Any], depth: int):
        """Recursively calculate complexity metrics."""
        if depth > metrics['max_depth']:
            metrics['max_depth'] = depth
        
        for element in elements:
            if not isinstance(element, dict) or 'type' not in element:
                continue
                
            metrics['total_elements'] += 1
            element_type = element['type']
            
            # Update element distribution
            metrics['element_distribution'][element_type] = (
                metrics['element_distribution'].get(element_type, 0) + 1
            )
            
            # Update category distribution
            category = self._get_element_category(element_type)
            if category:
                metrics['category_distribution'][category] = (
                    metrics['category_distribution'].get(category, 0) + 1
                )
            
            # Check for nested structure
            if 'children' in element and isinstance(element['children'], list):
                if element['children']:
                    metrics['has_nested_structure'] = True
                    self._calculate_complexity_recursive(
                        element['children'], metrics, depth + 1
                    )
    
    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall complexity score based on metrics."""
        base_score = metrics['total_elements']
        depth_factor = metrics['max_depth'] * 2
        nesting_factor = 5 if metrics['has_nested_structure'] else 0
        diversity_factor = len(metrics['element_distribution']) * 0.5
        
        return base_score + depth_factor + nesting_factor + diversity_factor
    
    def validate_semantic_consistency(self, layout: Dict[str, Any], html_elements: List[str]) -> Dict[str, Any]:
        """
        Validate consistency between XOJL layout and HTML elements.
        
        Args:
            layout: Parsed XOJL layout dictionary
            html_elements: List of HTML element types from structure
            
        Returns:
            Consistency validation result
        """
        layout_elements = set(self.extract_used_elements(layout))
        html_elements_set = set(html_elements)
        
        # Find semantic gaps
        semantic_only = layout_elements - html_elements_set
        html_only = html_elements_set - layout_elements
        common_elements = layout_elements & html_elements_set
        
        consistency_score = (
            len(common_elements) / max(len(layout_elements), 1)
        ) if layout_elements else 1.0
        
        return {
            'consistency_score': consistency_score,
            'is_consistent': consistency_score >= 0.7,
            'common_elements': sorted(list(common_elements)),
            'semantic_only_elements': sorted(list(semantic_only)),
            'html_only_elements': sorted(list(html_only)),
            'total_semantic_elements': len(layout_elements),
            'total_html_elements': len(html_elements_set)
        }
    
    def get_layout_category(self, layout: Dict[str, Any]) -> str:
        """
        Determine the primary category of the layout.
        
        Args:
            layout: Parsed XOJL layout dictionary
            
        Returns:
            Primary layout category
        """
        elements = self.extract_used_elements(layout)
        category_counts = {}
        
        for element in elements:
            category = self._get_element_category(element)
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        if not category_counts:
            return 'unknown'
        
        # Return category with most elements
        return max(category_counts.items(), key=lambda x: x[1])[0] 