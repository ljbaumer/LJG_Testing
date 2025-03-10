import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from dcf_tool.utils.financial_utils import (
    calculate_terminal_value_perpetuity,
    calculate_terminal_value_exit_multiple
)


class GrowthMethod:
    """Base class for terminal value calculation methods."""
    
    def __init__(self, name: str):
        """
        Initialize the growth method.
        
        Args:
            name: Name of the growth method
        """
        self.name = name
    
    def calculate_terminal_value(self, **kwargs) -> float:
        """
        Calculate terminal value.
        
        Returns:
            Terminal value
        """
        raise NotImplementedError("Subclasses must implement calculate_terminal_value")
    
    def get_parameters(self) -> Dict[str, Tuple[str, str, Optional[float]]]:
        """
        Get parameters required for this growth method.
        
        Returns:
            Dictionary of parameter names mapped to tuples of (label, description, default_value)
        """
        raise NotImplementedError("Subclasses must implement get_parameters")


class PerpetuityGrowthMethod(GrowthMethod):
    """Terminal value calculation using the perpetuity growth method."""
    
    def __init__(self):
        """Initialize the perpetuity growth method."""
        super().__init__("Perpetuity Growth Method")
    
    def calculate_terminal_value(self, 
                                final_cash_flow: float, 
                                growth_rate: float, 
                                discount_rate: float, 
                                **kwargs) -> float:
        """
        Calculate terminal value using the perpetuity growth method.
        
        Args:
            final_cash_flow: Cash flow in the final forecast year
            growth_rate: Long-term growth rate (as a decimal)
            discount_rate: Discount rate (WACC, as a decimal)
            
        Returns:
            Terminal value
        """
        return calculate_terminal_value_perpetuity(
            final_cash_flow=final_cash_flow,
            growth_rate=growth_rate,
            discount_rate=discount_rate
        )
    
    def get_parameters(self) -> Dict[str, Tuple[str, str, Optional[float]]]:
        """
        Get parameters required for the perpetuity growth method.
        
        Returns:
            Dictionary of parameter names mapped to tuples of (label, description, default_value)
        """
        return {
            "growth_rate": (
                "Long-term Growth Rate", 
                "Expected long-term growth rate of cash flows into perpetuity (e.g., 0.02 for 2%)",
                0.02
            )
        }


class ExitMultipleMethod(GrowthMethod):
    """Terminal value calculation using the exit multiple method."""
    
    def __init__(self):
        """Initialize the exit multiple method."""
        super().__init__("Exit Multiple Method")
    
    def calculate_terminal_value(self, 
                                final_metric: float, 
                                multiple: float, 
                                **kwargs) -> float:
        """
        Calculate terminal value using the exit multiple method.
        
        Args:
            final_metric: Financial metric in the final forecast year (e.g., EBITDA)
            multiple: Exit multiple to apply
            
        Returns:
            Terminal value
        """
        return calculate_terminal_value_exit_multiple(
            final_metric=final_metric,
            multiple=multiple
        )
    
    def get_parameters(self) -> Dict[str, Tuple[str, str, Optional[float]]]:
        """
        Get parameters required for the exit multiple method.
        
        Returns:
            Dictionary of parameter names mapped to tuples of (label, description, default_value)
        """
        return {
            "multiple_type": (
                "Multiple Type",
                "Type of multiple to use (e.g., EV/EBITDA, EV/EBIT)",
                "EV/EBITDA"
            ),
            "multiple": (
                "Multiple Value",
                "Value of the multiple to apply",
                8.0
            )
        }


def get_available_growth_methods() -> Dict[str, GrowthMethod]:
    """
    Get all available growth methods.
    
    Returns:
        Dictionary of growth method names mapped to their instances
    """
    methods = {
        "perpetuity_growth": PerpetuityGrowthMethod(),
        "exit_multiple": ExitMultipleMethod()
    }
    
    return methods


def get_growth_method(method_name: str) -> Optional[GrowthMethod]:
    """
    Get a growth method by name.
    
    Args:
        method_name: Name of the growth method
        
    Returns:
        GrowthMethod instance or None if not found
    """
    methods = get_available_growth_methods()
    return methods.get(method_name)
