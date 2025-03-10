import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScenarioManager:
    """
    Manages different scenarios for the 3-statement model.
    """
    
    def __init__(self):
        """
        Initialize the ScenarioManager.
        """
        self.scenarios = {}
        self.base_parameters = {}
        self.upside_parameters = {}
        self.downside_parameters = {}
        self.custom_scenarios = {}
    
    def set_base_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the base case parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self.base_parameters = parameters.copy()
        self.scenarios["base"] = parameters.copy()
    
    def set_upside_parameters(self, parameters: Dict[str, Any] = None, percentage_change: float = 0.1) -> None:
        """
        Set the upside case parameters.
        
        Args:
            parameters: Dictionary of parameter names and values (if None, derive from base case)
            percentage_change: Percentage change from base case (if parameters is None)
        """
        if parameters is not None:
            self.upside_parameters = parameters.copy()
        else:
            # Derive from base case
            if not self.base_parameters:
                raise ValueError("Base parameters must be set before deriving upside parameters")
            
            self.upside_parameters = self.base_parameters.copy()
            
            # Adjust growth-related parameters upward
            for param in ["revenue_growth", "ebitda_margin", "net_margin"]:
                if param in self.upside_parameters:
                    self.upside_parameters[param] = self.upside_parameters[param] * (1 + percentage_change)
            
            # Adjust cost-related parameters downward
            for param in ["tax_rate", "capex_to_revenue", "opex_to_revenue"]:
                if param in self.upside_parameters:
                    self.upside_parameters[param] = self.upside_parameters[param] * (1 - percentage_change)
        
        self.scenarios["upside"] = self.upside_parameters.copy()
    
    def set_downside_parameters(self, parameters: Dict[str, Any] = None, percentage_change: float = 0.1) -> None:
        """
        Set the downside case parameters.
        
        Args:
            parameters: Dictionary of parameter names and values (if None, derive from base case)
            percentage_change: Percentage change from base case (if parameters is None)
        """
        if parameters is not None:
            self.downside_parameters = parameters.copy()
        else:
            # Derive from base case
            if not self.base_parameters:
                raise ValueError("Base parameters must be set before deriving downside parameters")
            
            self.downside_parameters = self.base_parameters.copy()
            
            # Adjust growth-related parameters downward
            for param in ["revenue_growth", "ebitda_margin", "net_margin"]:
                if param in self.downside_parameters:
                    self.downside_parameters[param] = self.downside_parameters[param] * (1 - percentage_change)
            
            # Adjust cost-related parameters upward
            for param in ["tax_rate", "capex_to_revenue", "opex_to_revenue"]:
                if param in self.downside_parameters:
                    self.downside_parameters[param] = self.downside_parameters[param] * (1 + percentage_change)
        
        self.scenarios["downside"] = self.downside_parameters.copy()
    
    def add_custom_scenario(self, name: str, parameters: Dict[str, Any]) -> None:
        """
        Add a custom scenario.
        
        Args:
            name: Name of the scenario
            parameters: Dictionary of parameter names and values
        """
        self.custom_scenarios[name] = parameters.copy()
        self.scenarios[name] = parameters.copy()
    
    def get_scenario_parameters(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get the parameters for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Dictionary of parameter names and values
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        return self.scenarios[scenario_name].copy()
    
    def get_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all scenarios.
        
        Returns:
            Dictionary mapping scenario names to parameter dictionaries
        """
        return self.scenarios.copy()
    
    def compare_scenarios(self, parameter_name: str) -> Dict[str, Any]:
        """
        Compare a specific parameter across all scenarios.
        
        Args:
            parameter_name: Name of the parameter to compare
            
        Returns:
            Dictionary mapping scenario names to parameter values
        """
        comparison = {}
        
        for scenario_name, parameters in self.scenarios.items():
            if parameter_name in parameters:
                comparison[scenario_name] = parameters[parameter_name]
        
        return comparison
    
    def apply_scenario_to_forecast(self, forecast_model, scenario_name: str) -> pd.DataFrame:
        """
        Apply a scenario to a forecast model and generate forecast data.
        
        Args:
            forecast_model: The forecast model to use
            scenario_name: Name of the scenario to apply
            
        Returns:
            DataFrame containing the forecast data
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        # Get the scenario parameters
        parameters = self.scenarios[scenario_name]
        
        # Apply the parameters to the forecast model
        forecast_model.set_parameters(parameters)
        
        # Generate the forecast
        forecast_data = forecast_model.generate_forecast()
        
        return forecast_data
