import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import datetime

class ScenarioManager:
    """
    Manages saving, loading, and comparing different valuation scenarios.
    """
    
    def __init__(self, scenarios_dir: str = None):
        """
        Initialize the ScenarioManager.
        
        Args:
            scenarios_dir: Directory path where scenario files are stored
        """
        # If no directory is provided, create a scenarios directory in the data directory
        if scenarios_dir is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scenarios_dir = os.path.join(current_dir, "data", "scenarios")
        
        self.scenarios_dir = scenarios_dir
        
        # Create the scenarios directory if it doesn't exist
        if not os.path.exists(self.scenarios_dir):
            os.makedirs(self.scenarios_dir)
    
    def save_scenario(self, 
                     name: str, 
                     description: str, 
                     parameters: Dict[str, Any], 
                     results: Dict[str, Any] = None,
                     data: pd.DataFrame = None) -> str:
        """
        Save a scenario to a file.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
            parameters: Dictionary of model parameters
            results: Dictionary of valuation results (optional)
            data: DataFrame containing the financial data (optional)
            
        Returns:
            Path to the saved scenario file
        """
        # Create a filename based on the scenario name
        safe_name = name.replace(" ", "_").lower()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        file_path = os.path.join(self.scenarios_dir, filename)
        
        # Prepare the scenario data
        scenario_data = {
            "name": name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "parameters": parameters
        }
        
        # Add results if provided
        if results is not None:
            scenario_data["results"] = results
        
        # Add data if provided
        if data is not None:
            scenario_data["data"] = data.to_dict(orient="records")
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        
        return file_path
    
    def load_scenario(self, file_path: str) -> Dict[str, Any]:
        """
        Load a scenario from a file.
        
        Args:
            file_path: Path to the scenario file
            
        Returns:
            Dictionary containing the scenario data
        """
        try:
            with open(file_path, 'r') as f:
                scenario_data = json.load(f)
            
            # Convert data back to DataFrame if present
            if "data" in scenario_data:
                scenario_data["data"] = pd.DataFrame(scenario_data["data"])
            
            return scenario_data
        except Exception as e:
            raise Exception(f"Error loading scenario from {file_path}: {str(e)}")
    
    def get_available_scenarios(self) -> List[Dict[str, str]]:
        """
        Get a list of available scenarios.
        
        Returns:
            List of dictionaries containing scenario metadata
        """
        scenarios = []
        
        # List all JSON files in the scenarios directory
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.scenarios_dir, filename)
                
                try:
                    # Load basic metadata without loading the entire file
                    with open(file_path, 'r') as f:
                        scenario_data = json.load(f)
                    
                    scenarios.append({
                        "name": scenario_data.get("name", "Unnamed"),
                        "description": scenario_data.get("description", ""),
                        "created_at": scenario_data.get("created_at", ""),
                        "file_path": file_path
                    })
                except:
                    # Skip files that can't be loaded
                    continue
        
        # Sort by creation date (newest first)
        scenarios.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return scenarios
    
    def delete_scenario(self, file_path: str) -> bool:
        """
        Delete a scenario file.
        
        Args:
            file_path: Path to the scenario file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except:
            return False
    
    def compare_scenarios(self, scenario_paths: List[str]) -> Dict[str, Any]:
        """
        Compare multiple scenarios.
        
        Args:
            scenario_paths: List of paths to scenario files
            
        Returns:
            Dictionary containing comparison data
        """
        if not scenario_paths:
            return {}
        
        scenarios = []
        for path in scenario_paths:
            try:
                scenario = self.load_scenario(path)
                scenarios.append(scenario)
            except:
                # Skip scenarios that can't be loaded
                continue
        
        if not scenarios:
            return {}
        
        # Extract parameters and results for comparison
        comparison = {
            "scenarios": [s.get("name", "Unnamed") for s in scenarios],
            "parameters": {},
            "results": {}
        }
        
        # Compare parameters
        all_params = set()
        for scenario in scenarios:
            if "parameters" in scenario:
                all_params.update(scenario["parameters"].keys())
        
        for param in all_params:
            comparison["parameters"][param] = [
                scenario.get("parameters", {}).get(param, None) 
                for scenario in scenarios
            ]
        
        # Compare results
        all_results = set()
        for scenario in scenarios:
            if "results" in scenario:
                all_results.update(scenario["results"].keys())
        
        for result in all_results:
            comparison["results"][result] = [
                scenario.get("results", {}).get(result, None) 
                for scenario in scenarios
            ]
        
        return comparison
