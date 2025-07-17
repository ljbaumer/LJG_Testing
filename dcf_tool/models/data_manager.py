import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class DataManager:
    """
    Handles loading, saving, and validating financial data for the DCF model.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the DataManager with the directory containing data files.
        
        Args:
            data_dir: Directory path where data files are stored
        """
        self.data_dir = data_dir
        self.sample_data_path = os.path.join(data_dir, "sample_data.csv")
        self.user_data_path = os.path.join(data_dir, "user_data.csv")
        self.data = None
        
    def load_data(self, use_user_data: bool = False) -> pd.DataFrame:
        """
        Load financial data from either sample data or user-modified data.
        
        Args:
            use_user_data: If True, load from user_data.csv if it exists, otherwise from sample_data.csv
            
        Returns:
            DataFrame containing the financial data
        """
        file_path = self.user_data_path if use_user_data and os.path.exists(self.user_data_path) else self.sample_data_path
        
        try:
            data = pd.read_csv(file_path)
            # Set Year as index but keep it as a column too
            data = data.set_index('Year', drop=False)
            self.data = data
            return data
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {str(e)}")
    
    def save_data(self, data: pd.DataFrame, as_user_data: bool = True) -> str:
        """
        Save financial data to a CSV file.
        
        Args:
            data: DataFrame containing the financial data to save
            as_user_data: If True, save to user_data.csv, otherwise to sample_data.csv
            
        Returns:
            Path to the saved file
        """
        file_path = self.user_data_path if as_user_data else self.sample_data_path
        
        try:
            # Reset index if Year is the index
            if data.index.name == 'Year':
                data = data.reset_index(drop=True)
            
            # Ensure Year is the first column
            if 'Year' in data.columns and data.columns[0] != 'Year':
                year_col = data.pop('Year')
                data.insert(0, 'Year', year_col)
                
            data.to_csv(file_path, index=False)
            return file_path
        except Exception as e:
            raise Exception(f"Error saving data to {file_path}: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the financial data for consistency and completeness.
        
        Args:
            data: DataFrame containing the financial data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_columns = [
            'Year', 'Revenue', 'EBITDA', 'Depreciation_Amortization', 
            'EBIT', 'Tax_Rate', 'NOPAT', 'Capital_Expenditures', 
            'Change_in_Working_Capital', 'Free_Cash_Flow'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for numeric data in key columns
        numeric_columns = [col for col in required_columns if col != 'Year']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column {col} must contain numeric values")
        
        # Check for consistency in calculations
        if all(col in data.columns for col in ['EBIT', 'Tax_Rate', 'NOPAT']):
            calculated_nopat = data['EBIT'] * (1 - data['Tax_Rate'])
            if not np.allclose(calculated_nopat, data['NOPAT'], rtol=0.01):
                errors.append("NOPAT values are inconsistent with EBIT and Tax_Rate")
        
        if all(col in data.columns for col in ['EBITDA', 'Depreciation_Amortization', 'EBIT']):
            calculated_ebit = data['EBITDA'] - data['Depreciation_Amortization']
            if not np.allclose(calculated_ebit, data['EBIT'], rtol=0.01):
                errors.append("EBIT values are inconsistent with EBITDA and Depreciation_Amortization")
        
        if all(col in data.columns for col in ['NOPAT', 'Depreciation_Amortization', 'Capital_Expenditures', 'Change_in_Working_Capital', 'Free_Cash_Flow']):
            calculated_fcf = data['NOPAT'] + data['Depreciation_Amortization'] - data['Capital_Expenditures'] - data['Change_in_Working_Capital']
            if not np.allclose(calculated_fcf, data['Free_Cash_Flow'], rtol=0.01):
                errors.append("Free Cash Flow values are inconsistent with component calculations")
        
        return len(errors) == 0, errors
    
    def extend_forecast_years(self, data: pd.DataFrame, forecast_years: int = 5) -> pd.DataFrame:
        """
        Extend the data with forecast years.
        
        Args:
            data: DataFrame containing historical financial data
            forecast_years: Number of years to forecast
            
        Returns:
            DataFrame with historical and forecast years
        """
        if 'Year' not in data.columns:
            raise ValueError("Data must contain a 'Year' column")
        
        # Get the last historical year
        last_year = data['Year'].max()
        
        # Create a new DataFrame for forecast years
        forecast_data = []
        
        for i in range(1, forecast_years + 1):
            forecast_year = last_year + i
            forecast_row = {'Year': forecast_year}
            forecast_data.append(forecast_row)
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Combine historical and forecast data
        combined_df = pd.concat([data, forecast_df], ignore_index=True)
        
        return combined_df
    
    def calculate_historical_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate average historical metrics for use in forecasting.
        
        Args:
            data: DataFrame containing historical financial data
            
        Returns:
            Dictionary of average metrics
        """
        historical_data = data[~data['Year'].isin(self.get_forecast_years(data))]
        
        metrics = {}
        
        if 'Revenue_Growth' in historical_data.columns:
            metrics['avg_revenue_growth'] = historical_data['Revenue_Growth'].mean()
        
        if 'EBITDA_Margin' in historical_data.columns:
            metrics['avg_ebitda_margin'] = historical_data['EBITDA_Margin'].mean()
        
        if all(col in historical_data.columns for col in ['Depreciation_Amortization', 'Revenue']):
            metrics['avg_da_to_revenue'] = (historical_data['Depreciation_Amortization'] / historical_data['Revenue']).mean()
        
        if all(col in historical_data.columns for col in ['Capital_Expenditures', 'Revenue']):
            metrics['avg_capex_to_revenue'] = (historical_data['Capital_Expenditures'] / historical_data['Revenue']).mean()
        
        if all(col in historical_data.columns for col in ['Change_in_Working_Capital', 'Revenue']):
            metrics['avg_wcchange_to_revenue'] = (historical_data['Change_in_Working_Capital'] / historical_data['Revenue']).mean()
        
        if 'Tax_Rate' in historical_data.columns:
            metrics['avg_tax_rate'] = historical_data['Tax_Rate'].mean()
        
        return metrics
    
    def get_forecast_years(self, data: pd.DataFrame) -> List[int]:
        """
        Identify which years in the data are forecast years (those without complete data).
        
        Args:
            data: DataFrame containing financial data
            
        Returns:
            List of forecast years
        """
        # Years with missing FCF are considered forecast years
        if 'Free_Cash_Flow' in data.columns:
            forecast_years = data[data['Free_Cash_Flow'].isna()]['Year'].tolist()
            return forecast_years
        
        return []
