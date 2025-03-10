import os
import pandas as pd
from typing import Dict, List, Optional, Any


class ExcelConverter:
    """
    Handles conversion of Excel financial data to CSV format.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the ExcelConverter with the directory for data files.
        
        Args:
            data_dir: Directory path where data files are stored
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def read_financial_excel(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """
        Read financial statements from an Excel file.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary mapping statement types to DataFrames
        """
        # Check if the file exists
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        # Try to read the Excel file
        try:
            # Read the Excel file
            excel_file = pd.ExcelFile(excel_path)
            
            # Get the sheet names
            sheet_names = excel_file.sheet_names
            
            # Initialize the dictionary to store the financial statements
            financial_statements = {}
            
            # Check for common sheet names for financial statements
            income_statement_sheets = ["Income Statement", "Income", "P&L", "Profit and Loss"]
            balance_sheet_sheets = ["Balance Sheet", "Balance", "BS"]
            cash_flow_sheets = ["Cash Flow", "Cash Flow Statement", "CF"]
            
            # Find the income statement sheet
            income_statement_sheet = None
            for sheet in income_statement_sheets:
                if sheet in sheet_names:
                    income_statement_sheet = sheet
                    break
            
            # Find the balance sheet sheet
            balance_sheet_sheet = None
            for sheet in balance_sheet_sheets:
                if sheet in sheet_names:
                    balance_sheet_sheet = sheet
                    break
            
            # Find the cash flow sheet
            cash_flow_sheet = None
            for sheet in cash_flow_sheets:
                if sheet in sheet_names:
                    cash_flow_sheet = sheet
                    break
            
            # If we couldn't find the sheets by name, try to infer from the content
            if income_statement_sheet is None and balance_sheet_sheet is None and cash_flow_sheet is None:
                # Try to infer from the content
                for sheet in sheet_names:
                    # Read the sheet
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    
                    # Check if it's an income statement
                    if any(col.lower() in ["revenue", "sales", "income"] for col in df.columns) and any(col.lower() in ["net income", "net profit", "net earnings"] for col in df.columns):
                        income_statement_sheet = sheet
                    
                    # Check if it's a balance sheet
                    elif any(col.lower() in ["assets", "liabilities", "equity"] for col in df.columns):
                        balance_sheet_sheet = sheet
                    
                    # Check if it's a cash flow statement
                    elif any(col.lower() in ["cash flow", "operating activities", "investing activities", "financing activities"] for col in df.columns):
                        cash_flow_sheet = sheet
            
            # If we still couldn't find the sheets, use the first three sheets
            if income_statement_sheet is None and balance_sheet_sheet is None and cash_flow_sheet is None and len(sheet_names) >= 3:
                income_statement_sheet = sheet_names[0]
                balance_sheet_sheet = sheet_names[1]
                cash_flow_sheet = sheet_names[2]
            
            # Read the income statement
            if income_statement_sheet is not None:
                financial_statements["income_statement"] = pd.read_excel(excel_file, sheet_name=income_statement_sheet)
            
            # Read the balance sheet
            if balance_sheet_sheet is not None:
                financial_statements["balance_sheet"] = pd.read_excel(excel_file, sheet_name=balance_sheet_sheet)
            
            # Read the cash flow statement
            if cash_flow_sheet is not None:
                financial_statements["cash_flow"] = pd.read_excel(excel_file, sheet_name=cash_flow_sheet)
            
            return financial_statements
        
        except Exception as e:
            raise Exception(f"Error reading Excel file: {e}")
    
    def convert_excel_to_csv(self, excel_path: str) -> Dict[str, str]:
        """
        Convert an Excel file to CSV files.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary mapping statement types to CSV file paths
        """
        # Read the Excel file
        financial_statements = self.read_financial_excel(excel_path)
        
        # Get the base name of the Excel file
        base_name = os.path.splitext(os.path.basename(excel_path))[0]
        
        # Create a directory for the processed files
        processed_dir = os.path.join(self.processed_dir, base_name)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Initialize the dictionary to store the CSV file paths
        csv_paths = {}
        
        # Convert each statement to CSV
        for statement_type, df in financial_statements.items():
            # Create the CSV file path
            csv_path = os.path.join(processed_dir, f"{statement_type.replace('_', ' ').title()}.csv")
            
            # Save the DataFrame to CSV
            df.to_csv(csv_path, index=False)
            
            # Store the CSV file path
            csv_paths[statement_type] = csv_path
        
        return csv_paths
