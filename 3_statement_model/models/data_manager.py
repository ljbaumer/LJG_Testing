import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.excel_converter import ExcelConverter
from utils.financial_utils import calculate_growth_rate


class DataManager:
    """
    Handles loading, saving, and validating financial data for the 3-statement model.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the DataManager with the directory containing data files.
        
        Args:
            data_dir: Directory path where data files are stored
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.excel_converter = ExcelConverter(data_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize data containers
        self.income_statement = None
        self.balance_sheet = None
        self.cash_flow = None
        self.combined_data = None
        self.historical_end_year = None
    
    def load_excel_file(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load financial data from an Excel file.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary mapping statement types to DataFrames
        """
        # Read financial statements from Excel
        financial_statements = self.excel_converter.read_financial_excel(excel_path)
        
        # Store the statements
        if "income_statement" in financial_statements:
            self.income_statement = financial_statements["income_statement"]
        
        if "balance_sheet" in financial_statements:
            self.balance_sheet = financial_statements["balance_sheet"]
        
        if "cash_flow" in financial_statements:
            self.cash_flow = financial_statements["cash_flow"]
        
        # Convert to CSV for persistence
        self.excel_converter.convert_excel_to_csv(excel_path)
        
        return financial_statements
    
    def load_csv_files(self, file_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load financial data from CSV files.
        
        Args:
            file_dir: Directory containing CSV files
            
        Returns:
            Dictionary mapping statement types to DataFrames
        """
        financial_statements = {}
        
        # Check for income statement
        income_statement_path = os.path.join(file_dir, "Income Statement.csv")
        if os.path.exists(income_statement_path):
            financial_statements["income_statement"] = pd.read_csv(income_statement_path)
            self.income_statement = financial_statements["income_statement"]
        
        # Check for balance sheet
        balance_sheet_path = os.path.join(file_dir, "Balance Sheet.csv")
        if os.path.exists(balance_sheet_path):
            financial_statements["balance_sheet"] = pd.read_csv(balance_sheet_path)
            self.balance_sheet = financial_statements["balance_sheet"]
        
        # Check for cash flow statement
        cash_flow_path = os.path.join(file_dir, "Cash Flow.csv")
        if os.path.exists(cash_flow_path):
            financial_statements["cash_flow"] = pd.read_csv(cash_flow_path)
            self.cash_flow = financial_statements["cash_flow"]
        
        return financial_statements
    
    def save_data(self, data: Dict[str, pd.DataFrame], file_dir: str) -> None:
        """
        Save financial data to CSV files.
        
        Args:
            data: Dictionary mapping statement types to DataFrames
            file_dir: Directory to save the CSV files
        """
        os.makedirs(file_dir, exist_ok=True)
        
        for statement_type, df in data.items():
            file_name = f"{statement_type.replace('_', ' ').title()}.csv"
            file_path = os.path.join(file_dir, file_name)
            df.to_csv(file_path, index=False)
    
    def standardize_income_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the income statement format.
        
        Args:
            df: Income statement DataFrame
            
        Returns:
            Standardized income statement DataFrame
        """
        # Create a copy to avoid modifying the original
        std_df = df.copy()
        
        # Identify the year column
        year_col = None
        for col in std_df.columns:
            if isinstance(col, str) and any(x in col.lower() for x in ['year', 'date', 'period']):
                year_col = col
                break
        
        if year_col is None and std_df.shape[1] > 1:
            # Assume the first column is the year/period
            year_col = std_df.columns[0]
        
        # Rename the year column
        if year_col is not None:
            std_df.rename(columns={year_col: 'Year'}, inplace=True)
        
        # Ensure Year column is numeric
        if 'Year' in std_df.columns:
            std_df['Year'] = pd.to_numeric(std_df['Year'], errors='coerce')
        
        # Identify and standardize key metrics
        column_mapping = {}
        
        for col in std_df.columns:
            col_lower = str(col).lower()
            
            # Revenue
            if any(x in col_lower for x in ['revenue', 'sales', 'turnover']):
                column_mapping[col] = 'Revenue'
            
            # Cost of Revenue
            elif any(x in col_lower for x in ['cost of revenue', 'cost of sales', 'cogs', 'cost of goods sold']):
                column_mapping[col] = 'Cost_of_Revenue'
            
            # Gross Profit
            elif 'gross profit' in col_lower or 'gross margin' in col_lower:
                column_mapping[col] = 'Gross_Profit'
            
            # Operating Expenses
            elif any(x in col_lower for x in ['operating expenses', 'opex']):
                column_mapping[col] = 'Operating_Expenses'
            
            # Research and Development
            elif any(x in col_lower for x in ['research', 'r&d', 'development']):
                column_mapping[col] = 'Research_Development'
            
            # Sales and Marketing
            elif any(x in col_lower for x in ['sales and marketing', 'marketing', 's&m']):
                column_mapping[col] = 'Sales_Marketing'
            
            # General and Administrative
            elif any(x in col_lower for x in ['general', 'administrative', 'g&a']):
                column_mapping[col] = 'General_Administrative'
            
            # Depreciation and Amortization
            elif any(x in col_lower for x in ['depreciation', 'amortization', 'd&a']):
                column_mapping[col] = 'Depreciation_Amortization'
            
            # EBITDA
            elif 'ebitda' in col_lower:
                column_mapping[col] = 'EBITDA'
            
            # EBIT / Operating Income
            elif 'ebit' in col_lower or 'operating income' in col_lower or 'operating profit' in col_lower:
                column_mapping[col] = 'EBIT'
            
            # Interest Expense
            elif 'interest expense' in col_lower or 'interest paid' in col_lower:
                column_mapping[col] = 'Interest_Expense'
            
            # Interest Income
            elif 'interest income' in col_lower or 'interest received' in col_lower:
                column_mapping[col] = 'Interest_Income'
            
            # Other Income/Expense
            elif 'other income' in col_lower or 'other expense' in col_lower:
                column_mapping[col] = 'Other_Income_Expense'
            
            # Pre-tax Income / EBT
            elif any(x in col_lower for x in ['pre-tax', 'pretax', 'ebt', 'earnings before tax']):
                column_mapping[col] = 'Pretax_Income'
            
            # Income Tax
            elif 'tax' in col_lower and not 'pre-tax' in col_lower and not 'pretax' in col_lower:
                column_mapping[col] = 'Income_Tax'
            
            # Net Income
            elif 'net income' in col_lower or 'net profit' in col_lower or 'net earnings' in col_lower:
                column_mapping[col] = 'Net_Income'
        
        # Rename columns
        std_df.rename(columns=column_mapping, inplace=True)
        
        # Ensure all values are numeric
        for col in std_df.columns:
            if col != 'Year':
                std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
        
        # Sort by year
        if 'Year' in std_df.columns:
            std_df.sort_values('Year', inplace=True)
        
        # Calculate missing metrics if possible
        if 'Revenue' in std_df.columns and 'Cost_of_Revenue' in std_df.columns and 'Gross_Profit' not in std_df.columns:
            std_df['Gross_Profit'] = std_df['Revenue'] - std_df['Cost_of_Revenue']
        
        if 'Gross_Profit' in std_df.columns and 'Operating_Expenses' in std_df.columns and 'EBIT' not in std_df.columns:
            std_df['EBIT'] = std_df['Gross_Profit'] - std_df['Operating_Expenses']
        
        if 'EBIT' in std_df.columns and 'Depreciation_Amortization' in std_df.columns and 'EBITDA' not in std_df.columns:
            std_df['EBITDA'] = std_df['EBIT'] + std_df['Depreciation_Amortization']
        
        if 'EBIT' in std_df.columns and 'Interest_Expense' in std_df.columns and 'Pretax_Income' not in std_df.columns:
            std_df['Pretax_Income'] = std_df['EBIT'] - std_df['Interest_Expense']
        
        if 'Pretax_Income' in std_df.columns and 'Income_Tax' in std_df.columns and 'Net_Income' not in std_df.columns:
            std_df['Net_Income'] = std_df['Pretax_Income'] - std_df['Income_Tax']
        
        # Calculate growth rates
        if 'Revenue' in std_df.columns:
            std_df['Revenue_Growth'] = std_df['Revenue'].pct_change()
        
        if 'EBITDA' in std_df.columns:
            std_df['EBITDA_Margin'] = std_df['EBITDA'] / std_df['Revenue']
        
        if 'Net_Income' in std_df.columns:
            std_df['Net_Margin'] = std_df['Net_Income'] / std_df['Revenue']
        
        return std_df
    
    def standardize_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the balance sheet format.
        
        Args:
            df: Balance sheet DataFrame
            
        Returns:
            Standardized balance sheet DataFrame
        """
        # Create a copy to avoid modifying the original
        std_df = df.copy()
        
        # Identify the year column
        year_col = None
        for col in std_df.columns:
            if isinstance(col, str) and any(x in col.lower() for x in ['year', 'date', 'period']):
                year_col = col
                break
        
        if year_col is None and std_df.shape[1] > 1:
            # Assume the first column is the year/period
            year_col = std_df.columns[0]
        
        # Rename the year column
        if year_col is not None:
            std_df.rename(columns={year_col: 'Year'}, inplace=True)
        
        # Ensure Year column is numeric
        if 'Year' in std_df.columns:
            std_df['Year'] = pd.to_numeric(std_df['Year'], errors='coerce')
        
        # Identify and standardize key metrics
        column_mapping = {}
        
        for col in std_df.columns:
            col_lower = str(col).lower()
            
            # Cash and Cash Equivalents
            if any(x in col_lower for x in ['cash', 'cash equivalent']):
                column_mapping[col] = 'Cash_Equivalents'
            
            # Short-term Investments
            elif any(x in col_lower for x in ['short-term investment', 'short term investment']):
                column_mapping[col] = 'Short_Term_Investments'
            
            # Accounts Receivable
            elif 'receivable' in col_lower or 'ar' == col_lower:
                column_mapping[col] = 'Accounts_Receivable'
            
            # Inventory
            elif 'inventory' in col_lower:
                column_mapping[col] = 'Inventory'
            
            # Other Current Assets
            elif 'other current asset' in col_lower:
                column_mapping[col] = 'Other_Current_Assets'
            
            # Total Current Assets
            elif 'total current asset' in col_lower or 'current asset total' in col_lower:
                column_mapping[col] = 'Total_Current_Assets'
            
            # Property, Plant, and Equipment
            elif any(x in col_lower for x in ['property', 'plant', 'equipment', 'ppe']):
                column_mapping[col] = 'PPE'
            
            # Goodwill
            elif 'goodwill' in col_lower:
                column_mapping[col] = 'Goodwill'
            
            # Intangible Assets
            elif 'intangible' in col_lower:
                column_mapping[col] = 'Intangible_Assets'
            
            # Long-term Investments
            elif any(x in col_lower for x in ['long-term investment', 'long term investment']):
                column_mapping[col] = 'Long_Term_Investments'
            
            # Other Non-current Assets
            elif 'other non-current asset' in col_lower or 'other noncurrent asset' in col_lower:
                column_mapping[col] = 'Other_Noncurrent_Assets'
            
            # Total Non-current Assets
            elif any(x in col_lower for x in ['total non-current asset', 'total noncurrent asset', 'non-current asset total']):
                column_mapping[col] = 'Total_Noncurrent_Assets'
            
            # Total Assets
            elif 'total asset' in col_lower or 'asset total' in col_lower:
                column_mapping[col] = 'Total_Assets'
            
            # Accounts Payable
            elif 'payable' in col_lower or 'ap' == col_lower:
                column_mapping[col] = 'Accounts_Payable'
            
            # Short-term Debt
            elif any(x in col_lower for x in ['short-term debt', 'short term debt', 'current debt']):
                column_mapping[col] = 'Short_Term_Debt'
            
            # Deferred Revenue
            elif 'deferred revenue' in col_lower or 'unearned revenue' in col_lower:
                column_mapping[col] = 'Deferred_Revenue'
            
            # Other Current Liabilities
            elif 'other current liabilit' in col_lower:
                column_mapping[col] = 'Other_Current_Liabilities'
            
            # Total Current Liabilities
            elif 'total current liabilit' in col_lower or 'current liabilit total' in col_lower:
                column_mapping[col] = 'Total_Current_Liabilities'
            
            # Long-term Debt
            elif any(x in col_lower for x in ['long-term debt', 'long term debt', 'noncurrent debt']):
                column_mapping[col] = 'Long_Term_Debt'
            
            # Deferred Tax Liabilities
            elif 'deferred tax' in col_lower:
                column_mapping[col] = 'Deferred_Tax_Liabilities'
            
            # Other Non-current Liabilities
            elif 'other non-current liabilit' in col_lower or 'other noncurrent liabilit' in col_lower:
                column_mapping[col] = 'Other_Noncurrent_Liabilities'
            
            # Total Non-current Liabilities
            elif any(x in col_lower for x in ['total non-current liabilit', 'total noncurrent liabilit']):
                column_mapping[col] = 'Total_Noncurrent_Liabilities'
            
            # Total Liabilities
            elif 'total liabilit' in col_lower or 'liabilit total' in col_lower:
                column_mapping[col] = 'Total_Liabilities'
            
            # Common Stock
            elif 'common stock' in col_lower:
                column_mapping[col] = 'Common_Stock'
            
            # Retained Earnings
            elif 'retained earning' in col_lower:
                column_mapping[col] = 'Retained_Earnings'
            
            # Treasury Stock
            elif 'treasury stock' in col_lower:
                column_mapping[col] = 'Treasury_Stock'
            
            # Other Equity
            elif 'other equity' in col_lower:
                column_mapping[col] = 'Other_Equity'
            
            # Total Shareholders' Equity
            elif any(x in col_lower for x in ['total equity', 'equity total', 'shareholders equity', 'stockholders equity']):
                column_mapping[col] = 'Total_Equity'
        
        # Rename columns
        std_df.rename(columns=column_mapping, inplace=True)
        
        # Ensure all values are numeric
        for col in std_df.columns:
            if col != 'Year':
                std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
        
        # Sort by year
        if 'Year' in std_df.columns:
            std_df.sort_values('Year', inplace=True)
        
        # Calculate missing metrics if possible
        if 'Cash_Equivalents' in std_df.columns and 'Short_Term_Investments' in std_df.columns and 'Accounts_Receivable' in std_df.columns and 'Inventory' in std_df.columns and 'Other_Current_Assets' in std_df.columns and 'Total_Current_Assets' not in std_df.columns:
            std_df['Total_Current_Assets'] = std_df['Cash_Equivalents'] + std_df['Short_Term_Investments'] + std_df['Accounts_Receivable'] + std_df['Inventory'] + std_df['Other_Current_Assets']
        
        if 'PPE' in std_df.columns and 'Goodwill' in std_df.columns and 'Intangible_Assets' in std_df.columns and 'Long_Term_Investments' in std_df.columns and 'Other_Noncurrent_Assets' in std_df.columns and 'Total_Noncurrent_Assets' not in std_df.columns:
            std_df['Total_Noncurrent_Assets'] = std_df['PPE'] + std_df['Goodwill'] + std_df['Intangible_Assets'] + std_df['Long_Term_Investments'] + std_df['Other_Noncurrent_Assets']
        
        if 'Total_Current_Assets' in std_df.columns and 'Total_Noncurrent_Assets' in std_df.columns and 'Total_Assets' not in std_df.columns:
            std_df['Total_Assets'] = std_df['Total_Current_Assets'] + std_df['Total_Noncurrent_Assets']
        
        if 'Accounts_Payable' in std_df.columns and 'Short_Term_Debt' in std_df.columns and 'Deferred_Revenue' in std_df.columns and 'Other_Current_Liabilities' in std_df.columns and 'Total_Current_Liabilities' not in std_df.columns:
            std_df['Total_Current_Liabilities'] = std_df['Accounts_Payable'] + std_df['Short_Term_Debt'] + std_df['Deferred_Revenue'] + std_df['Other_Current_Liabilities']
        
        if 'Long_Term_Debt' in std_df.columns and 'Deferred_Tax_Liabilities' in std_df.columns and 'Other_Noncurrent_Liabilities' in std_df.columns and 'Total_Noncurrent_Liabilities' not in std_df.columns:
            std_df['Total_Noncurrent_Liabilities'] = std_df['Long_Term_Debt'] + std_df['Deferred_Tax_Liabilities'] + std_df['Other_Noncurrent_Liabilities']
        
        if 'Total_Current_Liabilities' in std_df.columns and 'Total_Noncurrent_Liabilities' in std_df.columns and 'Total_Liabilities' not in std_df.columns:
            std_df['Total_Liabilities'] = std_df['Total_Current_Liabilities'] + std_df['Total_Noncurrent_Liabilities']
        
        if 'Common_Stock' in std_df.columns and 'Retained_Earnings' in std_df.columns and 'Treasury_Stock' in std_df.columns and 'Other_Equity' in std_df.columns and 'Total_Equity' not in std_df.columns:
            std_df['Total_Equity'] = std_df['Common_Stock'] + std_df['Retained_Earnings'] + std_df['Treasury_Stock'] + std_df['Other_Equity']
        
        if 'Total_Liabilities' in std_df.columns and 'Total_Equity' in std_df.columns:
            # Check if liabilities + equity = assets
            std_df['Liabilities_Equity'] = std_df['Total_Liabilities'] + std_df['Total_Equity']
        
        return std_df
    
    def standardize_cash_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the cash flow statement format.
        
        Args:
            df: Cash flow statement DataFrame
            
        Returns:
            Standardized cash flow statement DataFrame
        """
        # Create a copy to avoid modifying the original
        std_df = df.copy()
        
        # Identify the year column
        year_col = None
        for col in std_df.columns:
            if isinstance(col, str) and any(x in col.lower() for x in ['year', 'date', 'period']):
                year_col = col
                break
        
        if year_col is None and std_df.shape[1] > 1:
            # Assume the first column is the year/period
            year_col = std_df.columns[0]
        
        # Rename the year column
        if year_col is not None:
            std_df.rename(columns={year_col: 'Year'}, inplace=True)
        
        # Ensure Year column is numeric
        if 'Year' in std_df.columns:
            std_df['Year'] = pd.to_numeric(std_df['Year'], errors='coerce')
        
        # Identify and standardize key metrics
        column_mapping = {}
        
        for col in std_df.columns:
            col_lower = str(col).lower()
            
            # Net Income
            if 'net income' in col_lower or 'net profit' in col_lower or 'net earnings' in col_lower:
                column_mapping[col] = 'Net_Income'
            
            # Depreciation and Amortization
            elif any(x in col_lower for x in ['depreciation', 'amortization', 'd&a']):
                column_mapping[col] = 'Depreciation_Amortization'
            
            # Stock-based Compensation
            elif 'stock' in col_lower and 'compensation' in col_lower:
                column_mapping[col] = 'Stock_Based_Compensation'
            
            # Changes in Working Capital
            elif 'working capital' in col_lower or 'change in working capital' in col_lower:
                column_mapping[col] = 'Change_In_Working_Capital'
            
            # Change in Accounts Receivable
            elif 'change' in col_lower and 'receivable' in col_lower:
                column_mapping[col] = 'Change_In_Accounts_Receivable'
            
            # Change in Inventory
            elif 'change' in col_lower and 'inventory' in col_lower:
                column_mapping[col] = 'Change_In_Inventory'
            
            # Change in Accounts Payable
            elif 'change' in col_lower and 'payable' in col_lower:
                column_mapping[col] = 'Change_In_Accounts_Payable'
            
            # Change in Deferred Revenue
            elif 'change' in col_lower and 'deferred revenue' in col_lower:
                column_mapping[col] = 'Change_In_Deferred_Revenue'
            
            # Other Operating Activities
            elif 'other operating' in col_lower:
                column_mapping[col] = 'Other_Operating_Activities'
            
            # Net Cash from Operating Activities
            elif any(x in col_lower for x in ['net cash from operating', 'net cash provided by operating', 'operating cash flow']):
                column_mapping[col] = 'Net_Cash_From_Operating'
            
            # Capital Expenditures
            elif any(x in col_lower for x in ['capital expenditure', 'capex']):
                column_mapping[col] = 'Capital_Expenditures'
            
            # Acquisitions
            elif 'acquisition' in col_lower:
                column_mapping[col] = 'Acquisitions'
            
            # Purchases of Investments
            elif 'purchase' in col_lower and 'investment' in col_lower:
                column_mapping[col] = 'Purchases_Of_Investments'
            
            # Sales of Investments
            elif 'sale' in col_lower and 'investment' in col_lower:
                column_mapping[col] = 'Sales_Of_Investments'
            
            # Other Investing Activities
            elif 'other investing' in col_lower:
                column_mapping[col] = 'Other_Investing_Activities'
            
            # Net Cash from Investing Activities
            elif any(x in col_lower for x in ['net cash from investing', 'net cash used in investing']):
                column_mapping[col] = 'Net_Cash_From_Investing'
            
            # Debt Issuance
            elif 'debt issuance' in col_lower or 'issuance of debt' in col_lower:
                column_mapping[col] = 'Debt_Issuance'
            
            # Debt Repayment
            elif 'debt repayment' in col_lower or 'repayment of debt' in col_lower:
                column_mapping[col] = 'Debt_Repayment'
            
            # Stock Issuance
            elif 'stock issuance' in col_lower or 'issuance of stock' in col_lower:
                column_mapping[col] = 'Stock_Issuance'
            
            # Stock Repurchase
            elif 'stock repurchase' in col_lower or 'repurchase of stock' in col_lower:
                column_mapping[col] = 'Stock_Repurchase'
            
            # Dividends Paid
            elif 'dividend' in col_lower:
                column_mapping[col] = 'Dividends_Paid'
            
            # Other Financing Activities
            elif 'other financing' in col_lower:
                column_mapping[col] = 'Other_Financing_Activities'
            
            # Net Cash from Financing Activities
            elif any(x in col_lower for x in ['net cash from financing', 'net cash used in financing']):
                column_mapping[col] = 'Net_Cash_From_Financing'
            
            # Effect of Exchange Rate Changes
            elif 'exchange rate' in col_lower:
                column_mapping[col] = 'Effect_Of_Exchange_Rate'
            
            # Net Change in Cash
            elif 'net change in cash' in col_lower or 'change in cash' in col_lower:
                column_mapping[col] = 'Net_Change_In_Cash'
            
            # Beginning Cash Balance
            elif 'beginning cash' in col_lower or 'cash at beginning' in col_lower:
                column_mapping[col] = 'Beginning_Cash_Balance'
            
            # Ending Cash Balance
            elif 'ending cash' in col_lower or 'cash at end' in col_lower:
                column_mapping[col] = 'Ending_Cash_Balance'
            
            # Free Cash Flow
            elif 'free cash flow' in col_lower:
                column_mapping[col] = 'Free_Cash_Flow'
        
        # Rename columns
        std_df.rename(columns=column_mapping, inplace=True)
        
        # Ensure all values are numeric
        for col in std_df.columns:
            if col != 'Year':
                std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
        
        # Sort by year
        if 'Year' in std_df.columns:
            std_df.sort_values('Year', inplace=True)
        
        # Calculate missing metrics if possible
        if 'Net_Cash_From_Operating' in std_df.columns and 'Capital_Expenditures' in std_df.columns and 'Free_Cash_Flow' not in std_df.columns:
            std_df['Free_Cash_Flow'] = std_df['Net_Cash_From_Operating'] - std_df['Capital_Expenditures']
        
        if 'Net_Cash_From_Operating' in std_df.columns and 'Net_Cash_From_Investing' in std_df.columns and 'Net_Cash_From_Financing' in std_df.columns and 'Effect_Of_Exchange_Rate' in std_df.columns and 'Net_Change_In_Cash' not in std_df.columns:
            std_df['Net_Change_In_Cash'] = std_df['Net_Cash_From_Operating'] + std_df['Net_Cash_From_Investing'] + std_df['Net_Cash_From_Financing'] + std_df['Effect_Of_Exchange_Rate']
        
        if 'Beginning_Cash_Balance' in std_df.columns and 'Net_Change_In_Cash' in std_df.columns and 'Ending_Cash_Balance' not in std_df.columns:
            std_df['Ending_Cash_Balance'] = std_df['Beginning_Cash_Balance'] + std_df['Net_Change_In_Cash']
        
        return std_df
    
    def standardize_data(self) -> Dict[str, pd.DataFrame]:
        """
        Standardize all financial statements.
        
        Returns:
            Dictionary mapping statement types to standardized DataFrames
        """
        standardized_data = {}
        
        if self.income_statement is not None:
            standardized_data["income_statement"] = self.standardize_income_statement(self.income_statement)
        
        if self.balance_sheet is not None:
            standardized_data["balance_sheet"] = self.standardize_balance_sheet(self.balance_sheet)
        
        if self.cash_flow is not None:
            standardized_data["cash_flow"] = self.standardize_cash_flow(self.cash_flow)
        
        return standardized_data
    
    def combine_statements(self) -> pd.DataFrame:
        """
        Combine all financial statements into a single DataFrame.
        
        Returns:
            Combined DataFrame with all financial data
        """
        if self.income_statement is None or self.balance_sheet is None or self.cash_flow is None:
            raise ValueError("All financial statements must be loaded before combining")
        
        # Standardize the data
        std_data = self.standardize_data()
        
        # Get the year ranges
        income_years = std_data["income_statement"]["Year"].tolist()
        balance_years = std_data["balance_sheet"]["Year"].tolist()
        cash_flow_years = std_data["cash_flow"]["Year"].tolist()
        
        # Find the common years
        common_years = sorted(set(income_years).intersection(set(balance_years)).intersection(set(cash_flow_years)))
        
        if not common_years:
            raise ValueError("No common years found across financial statements")
        
        # Create a new DataFrame with the common years
        combined_df = pd.DataFrame({"Year": common_years})
        
        # Add income statement data
        income_df = std_data["income_statement"]
        for col in income_df.columns:
            if col != "Year":
                combined_df[f"IS_{col}"] = income_df.set_index("Year").loc[common_years, col].values
        
        # Add balance sheet data
        balance_df = std_data["balance_sheet"]
        for col in balance_df.columns:
            if col != "Year":
                combined_df[f"BS_{col}"] = balance_df.set_index("Year").loc[common_years, col].values
        
        # Add cash flow data
        cash_flow_df = std_data["cash_flow"]
        for col in cash_flow_df.columns:
            if col != "Year":
                combined_df[f"CF_{col}"] = cash_flow_df.set_index("Year").loc[common_years, col].values
        
        self.combined_data = combined_df
        
        # Set the historical end year
        self.historical_end_year = max(common_years)
        
        return combined_df
    
    def extend_forecast_years(self, forecast_years: int = 5) -> pd.DataFrame:
        """
        Extend the combined data with forecast years.
        
        Args:
            forecast_years: Number of years to forecast
            
        Returns:
            DataFrame with historical and forecast years
        """
        if self.combined_data is None:
            raise ValueError("Combined data must be created before extending forecast years")
        
        if self.historical_end_year is None:
            raise ValueError("Historical end year must be set before extending forecast years")
        
        # Create a copy of the combined data
        extended_df = self.combined_data.copy()
        
        # Create forecast years
        forecast_data = []
        for i in range(1, forecast_years + 1):
            forecast_year = self.historical_end_year + i
            forecast_row = {"Year": forecast_year}
            forecast_data.append(forecast_row)
        
        # Create a DataFrame for forecast years
        forecast_df = pd.DataFrame(forecast_data)
        
        # Combine historical and forecast data
        combined_df = pd.concat([extended_df, forecast_df], ignore_index=True)
        
        return combined_df
    
    def calculate_historical_metrics(self) -> Dict[str, float]:
        """
        Calculate average historical metrics for use in forecasting.
        
        Returns:
            Dictionary of average metrics
        """
        if self.combined_data is None:
            raise ValueError("Combined data must be created before calculating historical metrics")
        
        if self.historical_end_year is None:
            raise ValueError("Historical end year must be set before calculating historical metrics")
        
        # Get historical data
        historical_data = self.combined_data[self.combined_data["Year"] <= self.historical_end_year]
        
        metrics = {}
        
        # Revenue Growth
        if "IS_Revenue" in historical_data.columns:
            revenue_growth = historical_data["IS_Revenue"].pct_change().dropna()
            metrics["avg_revenue_growth"] = revenue_growth.mean()
        
        # EBITDA Margin
        if "IS_EBITDA" in historical_data.columns and "IS_Revenue" in historical_data.columns:
            ebitda_margin = historical_data["IS_EBITDA"] / historical_data["IS_Revenue"]
            metrics["avg_ebitda_margin"] = ebitda_margin.mean()
        
        # Net Margin
        if "IS_Net_Income" in historical_data.columns and "IS_Revenue" in historical_data.columns:
            net_margin = historical_data["IS_Net_Income"] / historical_data["IS_Revenue"]
            metrics["avg_net_margin"] = net_margin.mean()
        
        # Tax Rate
        if "IS_Income_Tax" in historical_data.columns and "IS_Pretax_Income" in historical_data.columns:
            tax_rate = historical_data["IS_Income_Tax"] / historical_data["IS_Pretax_Income"]
            metrics["avg_tax_rate"] = tax_rate.mean()
        
        # Depreciation & Amortization to Revenue
        if "IS_Depreciation_Amortization" in historical_data.columns and "IS_Revenue" in historical_data.columns:
            da_to_revenue = historical_data["IS_Depreciation_Amortization"] / historical_data["IS_Revenue"]
            metrics["avg_da_to_revenue"] = da_to_revenue.mean()
        
        # Capital Expenditures to Revenue
        if "CF_Capital_Expenditures" in historical_data.columns and "IS_Revenue" in historical_data.columns:
            capex_to_revenue = historical_data["CF_Capital_Expenditures"] / historical_data["IS_Revenue"]
            metrics["avg_capex_to_revenue"] = capex_to_revenue.mean()
        
        # Working Capital to Revenue
        if "CF_Change_In_Working_Capital" in historical_data.columns and "IS_Revenue" in historical_data.columns:
            wc_to_revenue = historical_data["CF_Change_In_Working_Capital"] / historical_data["IS_Revenue"]
            metrics["avg_wc_to_revenue"] = wc_to_revenue.mean()
        
        # Days Sales Outstanding (DSO)
        if "BS_Accounts_Receivable" in historical_data.columns and "IS_Revenue" in historical_data.columns:
            dso = (historical_data["BS_Accounts_Receivable"] / historical_data["IS_Revenue"]) * 365
            metrics["avg_dso"] = dso.mean()
        
        # Days Inventory Outstanding (DIO)
        if "BS_Inventory" in historical_data.columns and "IS_Cost_of_Revenue" in historical_data.columns:
            dio = (historical_data["BS_Inventory"] / historical_data["IS_Cost_of_Revenue"]) * 365
            metrics["avg_dio"] = dio.mean()
        
        # Days Payable Outstanding (DPO)
        if "BS_Accounts_Payable" in historical_data.columns and "IS_Cost_of_Revenue" in historical_data.columns:
            dpo = (historical_data["BS_Accounts_Payable"] / historical_data["IS_Cost_of_Revenue"]) * 365
            metrics["avg_dpo"] = dpo.mean()
        
        # Customer Acquisition Cost (CAC) - for software companies
        if "IS_Sales_Marketing" in historical_data.columns:
            # This is a simplification; in reality, you would need customer count data
            metrics["avg_sales_marketing_to_revenue"] = (historical_data["IS_Sales_Marketing"] / historical_data["IS_Revenue"]).mean()
        
        return metrics
    
    def get_forecast_years(self, data: pd.DataFrame) -> List[int]:
        """
        Identify which years in the data are forecast years.
        
        Args:
            data: DataFrame containing financial data
            
        Returns:
            List of forecast years
        """
        if self.historical_end_year is None:
            raise ValueError("Historical end year must be set before identifying forecast years")
        
        return sorted(data[data["Year"] > self.historical_end_year]["Year"].tolist())
