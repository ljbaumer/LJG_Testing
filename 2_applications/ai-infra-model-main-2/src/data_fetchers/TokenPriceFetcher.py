import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests

DEFAULT_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/models")
DEFAULT_TIMEOUT_SEC = float(os.getenv("PRICE_TIMEOUT_SEC", "5"))

MODEL_PRICE_ARCHIVE_DIR = "data/model_price_archive"

@dataclass
class ModelPricing:
    prompt_price: float  # USD per token
    completion_price: float
    max_completion_tokens: Optional[int]

class TokenPriceFetcher:
    def __init__(self, price_csv_path: Optional[str] = None):
        """
        Initialize the TokenPriceFetcher.
        
        Args:
            price_csv_path: Optional path to an existing price CSV file.
                           If provided, prices will be loaded from this file.
        """
        self.api_url = DEFAULT_API_URL
        self.prices_df = None

        # Load prices from CSV if provided
        if price_csv_path is not None:
            self.load_prices_from_csv(price_csv_path)

    def load_prices_from_csv(self, csv_path: str) -> None:
        """
        Load model prices from an existing CSV file.
        
        Args:
            csv_path: Path to the CSV file containing model pricing information
        """
        self.prices_df = pd.read_csv(csv_path)
        print(f"Loaded prices from {csv_path}")

    def create_prices_csv(self, model_data: List[Dict], filename: str) -> str:
        """Creates a DataFrame from model pricing data and saves it to a CSV file.
        
        Args:
            model_data: List of dictionaries containing model pricing information
        Returns:
            str: Path to the CSV file containing today's prices
        """
        self.prices_df = pd.DataFrame(model_data)
        os.makedirs(MODEL_PRICE_ARCHIVE_DIR, exist_ok=True) # create the directory if it's doesn't already exist
        self.prices_df.to_csv(filename, index=False)
        return filename

    def get_model_prices(self) -> str:
        """Fetches current token prices for all models from OpenRouter API.
        
        Returns:
            str: Path to the CSV file containing today's prices
        """
        today = pd.Timestamp.now().strftime("%Y%m%d")
        filename = f'{MODEL_PRICE_ARCHIVE_DIR}/model_prices_{today}.csv'

        # Today's price file doesn't exist so we need to pull the data
        if not os.path.exists(filename):
            try:
                response = requests.get(self.api_url, timeout=DEFAULT_TIMEOUT_SEC)
                response.raise_for_status()
                data = response.json()
                model_data = []
                for model in data.get("data", []):
                    model_data.append({
                        'model_id': model.get("id"),
                        'prompt_price': float(model["pricing"]["prompt"]),
                        'completion_price': float(model["pricing"]["completion"]),
                        'context_length': model.get("context_length"),
                        'max_completion_tokens': model.get("max_completion_tokens"),
                        'date': pd.Timestamp.now().date()
                    })
                if not model_data:
                    raise ValueError("Empty pricing data from API")
                self.create_prices_csv(model_data, filename)
                print(f"Created new price file for {today}")
            except Exception as e:
                # Fallback to most recent archived file
                print(f"Failed to fetch model prices ({e}). Falling back to most recent archive.")
                candidates = sorted(glob.glob(f"{MODEL_PRICE_ARCHIVE_DIR}/model_prices_*.csv"))
                if not candidates:
                    raise
                fallback = candidates[-1]
                print(f"Using fallback price file: {fallback}")
                filename = fallback
        else:
            print(f"Price file for {today} already exists, using that file!")

        return filename

    def get_model_token_prices(self, model_name: str, csv_path: Optional[str] = None) -> tuple[float, float]:
        """
        Get input and output token prices for a specific model.
        
        Args:
            model_name: Name/ID of the model
            csv_path: Optional path to CSV file. If None, uses cached DataFrame or loads latest.
            
        Returns:
            Tuple of (input_price, output_price) as floats
            
        Raises:
            ValueError: If the model is not found in the price data
        """
        # Load prices if needed
        if self.prices_df is None:
            if csv_path is None:
                csv_path = self.get_model_prices()
            self.prices_df = pd.read_csv(csv_path)

        # Find the model
        model_prices = self.prices_df[self.prices_df['model_id'] == model_name]
        if len(model_prices) == 0:
            raise ValueError(f"Unknown model: {model_name}")

        # Return the prices as a tuple
        return (
            float(model_prices['prompt_price'].iloc[0]),
            float(model_prices['completion_price'].iloc[0])
        )

if __name__ == "__main__":
    client = TokenPriceFetcher()
    price_path = client.get_model_prices()
    price_df = pd.read_csv(price_path)
    print("\nDataFrame of model prices:")
    print(price_df)
