"""
Data Generator Module - Generates mock investor data for IPO pricing simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class InvestorDataGenerator:
    """
    Generates mock investor data for IPO pricing simulations.
    Focuses on creating realistic tech IPO investor behavior patterns.
    """

    def __init__(
        self,
        num_investors: int = 100,
        price_range: Tuple[float, float] = (20.0, 25.0),
        total_shares: int = 10_000_000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the data generator with IPO parameters.

        Args:
            num_investors: Number of investors to generate
            price_range: Initial price range for the IPO (min, max)
            total_shares: Total number of shares available in the IPO
            seed: Random seed for reproducibility
        """
        self.num_investors = num_investors
        self.price_range = price_range
        self.total_shares = total_shares
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def generate_tech_ipo_data(self) -> pd.DataFrame:
        """
        Generate mock investor data for a tech IPO.
        
        Returns:
            DataFrame with investor bids and quantities
        """
        # Generate investor IDs
        investor_ids = [f"Investor_{i+1}" for i in range(self.num_investors)]
        
        # Generate bid prices with a distribution typical for tech IPOs
        # - Most bids cluster around the price range
        # - Some outliers both above and below
        # - Slight positive skew (more investors willing to pay premium)
        
        min_price, max_price = self.price_range
        mean_price = (min_price + max_price) / 2
        std_dev = (max_price - min_price) / 3
        
        # Generate base prices with normal distribution
        base_prices = np.random.normal(mean_price, std_dev, self.num_investors)
        
        # Add some skew for tech IPO characteristics (more premium bids)
        skew_factor = np.random.exponential(2, self.num_investors) - 1
        bid_prices = base_prices + skew_factor
        
        # Ensure minimum bid price is reasonable (not negative)
        bid_prices = np.maximum(bid_prices, min_price * 0.7)
        
        # Round to 2 decimal places
        bid_prices = np.round(bid_prices, 2)
        
        # Generate quantities with larger investors requesting more shares
        # - Log-normal distribution to represent mix of small and large investors
        # - Correlation between price and quantity (bigger investors often bid higher)
        
        # Base quantities with log-normal distribution
        mean_quantity = self.total_shares / (self.num_investors * 2)  # Aim for 2x oversubscription
        base_quantities = np.random.lognormal(
            mean=np.log(mean_quantity), 
            sigma=0.8, 
            size=self.num_investors
        )
        
        # Add some correlation with price (higher bids tend to request more shares)
        price_normalized = (bid_prices - min_price) / (max_price - min_price)
        quantity_adjustment = 1 + 0.5 * price_normalized
        quantities = base_quantities * quantity_adjustment
        
        # Round to nearest 100 shares
        quantities = np.round(quantities, -2).astype(int)
        
        # Ensure minimum quantity is 100 shares
        quantities = np.maximum(quantities, 100)
        
        # Create DataFrame
        data = pd.DataFrame({
            'investor_id': investor_ids,
            'bid_price': bid_prices,
            'shares_requested': quantities
        })
        
        return data
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate mock investor data based on the configured parameters.
        Currently just calls generate_tech_ipo_data() but could be extended
        to support different IPO types.
        
        Returns:
            DataFrame with investor bids and quantities
        """
        return self.generate_tech_ipo_data()


if __name__ == "__main__":
    # Example usage
    generator = InvestorDataGenerator(
        num_investors=100,
        price_range=(20.0, 25.0),
        total_shares=10_000_000,
        seed=42
    )
    
    data = generator.generate_data()
    print(f"Generated data for {len(data)} investors")
    print(data.head())
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Average Bid Price: ${data['bid_price'].mean():.2f}")
    print(f"Total Shares Requested: {data['shares_requested'].sum():,}")
    print(f"Oversubscription Ratio: {data['shares_requested'].sum() / generator.total_shares:.2f}x")
