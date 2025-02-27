"""
Data Generator Module - Generates mock investor data for IPO pricing simulations.

This module creates realistic mock data that simulates how investors would bid in an IPO.
It generates investor IDs, bid prices, and share quantities that follow distributions
typical of real-world tech IPOs.
"""

import numpy as np  # For numerical operations and random number generation
import pandas as pd  # For data manipulation and analysis
from typing import Dict, Tuple, Optional  # For type hints


class InvestorDataGenerator:
    """
    Generates mock investor data for IPO pricing simulations.
    
    This class creates realistic mock data that simulates how investors would bid in an IPO.
    It focuses on creating realistic tech IPO investor behavior patterns, including:
    - A mix of institutional and retail investors
    - Varying bid prices around the initial price range
    - Correlation between investor size and bid price
    - Log-normal distribution of requested share quantities
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
            seed: Random seed for reproducibility (allows generating the same data repeatedly)
        """
        self.num_investors = num_investors
        self.price_range = price_range
        self.total_shares = total_shares
        
        # Set random seed if provided
        # This ensures that the same "random" data is generated each time if the same seed is used
        if seed is not None:
            np.random.seed(seed)
    
    def generate_tech_ipo_data(self) -> pd.DataFrame:
        """
        Generate mock investor data for a tech IPO.
        
        This method creates a realistic dataset that simulates how investors would bid
        in a tech IPO. It generates:
        1. Unique investor IDs
        2. Bid prices that follow a distribution typical for tech IPOs
        3. Share quantities that follow a log-normal distribution (many small investors,
           fewer large investors)
        
        Returns:
            DataFrame with investor bids and quantities with columns:
            - investor_id: Unique identifier for each investor
            - bid_price: The price per share the investor is willing to pay
            - shares_requested: The number of shares the investor wants to purchase
        """
        # Step 1: Generate investor IDs (simple sequential naming)
        investor_ids = [f"Investor_{i+1}" for i in range(self.num_investors)]
        
        # Step 2: Generate bid prices with a distribution typical for tech IPOs
        # - Most bids cluster around the price range
        # - Some outliers both above and below
        # - Slight positive skew (more investors willing to pay premium)
        
        min_price, max_price = self.price_range
        mean_price = (min_price + max_price) / 2  # Center of the price range
        std_dev = (max_price - min_price) / 3  # Standard deviation based on range width
        
        # Generate base prices with normal distribution (bell curve)
        base_prices = np.random.normal(mean_price, std_dev, self.num_investors)
        
        # Add some skew for tech IPO characteristics (more premium bids)
        # Exponential distribution creates a right-skewed effect (more high values)
        skew_factor = np.random.exponential(2, self.num_investors) - 1
        bid_prices = base_prices + skew_factor
        
        # Ensure minimum bid price is reasonable (not negative or too low)
        # In real IPOs, some investors might bid below the range to get cheaper shares
        bid_prices = np.maximum(bid_prices, min_price * 0.7)
        
        # Round to 2 decimal places (realistic for dollar prices)
        bid_prices = np.round(bid_prices, 2)
        
        # Step 3: Generate quantities with larger investors requesting more shares
        # - Log-normal distribution to represent mix of small and large investors
        # - Correlation between price and quantity (bigger investors often bid higher)
        
        # Base quantities with log-normal distribution
        # Log-normal creates a right-skewed distribution (many small values, few large values)
        # This simulates the real world where there are many small investors and few large ones
        mean_quantity = self.total_shares / (self.num_investors * 2)  # Aim for 2x oversubscription
        base_quantities = np.random.lognormal(
            mean=np.log(mean_quantity),  # Mean of the underlying normal distribution
            sigma=0.8,  # Standard deviation of the underlying normal distribution
            size=self.num_investors
        )
        
        # Add some correlation with price (higher bids tend to request more shares)
        # This simulates that institutional investors often bid higher and request more shares
        price_normalized = (bid_prices - min_price) / (max_price - min_price)  # Scale to 0-1 range
        quantity_adjustment = 1 + 0.5 * price_normalized  # 50% more shares for highest bidders
        quantities = base_quantities * quantity_adjustment
        
        # Round to nearest 100 shares (realistic for IPO allocations)
        quantities = np.round(quantities, -2).astype(int)
        
        # Ensure minimum quantity is 100 shares (common minimum in real IPOs)
        quantities = np.maximum(quantities, 100)
        
        # Step 4: Create DataFrame with all the generated data
        data = pd.DataFrame({
            'investor_id': investor_ids,
            'bid_price': bid_prices,
            'shares_requested': quantities
        })
        
        return data
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate mock investor data based on the configured parameters.
        
        This is the main method to call from outside the class. Currently it just calls
        generate_tech_ipo_data() but could be extended to support different IPO types
        (e.g., traditional industries, biotech, etc.) with different bidding patterns.
        
        Returns:
            DataFrame with investor bids and quantities
        """
        return self.generate_tech_ipo_data()


# This conditional ensures that the code below only runs when this script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    # Example usage - this demonstrates how to use the class
    generator = InvestorDataGenerator(
        num_investors=100,
        price_range=(20.0, 25.0),
        total_shares=10_000_000,
        seed=42  # Using a fixed seed for reproducible results
    )
    
    # Generate the mock data
    data = generator.generate_data()
    print(f"Generated data for {len(data)} investors")
    print(data.head())  # Display the first 5 rows
    
    # Print some statistics about the generated data
    print("\nData Statistics:")
    print(f"Average Bid Price: ${data['bid_price'].mean():.2f}")
    print(f"Total Shares Requested: {data['shares_requested'].sum():,}")
    # Oversubscription ratio is a key metric in IPOs - it shows how many times the offering was oversubscribed
    print(f"Oversubscription Ratio: {data['shares_requested'].sum() / generator.total_shares:.2f}x")
