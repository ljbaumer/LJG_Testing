"""
Pricing Engine Module - Implements IPO pricing algorithms.

This module contains the implementation of the traditional bookbuilding approach
for IPO pricing. It calculates the demand curve, determines the clearing price,
allocates shares to investors, and calculates key IPO metrics.
"""

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from typing import Dict, Tuple, List, Optional  # For type hints


class TraditionalPricingEngine:
    """
    Implements a traditional bookbuilding approach for IPO pricing.
    
    In the traditional bookbuilding approach, the underwriter (investment bank) collects
    bids from investors and determines a final offer price based on demand. This class
    simulates that process by:
    1. Building a demand curve from investor bids
    2. Finding the clearing price where demand meets supply
    3. Applying typical IPO pricing adjustments (slight discount)
    4. Allocating shares to eligible investors
    5. Calculating key IPO performance metrics
    """
    
    def __init__(
        self,
        price_range: Tuple[float, float] = (20.0, 25.0),
        total_shares: int = 10_000_000,
    ):
        """
        Initialize the pricing engine with IPO parameters.
        
        Args:
            price_range: Initial price range for the IPO (min, max)
                         This is the range that would be published in the IPO prospectus
            total_shares: Total number of shares available in the IPO
                          This is the total size of the offering
        """
        self.price_range = price_range
        self.total_shares = total_shares
        # Extract min and max prices for easier access
        self.min_price, self.max_price = price_range
    
    def calculate_demand_curve(self, investor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the demand curve based on investor bids.
        
        The demand curve shows the cumulative number of shares demanded at each price point.
        It's created by sorting investors by their bid prices (highest to lowest) and
        calculating the cumulative sum of shares requested.
        
        Args:
            investor_data: DataFrame with investor bids and quantities
            
        Returns:
            DataFrame with cumulative demand at different price points
        """
        # Sort by bid price in descending order (highest bids first)
        sorted_data = investor_data.sort_values('bid_price', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative demand (running total of shares requested)
        # This creates the demand curve - at each price point, how many total shares are demanded
        sorted_data['cumulative_shares'] = sorted_data['shares_requested'].cumsum()
        
        return sorted_data
    
    def determine_clearing_price(self, demand_curve: pd.DataFrame) -> float:
        """
        Determine the clearing price where demand meets supply.
        
        The clearing price is the price at which the total demand equals the total supply.
        In IPO pricing, this is the theoretical price where all shares would be sold.
        However, in practice, the final price is often set slightly below this level
        to ensure full subscription and positive first-day performance.
        
        Args:
            demand_curve: DataFrame with cumulative demand
            
        Returns:
            Final IPO price after adjustments
        """
        # Find the highest price where cumulative demand exceeds total shares
        # This is the price point where all shares would be sold (clearing price)
        clearing_idx = demand_curve[demand_curve['cumulative_shares'] >= self.total_shares].index.min()
        
        if pd.isna(clearing_idx):
            # If demand never exceeds supply (undersubscribed IPO), use the minimum price
            return self.min_price
        
        # Get the actual clearing price from the demand curve
        clearing_price = demand_curve.loc[clearing_idx, 'bid_price']
        
        # In traditional IPO pricing, the final price is often set slightly below clearing
        # to ensure full subscription and positive first-day performance
        # This creates the "IPO discount" that often leads to first-day "pops"
        discount_factor = 0.95  # 5% discount
        final_price = max(clearing_price * discount_factor, self.min_price)
        
        # Round to nearest $0.25 (common in IPO pricing)
        # IPO prices are typically set at these quarter-dollar increments
        final_price = round(final_price * 4) / 4
        
        # Ensure price is within the initial range
        # IPO prices almost always stay within the initial range published in the prospectus
        final_price = max(min(final_price, self.max_price), self.min_price)
        
        return final_price
    
    def allocate_shares(
        self, 
        investor_data: pd.DataFrame, 
        final_price: float
    ) -> pd.DataFrame:
        """
        Allocate shares to investors based on the final price.
        
        This method implements a pro-rata allocation strategy where:
        1. Only investors who bid at or above the final price are eligible
        2. If oversubscribed, each eligible investor gets a proportional allocation
        3. Allocations are rounded to the nearest 100 shares
        4. Any remaining shares due to rounding are allocated to the largest investors
        
        Args:
            investor_data: DataFrame with investor bids and quantities
            final_price: Final IPO price
            
        Returns:
            DataFrame with share allocations
        """
        # Create a copy to avoid modifying the original data
        allocation_data = investor_data.copy()
        
        # Step 1: Determine eligible investors (those willing to pay at least the final price)
        # In a real IPO, only investors who bid at or above the final price receive allocations
        allocation_data['eligible'] = allocation_data['bid_price'] >= final_price
        
        # Step 2: Calculate total eligible demand
        total_eligible_demand = allocation_data.loc[allocation_data['eligible'], 'shares_requested'].sum()
        
        # Step 3: Calculate allocation ratio (may be less than 1 if oversubscribed)
        # This is the pro-rata allocation factor - each investor gets this percentage of their request
        allocation_ratio = min(1.0, self.total_shares / total_eligible_demand)
        
        # Step 4: Calculate allocated shares
        # Initialize all allocations to zero
        allocation_data['shares_allocated'] = 0
        
        # Allocate shares to eligible investors based on the allocation ratio
        allocation_data.loc[allocation_data['eligible'], 'shares_allocated'] = (
            allocation_data.loc[allocation_data['eligible'], 'shares_requested'] * allocation_ratio
        ).astype(int)
        
        # Step 5: Round allocations to nearest 100 shares (common in IPO allocations)
        allocation_data['shares_allocated'] = (
            np.round(allocation_data['shares_allocated'], -2)
        ).astype(int)
        
        # Step 6: Handle any remaining shares due to rounding
        remaining_shares = self.total_shares - allocation_data['shares_allocated'].sum()
        
        if remaining_shares > 0:
            # Allocate remaining shares to largest eligible investors
            # This is a common approach to handle rounding differences
            eligible_investors = allocation_data[allocation_data['eligible']].sort_values(
                'shares_requested', ascending=False  # Sort by size (largest first)
            )
            
            # Distribute 100 shares at a time to largest investors until all allocated
            share_increment = 100  # Standard lot size
            idx = 0
            while remaining_shares >= share_increment and idx < len(eligible_investors):
                investor_idx = eligible_investors.index[idx % len(eligible_investors)]
                allocation_data.loc[investor_idx, 'shares_allocated'] += share_increment
                remaining_shares -= share_increment
                idx += 1
        
        return allocation_data
    
    def calculate_metrics(
        self, 
        allocation_data: pd.DataFrame, 
        final_price: float
    ) -> Dict:
        """
        Calculate key metrics for the IPO.
        
        This method calculates important performance metrics that would be analyzed
        in a real IPO, including:
        - Total demand and eligible demand
        - Capital raised
        - Oversubscription ratios
        - "Money left on table" (potential first-day pop)
        - Allocation statistics
        
        Args:
            allocation_data: DataFrame with share allocations
            final_price: Final IPO price
            
        Returns:
            Dictionary with key metrics
        """
        # Calculate total demand (all investor requests)
        total_demand = allocation_data['shares_requested'].sum()
        
        # Calculate eligible demand (only from investors bidding at or above final price)
        eligible_demand = allocation_data.loc[
            allocation_data['bid_price'] >= final_price, 'shares_requested'
        ].sum()
        
        # Calculate total shares allocated
        total_allocated = allocation_data['shares_allocated'].sum()
        
        # Calculate capital raised (total proceeds from the IPO)
        capital_raised = total_allocated * final_price
        
        # Calculate oversubscription ratio (total demand / total supply)
        # This shows how many times the offering was oversubscribed
        oversubscription_ratio = total_demand / self.total_shares
        
        # Calculate eligible oversubscription ratio (eligible demand / total supply)
        # This shows how many times the offering was oversubscribed by investors
        # willing to pay at least the final price
        eligible_oversubscription_ratio = eligible_demand / self.total_shares
        
        # Calculate average bid price across all investors
        average_bid_price = allocation_data['bid_price'].mean()
        
        # Estimate "money left on table" (potential first-day pop)
        # This is a rough estimate based on the difference between average bid and final price
        # It represents the potential additional capital that could have been raised
        # if the IPO had been priced at the average bid price
        money_left = (average_bid_price - final_price) * total_allocated
        
        # Calculate percentage of investors who received allocations
        pct_investors_allocated = (allocation_data['shares_allocated'] > 0).mean() * 100
        
        # Return all metrics in a dictionary
        return {
            'final_price': final_price,
            'total_shares_available': self.total_shares,
            'total_shares_allocated': total_allocated,
            'total_demand': total_demand,
            'eligible_demand': eligible_demand,
            'capital_raised': capital_raised,
            'oversubscription_ratio': oversubscription_ratio,
            'eligible_oversubscription_ratio': eligible_oversubscription_ratio,
            'average_bid_price': average_bid_price,
            'estimated_money_left': money_left,
            'pct_investors_allocated': pct_investors_allocated
        }
    
    def run_pricing_process(self, investor_data: pd.DataFrame) -> Dict:
        """
        Run the complete IPO pricing process.
        
        This is the main method to call from outside the class. It executes the entire
        IPO pricing workflow:
        1. Calculate the demand curve
        2. Determine the final price
        3. Allocate shares to investors
        4. Calculate performance metrics
        
        Args:
            investor_data: DataFrame with investor bids and quantities
            
        Returns:
            Dictionary with pricing results and metrics containing:
            - demand_curve: DataFrame with the calculated demand curve
            - allocation_data: DataFrame with the final share allocations
            - metrics: Dictionary with key IPO performance metrics
        """
        # Step 1: Calculate demand curve
        demand_curve = self.calculate_demand_curve(investor_data)
        
        # Step 2: Determine final price
        final_price = self.determine_clearing_price(demand_curve)
        
        # Step 3: Allocate shares
        allocation_data = self.allocate_shares(investor_data, final_price)
        
        # Step 4: Calculate metrics
        metrics = self.calculate_metrics(allocation_data, final_price)
        
        # Return all results in a dictionary
        return {
            'demand_curve': demand_curve,
            'allocation_data': allocation_data,
            'metrics': metrics
        }


# This conditional ensures that the code below only runs when this script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    # Example usage - this demonstrates how to use the class
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import data_generator
    sys.path.append(str(Path(__file__).parent.parent))
    from ipo_pricing_tool.data_generator import InvestorDataGenerator
    
    # Step 1: Generate mock investor data
    generator = InvestorDataGenerator(
        num_investors=100,
        price_range=(20.0, 25.0),
        total_shares=10_000_000,
        seed=42  # Using a fixed seed for reproducible results
    )
    
    data = generator.generate_data()
    
    # Step 2: Run the IPO pricing process
    pricing_engine = TraditionalPricingEngine(
        price_range=(20.0, 25.0),
        total_shares=10_000_000
    )
    
    results = pricing_engine.run_pricing_process(data)
    
    # Step 3: Print key results
    print(f"Final IPO Price: ${results['metrics']['final_price']:.2f}")
    print(f"Capital Raised: ${results['metrics']['capital_raised']:,.2f}")
    print(f"Oversubscription Ratio: {results['metrics']['oversubscription_ratio']:.2f}x")
