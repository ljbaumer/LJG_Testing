"""
Pricing Engine Module - Implements IPO pricing algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


class TraditionalPricingEngine:
    """
    Implements a traditional bookbuilding approach for IPO pricing.
    
    In the traditional approach, the underwriter (this algorithm) collects
    bids from investors and determines a final offer price based on demand.
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
            total_shares: Total number of shares available in the IPO
        """
        self.price_range = price_range
        self.total_shares = total_shares
        self.min_price, self.max_price = price_range
    
    def calculate_demand_curve(self, investor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the demand curve based on investor bids.
        
        Args:
            investor_data: DataFrame with investor bids and quantities
            
        Returns:
            DataFrame with cumulative demand at different price points
        """
        # Sort by bid price in descending order
        sorted_data = investor_data.sort_values('bid_price', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative demand
        sorted_data['cumulative_shares'] = sorted_data['shares_requested'].cumsum()
        
        return sorted_data
    
    def determine_clearing_price(self, demand_curve: pd.DataFrame) -> float:
        """
        Determine the clearing price where demand meets supply.
        
        Args:
            demand_curve: DataFrame with cumulative demand
            
        Returns:
            Clearing price
        """
        # Find the highest price where cumulative demand exceeds total shares
        clearing_idx = demand_curve[demand_curve['cumulative_shares'] >= self.total_shares].index.min()
        
        if pd.isna(clearing_idx):
            # If demand never exceeds supply, use the minimum price
            return self.min_price
        
        clearing_price = demand_curve.loc[clearing_idx, 'bid_price']
        
        # In traditional IPO pricing, the final price is often set slightly below clearing
        # to ensure full subscription and positive first-day performance
        discount_factor = 0.95  # 5% discount
        final_price = max(clearing_price * discount_factor, self.min_price)
        
        # Round to nearest $0.25 (common in IPO pricing)
        final_price = round(final_price * 4) / 4
        
        # Ensure price is within the initial range
        final_price = max(min(final_price, self.max_price), self.min_price)
        
        return final_price
    
    def allocate_shares(
        self, 
        investor_data: pd.DataFrame, 
        final_price: float
    ) -> pd.DataFrame:
        """
        Allocate shares to investors based on the final price.
        
        Args:
            investor_data: DataFrame with investor bids and quantities
            final_price: Final IPO price
            
        Returns:
            DataFrame with share allocations
        """
        # Create a copy to avoid modifying the original
        allocation_data = investor_data.copy()
        
        # Determine eligible investors (those willing to pay at least the final price)
        allocation_data['eligible'] = allocation_data['bid_price'] >= final_price
        
        # Calculate total eligible demand
        total_eligible_demand = allocation_data.loc[allocation_data['eligible'], 'shares_requested'].sum()
        
        # Calculate allocation ratio (may be less than 1 if oversubscribed)
        allocation_ratio = min(1.0, self.total_shares / total_eligible_demand)
        
        # Calculate allocated shares
        allocation_data['shares_allocated'] = 0
        allocation_data.loc[allocation_data['eligible'], 'shares_allocated'] = (
            allocation_data.loc[allocation_data['eligible'], 'shares_requested'] * allocation_ratio
        ).astype(int)
        
        # Round allocations to nearest 100 shares
        allocation_data['shares_allocated'] = (
            np.round(allocation_data['shares_allocated'], -2)
        ).astype(int)
        
        # Handle any remaining shares due to rounding
        remaining_shares = self.total_shares - allocation_data['shares_allocated'].sum()
        
        if remaining_shares > 0:
            # Allocate remaining shares to largest eligible investors
            eligible_investors = allocation_data[allocation_data['eligible']].sort_values(
                'shares_requested', ascending=False
            )
            
            # Distribute 100 shares at a time to largest investors until all allocated
            share_increment = 100
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
        
        Args:
            allocation_data: DataFrame with share allocations
            final_price: Final IPO price
            
        Returns:
            Dictionary with key metrics
        """
        # Calculate total demand
        total_demand = allocation_data['shares_requested'].sum()
        
        # Calculate eligible demand (at or above final price)
        eligible_demand = allocation_data.loc[
            allocation_data['bid_price'] >= final_price, 'shares_requested'
        ].sum()
        
        # Calculate total shares allocated
        total_allocated = allocation_data['shares_allocated'].sum()
        
        # Calculate capital raised
        capital_raised = total_allocated * final_price
        
        # Calculate oversubscription ratio
        oversubscription_ratio = total_demand / self.total_shares
        
        # Calculate eligible oversubscription ratio
        eligible_oversubscription_ratio = eligible_demand / self.total_shares
        
        # Calculate average bid price
        average_bid_price = allocation_data['bid_price'].mean()
        
        # Estimate "money left on table" (potential first-day pop)
        # This is a rough estimate based on the difference between average bid and final price
        money_left = (average_bid_price - final_price) * total_allocated
        
        # Calculate percentage of investors who received allocations
        pct_investors_allocated = (allocation_data['shares_allocated'] > 0).mean() * 100
        
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
        
        Args:
            investor_data: DataFrame with investor bids and quantities
            
        Returns:
            Dictionary with pricing results and metrics
        """
        # Calculate demand curve
        demand_curve = self.calculate_demand_curve(investor_data)
        
        # Determine final price
        final_price = self.determine_clearing_price(demand_curve)
        
        # Allocate shares
        allocation_data = self.allocate_shares(investor_data, final_price)
        
        # Calculate metrics
        metrics = self.calculate_metrics(allocation_data, final_price)
        
        # Return results
        return {
            'demand_curve': demand_curve,
            'allocation_data': allocation_data,
            'metrics': metrics
        }


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import data_generator
    sys.path.append(str(Path(__file__).parent.parent))
    from ipo_pricing_tool.data_generator import InvestorDataGenerator
    
    # Generate mock data
    generator = InvestorDataGenerator(
        num_investors=100,
        price_range=(20.0, 25.0),
        total_shares=10_000_000,
        seed=42
    )
    
    data = generator.generate_data()
    
    # Run pricing process
    pricing_engine = TraditionalPricingEngine(
        price_range=(20.0, 25.0),
        total_shares=10_000_000
    )
    
    results = pricing_engine.run_pricing_process(data)
    
    # Print results
    print(f"Final IPO Price: ${results['metrics']['final_price']:.2f}")
    print(f"Capital Raised: ${results['metrics']['capital_raised']:,.2f}")
    print(f"Oversubscription Ratio: {results['metrics']['oversubscription_ratio']:.2f}x")
