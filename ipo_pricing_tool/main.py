#!/usr/bin/env python3
"""
IPO Pricing Tool - Main entry point

A Python tool to price shares in an IPO based on bid and ask data from investors.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Try to import as a package, but fall back to relative imports if that fails
try:
    from ipo_pricing_tool.data_generator import InvestorDataGenerator
    from ipo_pricing_tool.pricing_engine import TraditionalPricingEngine
    from ipo_pricing_tool.visualizer import IPOVisualizer
except ModuleNotFoundError:
    # When running directly from the package directory
    from data_generator import InvestorDataGenerator
    from pricing_engine import TraditionalPricingEngine
    from visualizer import IPOVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='IPO Pricing Tool - Price shares in an IPO based on investor bids'
    )
    
    # Basic IPO parameters
    parser.add_argument(
        '--num-investors',
        type=int,
        default=100,
        help='Number of investors to generate (default: 100)'
    )
    
    parser.add_argument(
        '--min-price',
        type=float,
        default=20.0,
        help='Minimum price in the IPO range (default: $20.00)'
    )
    
    parser.add_argument(
        '--max-price',
        type=float,
        default=25.0,
        help='Maximum price in the IPO range (default: $25.00)'
    )
    
    parser.add_argument(
        '--total-shares',
        type=int,
        default=10_000_000,
        help='Total number of shares available in the IPO (default: 10,000,000)'
    )
    
    # Random seed for reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible results (default: None)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output files (default: None)'
    )
    
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save investor data to CSV file'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to output directory'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots (useful for batch processing)'
    )
    
    return parser.parse_args()


def print_results(results: Dict):
    """Print IPO pricing results to the console."""
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print(f"IPO PRICING RESULTS")
    print("="*60)
    
    print(f"\nFinal IPO Price: ${metrics['final_price']:.2f}")
    print(f"Initial Price Range: ${metrics['total_shares_available']/1_000_000:.1f}M shares @ ${metrics['final_price']:.2f}")
    print(f"Total Capital Raised: ${metrics['capital_raised']:,.2f}")
    
    print("\nDemand Metrics:")
    print(f"  Total Demand: {metrics['total_demand']:,} shares")
    print(f"  Eligible Demand: {metrics['eligible_demand']:,} shares")
    print(f"  Oversubscription Ratio: {metrics['oversubscription_ratio']:.2f}x")
    print(f"  Eligible Oversubscription Ratio: {metrics['eligible_oversubscription_ratio']:.2f}x")
    
    print("\nAllocation Metrics:")
    print(f"  Total Shares Allocated: {metrics['total_shares_allocated']:,}")
    print(f"  Percentage of Investors Receiving Allocations: {metrics['pct_investors_allocated']:.1f}%")
    
    print("\nPricing Analysis:")
    print(f"  Average Bid Price: ${metrics['average_bid_price']:.2f}")
    print(f"  Estimated 'Money Left on Table': ${metrics['estimated_money_left']:,.2f}")
    print(f"  Potential First-Day Pop: {(metrics['average_bid_price'] / metrics['final_price'] - 1) * 100:.1f}%")
    
    print("\n" + "="*60)


def save_data(data: pd.DataFrame, allocation_data: pd.DataFrame, output_dir: str):
    """Save investor data and allocation results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original investor data
    data_path = os.path.join(output_dir, 'investor_data.csv')
    data.to_csv(data_path, index=False)
    print(f"Investor data saved to: {data_path}")
    
    # Save allocation results
    allocation_path = os.path.join(output_dir, 'allocation_results.csv')
    allocation_data.to_csv(allocation_path, index=False)
    print(f"Allocation results saved to: {allocation_path}")


def main():
    """Main entry point for the IPO pricing tool."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if needed
    if args.output_dir or args.save_data or args.save_plots:
        output_dir = args.output_dir or 'ipo_output'
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    # Generate investor data
    print(f"Generating mock data for {args.num_investors} investors...")
    generator = InvestorDataGenerator(
        num_investors=args.num_investors,
        price_range=(args.min_price, args.max_price),
        total_shares=args.total_shares,
        seed=args.seed
    )
    
    data = generator.generate_data()
    print(f"Generated data for {len(data)} investors")
    
    # Run pricing process
    print("\nRunning IPO pricing process...")
    pricing_engine = TraditionalPricingEngine(
        price_range=(args.min_price, args.max_price),
        total_shares=args.total_shares
    )
    
    results = pricing_engine.run_pricing_process(data)
    
    # Print results
    print_results(results)
    
    # Save data if requested
    if args.save_data and output_dir:
        save_data(data, results['allocation_data'], output_dir)
    
    # Create visualizations
    if not args.no_display or (args.save_plots and output_dir):
        print("\nCreating visualizations...")
        visualizer = IPOVisualizer()
        
        if args.save_plots and output_dir:
            figures = visualizer.create_all_visualizations(results, output_dir)
            print(f"Plots saved to: {output_dir}")
        else:
            figures = visualizer.create_all_visualizations(results)
        
        # Display plots if not disabled
        if not args.no_display:
            plt.show()


if __name__ == "__main__":
    main()
