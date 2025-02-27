#!/usr/bin/env python3
"""
IPO Pricing Tool - Main entry point

A Python tool to price shares in an IPO based on bid and ask data from investors.
This is the main script that coordinates the entire IPO pricing process.
"""

import os  # For file and directory operations
import argparse  # For parsing command line arguments
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations
from typing import Dict, Tuple, Optional  # For type hints to improve code readability

# Try to import modules as a package, but fall back to relative imports if that fails
# This allows the code to work both when installed as a package and when run directly
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
    """
    Parse command line arguments.
    
    This function sets up all the command-line options that can be used when running the tool.
    It defines default values and help text for each option.
    
    Returns:
        An object containing all the parsed command line arguments
    """
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
        default=10_000_000,  # The underscore is just for readability (10 million)
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
        action='store_true',  # This is a flag (True if specified, False otherwise)
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
    """
    Print IPO pricing results to the console.
    
    This function formats and displays all the key metrics from the IPO pricing process
    in a readable format, including pricing information, demand metrics, allocation
    statistics, and financial analysis.
    
    Args:
        results: Dictionary containing all the IPO pricing results and metrics
    """
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print(f"IPO PRICING RESULTS")
    print("="*60)
    
    # Basic pricing information
    print(f"\nFinal IPO Price: ${metrics['final_price']:.2f}")
    print(f"Initial Price Range: ${metrics['total_shares_available']/1_000_000:.1f}M shares @ ${metrics['final_price']:.2f}")
    print(f"Total Capital Raised: ${metrics['capital_raised']:,.2f}")  # The :, adds commas as thousand separators
    
    # Demand metrics section
    print("\nDemand Metrics:")
    print(f"  Total Demand: {metrics['total_demand']:,} shares")
    print(f"  Eligible Demand: {metrics['eligible_demand']:,} shares")  # Eligible means investors willing to pay at least the final price
    print(f"  Oversubscription Ratio: {metrics['oversubscription_ratio']:.2f}x")  # How many times the offering was oversubscribed
    print(f"  Eligible Oversubscription Ratio: {metrics['eligible_oversubscription_ratio']:.2f}x")
    
    # Allocation metrics section
    print("\nAllocation Metrics:")
    print(f"  Total Shares Allocated: {metrics['total_shares_allocated']:,}")
    print(f"  Percentage of Investors Receiving Allocations: {metrics['pct_investors_allocated']:.1f}%")
    
    # Pricing analysis section
    print("\nPricing Analysis:")
    print(f"  Average Bid Price: ${metrics['average_bid_price']:.2f}")
    # "Money left on table" is the potential additional capital that could have been raised
    # if the IPO had been priced at the average bid price
    print(f"  Estimated 'Money Left on Table': ${metrics['estimated_money_left']:,.2f}")
    # "First-day pop" is the expected price increase on the first day of trading
    print(f"  Potential First-Day Pop: {(metrics['average_bid_price'] / metrics['final_price'] - 1) * 100:.1f}%")
    
    print("\n" + "="*60)


def save_data(data: pd.DataFrame, allocation_data: pd.DataFrame, output_dir: str):
    """
    Save investor data and allocation results to CSV files.
    
    This function exports both the original investor bid data and the final allocation
    results to CSV files in the specified output directory.
    
    Args:
        data: DataFrame containing the original investor bid data
        allocation_data: DataFrame containing the final share allocations
        output_dir: Directory where the CSV files will be saved
    """
    # Create the output directory if it doesn't exist
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
    """
    Main entry point for the IPO pricing tool.
    
    This function coordinates the entire IPO pricing process:
    1. Parses command line arguments
    2. Generates mock investor data
    3. Runs the IPO pricing algorithm
    4. Displays and saves results
    5. Creates and shows visualizations
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if needed for saving data or plots
    if args.output_dir or args.save_data or args.save_plots:
        output_dir = args.output_dir or 'ipo_output'  # Default to 'ipo_output' if not specified
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    else:
        output_dir = None
    
    # Step 1: Generate mock investor data
    print(f"Generating mock data for {args.num_investors} investors...")
    generator = InvestorDataGenerator(
        num_investors=args.num_investors,
        price_range=(args.min_price, args.max_price),
        total_shares=args.total_shares,
        seed=args.seed  # Using a seed allows for reproducible results
    )
    
    data = generator.generate_data()
    print(f"Generated data for {len(data)} investors")
    
    # Step 2: Run the IPO pricing process
    print("\nRunning IPO pricing process...")
    pricing_engine = TraditionalPricingEngine(
        price_range=(args.min_price, args.max_price),
        total_shares=args.total_shares
    )
    
    results = pricing_engine.run_pricing_process(data)
    
    # Step 3: Display the results
    print_results(results)
    
    # Step 4: Save data if requested
    if args.save_data and output_dir:
        save_data(data, results['allocation_data'], output_dir)
    
    # Step 5: Create and display visualizations
    if not args.no_display or (args.save_plots and output_dir):
        print("\nCreating visualizations...")
        visualizer = IPOVisualizer()
        
        if args.save_plots and output_dir:
            # Create visualizations and save them to the output directory
            figures = visualizer.create_all_visualizations(results, output_dir)
            print(f"Plots saved to: {output_dir}")
        else:
            # Create visualizations without saving
            figures = visualizer.create_all_visualizations(results)
        
        # Display plots if not disabled
        if not args.no_display:
            plt.show()  # This will open a window with all the plots


# This conditional ensures that the main() function is only executed when this script
# is run directly (not when imported as a module)
if __name__ == "__main__":
    main()
