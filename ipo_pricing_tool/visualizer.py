"""
Visualizer Module - Creates visualizations for IPO pricing data.

This module generates various visualizations to help understand and analyze
the results of an IPO pricing process, including demand curves, allocation
distributions, bid distributions, and summary metrics.
"""

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations
import matplotlib.ticker as mtick  # For custom tick formatting
from typing import Dict, Tuple, List, Optional  # For type hints


class IPOVisualizer:
    """
    Creates visualizations for IPO pricing data.
    
    This class provides methods to create various visualizations that help understand
    and analyze the results of an IPO pricing process. It includes:
    1. Demand curve visualization
    2. Allocation distribution visualization
    3. Bid distribution visualization
    4. Summary metrics visualization
    
    Each visualization is designed to highlight different aspects of the IPO process
    and results, making it easier to understand the pricing dynamics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer with default settings.
        
        Args:
            figsize: Default figure size for plots (width, height) in inches
        """
        self.figsize = figsize
        
        # Set style for consistent, professional-looking plots
        plt.style.use('seaborn-v0_8-whitegrid')  # Clean, grid-based style
        
        # Set custom colors for consistent visualization
        self.colors = {
            'primary': '#1f77b4',    # Blue - used for main data series
            'secondary': '#ff7f0e',  # Orange - used for secondary data
            'tertiary': '#2ca02c',   # Green - used for tertiary data or positive indicators
            'highlight': '#d62728',  # Red - used for important thresholds or highlights
            'neutral': '#7f7f7f',    # Gray - used for background elements or ranges
        }
    
    def plot_demand_curve(
        self, 
        demand_curve: pd.DataFrame, 
        final_price: float,
        total_shares: int,
        price_range: Tuple[float, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the demand curve with the final price.
        
        This visualization shows the relationship between price and cumulative demand.
        It helps understand how demand changes at different price points and where
        the final price was set relative to the demand curve.
        
        Key elements:
        - Demand curve (blue line): Shows cumulative shares demanded at each price point
        - Final price (red line): The final IPO price
        - Total shares (green line): The total number of shares available
        - Price range (gray area): The initial IPO price range
        
        Args:
            demand_curve: DataFrame with cumulative demand
            final_price: Final IPO price
            total_shares: Total shares available
            price_range: Initial price range (min, max)
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot demand curve - this is the main visualization showing how demand
        # changes at different price points
        ax.plot(
            demand_curve['cumulative_shares'],  # X-axis: cumulative shares
            demand_curve['bid_price'],          # Y-axis: bid price
            marker='o',                         # Add markers at each data point
            linestyle='-',                      # Connect points with a line
            color=self.colors['primary'],       # Blue color
            alpha=0.7,                          # Slight transparency
            label='Demand Curve'                # Label for legend
        )
        
        # Add horizontal line for final price - this shows where the IPO was priced
        ax.axhline(
            y=final_price,                      # Y-coordinate for the line
            color=self.colors['highlight'],     # Red color
            linestyle='--',                     # Dashed line
            linewidth=2,                        # Thicker line for emphasis
            label=f'Final Price: ${final_price:.2f}'  # Label with the price
        )
        
        # Add vertical line for total shares - this shows the supply constraint
        ax.axvline(
            x=total_shares,                     # X-coordinate for the line
            color=self.colors['tertiary'],      # Green color
            linestyle='--',                     # Dashed line
            linewidth=2,                        # Thicker line for emphasis
            label=f'Total Shares: {total_shares:,}'  # Label with formatted number
        )
        
        # Add price range - this shows the initial range published in the prospectus
        min_price, max_price = price_range
        ax.axhspan(
            min_price,                          # Lower bound of the range
            max_price,                          # Upper bound of the range
            alpha=0.1,                          # Very transparent
            color=self.colors['neutral'],       # Gray color
            label=f'Initial Price Range: ${min_price:.2f}-${max_price:.2f}'  # Label with range
        )
        
        # Set labels and title
        ax.set_xlabel('Cumulative Shares', fontsize=12)
        ax.set_ylabel('Bid Price ($)', fontsize=12)
        ax.set_title('IPO Demand Curve', fontsize=16)
        
        # Format x-axis with commas for thousands (e.g., 1,000,000)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Format y-axis with dollar signs (e.g., $20.00)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:.2f}'))
        
        # Add grid for easier reading of values
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend to explain the elements
        ax.legend(loc='best', fontsize=10)
        
        # Adjust layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_allocation_distribution(
        self, 
        allocation_data: pd.DataFrame,
        final_price: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the distribution of share allocations.
        
        This visualization shows how many investors received allocations of different sizes.
        It helps understand the distribution of shares among investors and identify any
        concentration of allocations.
        
        Key elements:
        - Bar chart showing the number of investors in each allocation size category
        - Categories range from zero shares to 50,000+ shares
        
        Args:
            allocation_data: DataFrame with share allocations
            final_price: Final IPO price (used in the title)
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create bins for allocation sizes - these define the categories for the bar chart
        bins = [0, 1000, 5000, 10000, 50000, 100000, float('inf')]
        bin_labels = ['0', '1-1,000', '1,001-5,000', '5,001-10,000', '10,001-50,000', '50,001+']
        
        # Categorize allocations - assign each investor to a category based on their allocation
        allocation_data['allocation_category'] = pd.cut(
            allocation_data['shares_allocated'],  # Values to categorize
            bins=bins,                            # Bin edges
            labels=bin_labels,                    # Labels for each bin
            right=False                           # Intervals closed on the left
        )
        
        # Count investors in each category
        # value_counts() counts occurrences of each category
        # reindex() ensures all categories are shown in the correct order
        allocation_counts = allocation_data['allocation_category'].value_counts().reindex(bin_labels)
        
        # Plot bar chart - this shows the distribution of allocations
        bars = ax.bar(
            allocation_counts.index,       # X-axis: allocation categories
            allocation_counts.values,      # Y-axis: number of investors
            color=self.colors['secondary'], # Orange color
            alpha=0.7                      # Slight transparency
        )
        
        # Add data labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,  # X position (center of bar)
                height + 0.5,                      # Y position (just above bar)
                f'{int(height)}',                  # Text (number of investors)
                ha='center',                       # Horizontal alignment
                va='bottom',                       # Vertical alignment
                fontsize=10                        # Font size
            )
        
        # Set labels and title
        ax.set_xlabel('Allocation Size (Shares)', fontsize=12)
        ax.set_ylabel('Number of Investors', fontsize=12)
        ax.set_title(f'Distribution of Share Allocations at ${final_price:.2f}', fontsize=16)
        
        # Add grid for easier reading of values
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bid_distribution(
        self, 
        investor_data: pd.DataFrame,
        final_price: float,
        price_range: Tuple[float, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the distribution of investor bids.
        
        This visualization shows the distribution of bid prices across all investors.
        It helps understand investor sentiment and where the final price was set
        relative to the overall bid distribution.
        
        Key elements:
        - Histogram of bid prices
        - Final price line (red)
        - Initial price range (gray area)
        
        Args:
            investor_data: DataFrame with investor bids
            final_price: Final IPO price
            price_range: Initial price range (min, max)
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create histogram of bid prices
        min_price, max_price = price_range
        price_range_width = max_price - min_price
        
        # Determine bin edges for the histogram
        # We want enough bins to show the distribution clearly
        bin_width = price_range_width / 10  # 10 bins across the price range
        min_bid = investor_data['bid_price'].min()
        max_bid = investor_data['bid_price'].max()
        
        # Ensure bins cover all bids with some padding
        bin_min = max(0, min_bid - bin_width)  # Don't go below 0
        bin_max = max_bid + bin_width  # Add some padding above max bid
        
        # Create array of bin edges
        bins = np.arange(bin_min, bin_max + bin_width, bin_width)
        
        # Plot histogram - this shows the distribution of bid prices
        n, bins, patches = ax.hist(
            investor_data['bid_price'],  # Data to plot
            bins=bins,                   # Bin edges
            color=self.colors['primary'], # Blue color
            alpha=0.7,                   # Slight transparency
            edgecolor='black',           # Black outline
            linewidth=0.5                # Thin outline
        )
        
        # Add vertical line for final price - this shows where the IPO was priced
        ax.axvline(
            x=final_price,                      # X-coordinate for the line
            color=self.colors['highlight'],     # Red color
            linestyle='--',                     # Dashed line
            linewidth=2,                        # Thicker line for emphasis
            label=f'Final Price: ${final_price:.2f}'  # Label with the price
        )
        
        # Add price range - this shows the initial range published in the prospectus
        ax.axvspan(
            min_price,                          # Lower bound of the range
            max_price,                          # Upper bound of the range
            alpha=0.1,                          # Very transparent
            color=self.colors['neutral'],       # Gray color
            label=f'Initial Price Range: ${min_price:.2f}-${max_price:.2f}'  # Label with range
        )
        
        # Set labels and title
        ax.set_xlabel('Bid Price ($)', fontsize=12)
        ax.set_ylabel('Number of Investors', fontsize=12)
        ax.set_title('Distribution of Investor Bids', fontsize=16)
        
        # Format x-axis with dollar signs (e.g., $20.00)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:.2f}'))
        
        # Add grid for easier reading of values
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add legend to explain the elements
        ax.legend(loc='best', fontsize=10)
        
        # Adjust layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_summary(
        self, 
        metrics: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a summary visualization of key IPO metrics.
        
        This visualization provides a dashboard-style overview of the most important
        IPO metrics in four panels:
        1. Pricing comparison (final price vs. average bid)
        2. Supply and demand comparison
        3. Financial impact (capital raised and money left on table)
        4. Allocation metrics (oversubscription ratios and percentage of investors allocated)
        
        Args:
            metrics: Dictionary with IPO metrics
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        axs = axs.flatten()
        
        # Plot 1: Final Price vs. Average Bid
        ax1 = axs[0]
        prices = ['Final Price', 'Avg Bid Price']
        values = [metrics['final_price'], metrics['average_bid_price']]
        
        bars1 = ax1.bar(
            prices, 
            values,
            color=[self.colors['highlight'], self.colors['primary']],
            alpha=0.7
        )
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'${height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        ax1.set_title('Pricing', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=10)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Demand vs. Supply
        ax2 = axs[1]
        categories = ['Total Shares', 'Total Demand', 'Eligible Demand']
        values = [
            metrics['total_shares_available'],
            metrics['total_demand'],
            metrics['eligible_demand']
        ]
        
        bars2 = ax2.bar(
            categories,
            values,
            color=[self.colors['tertiary'], self.colors['primary'], self.colors['secondary']],
            alpha=0.7
        )
        
        # Add data labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + height*0.01,
                f'{int(height):,}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        ax2.set_title('Supply & Demand', fontsize=12)
        ax2.set_ylabel('Shares', fontsize=10)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax2.ticklabel_format(style='plain', axis='y')
        
        # Plot 3: Capital Raised & Money Left
        ax3 = axs[2]
        categories = ['Capital Raised', 'Est. Money Left']
        values = [
            metrics['capital_raised'],
            metrics['estimated_money_left']
        ]
        
        bars3 = ax3.bar(
            categories,
            values,
            color=[self.colors['tertiary'], self.colors['highlight']],
            alpha=0.7
        )
        
        # Add data labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width()/2.,
                height + height*0.01,
                f'${int(height):,}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        ax3.set_title('Financial Impact', fontsize=12)
        ax3.set_ylabel('Amount ($)', fontsize=10)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax3.ticklabel_format(style='plain', axis='y')
        
        # Plot 4: Oversubscription & Allocation
        ax4 = axs[3]
        
        # Create a secondary y-axis for percentage
        ax4_twin = ax4.twinx()
        
        # Bar data
        categories = ['Oversubscription', 'Eligible Oversub.']
        values = [
            metrics['oversubscription_ratio'],
            metrics['eligible_oversubscription_ratio']
        ]
        
        # Line data
        pct_allocated = metrics['pct_investors_allocated']
        
        # Plot bars for oversubscription
        bars4 = ax4.bar(
            categories,
            values,
            color=[self.colors['primary'], self.colors['secondary']],
            alpha=0.7
        )
        
        # Add data labels for bars
        for bar in bars4:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f'{height:.2f}x',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Plot line for percentage allocated
        ax4_twin.plot(
            [-0.5, 1.5],
            [pct_allocated, pct_allocated],
            color=self.colors['highlight'],
            linestyle='--',
            linewidth=2,
            label=f'% Investors Allocated: {pct_allocated:.1f}%'
        )
        
        ax4.set_title('Allocation Metrics', fontsize=12)
        ax4.set_ylabel('Ratio (x)', fontsize=10)
        ax4_twin.set_ylabel('% Investors Allocated', fontsize=10)
        ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax4_twin.set_ylim(0, 100)
        
        # Add legend for the line
        ax4_twin.legend(loc='upper right', fontsize=8)
        
        # Set overall title
        plt.suptitle(
            f'IPO Summary: ${metrics["final_price"]:.2f} per Share',
            fontsize=16,
            y=0.98
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_all_visualizations(
        self, 
        results: Dict,
        output_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create all visualizations for the IPO pricing results.
        
        Args:
            results: Dictionary with pricing results
            output_dir: Directory to save figures (optional)
            
        Returns:
            Dictionary with figure objects
        """
        demand_curve = results['demand_curve']
        allocation_data = results['allocation_data']
        metrics = results['metrics']
        
        final_price = metrics['final_price']
        total_shares = metrics['total_shares_available']
        
        # Extract price range from first and last rows of demand curve
        # (This assumes demand_curve is sorted by bid_price in descending order)
        max_price = demand_curve['bid_price'].max()
        min_price = demand_curve['bid_price'].min()
        price_range = (min_price, max_price)
        
        # Create save paths if output_dir provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            demand_curve_path = os.path.join(output_dir, 'demand_curve.png')
            allocation_dist_path = os.path.join(output_dir, 'allocation_distribution.png')
            bid_dist_path = os.path.join(output_dir, 'bid_distribution.png')
            metrics_summary_path = os.path.join(output_dir, 'metrics_summary.png')
        else:
            demand_curve_path = None
            allocation_dist_path = None
            bid_dist_path = None
            metrics_summary_path = None
        
        # Create visualizations
        demand_fig = self.plot_demand_curve(
            demand_curve,
            final_price,
            total_shares,
            price_range,
            save_path=demand_curve_path
        )
        
        allocation_fig = self.plot_allocation_distribution(
            allocation_data,
            final_price,
            save_path=allocation_dist_path
        )
        
        bid_fig = self.plot_bid_distribution(
            allocation_data,
            final_price,
            price_range,
            save_path=bid_dist_path
        )
        
        metrics_fig = self.plot_metrics_summary(
            metrics,
            save_path=metrics_summary_path
        )
        
        return {
            'demand_curve': demand_fig,
            'allocation_distribution': allocation_fig,
            'bid_distribution': bid_fig,
            'metrics_summary': metrics_fig
        }


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import other modules
    sys.path.append(str(Path(__file__).parent.parent))
    from ipo_pricing_tool.data_generator import InvestorDataGenerator
    from ipo_pricing_tool.pricing_engine import TraditionalPricingEngine
    
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
    
    # Create visualizations
    visualizer = IPOVisualizer()
    figures = visualizer.create_all_visualizations(results)
    
    # Show plots
    plt.show()
