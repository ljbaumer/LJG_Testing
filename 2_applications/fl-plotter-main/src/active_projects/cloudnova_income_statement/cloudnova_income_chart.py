"""
CloudNova Income Statement Chart Generator

Generates professional visualization of CloudNova's income statement data (2020-2024)
Uses FL-Plotter library with Goldman Sachs styling
"""

import matplotlib
# Use Agg backend only when running from command line, not in notebooks
if __name__ == "__main__":
    matplotlib.use('Agg')  # Non-display backend for faster execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Import PlotBuddy
# Add parent directories to path to find plot_buddy
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '../../utils')
sys.path.insert(0, utils_dir)
from plot_buddy import PlotBuddy

# =============================================
# CONSTANTS
# =============================================

# File paths
DATA_FILE = '../../../CloudNova_Income_Statement_2020_2024.csv'
OUTPUT_DIR = './outputs'
LOGO_FILE = 'gs_logo.png'

# Chart configuration
CHART_TITLE = 'CloudNova Income Statement Analysis'
CHART_SUBTITLE = 'Revenue growth and profitability trends (2020-2024)'
SOURCE_CITATION = 'CloudNova Financial Reports'

def load_cloudnova_data():
    """Load and prepare CloudNova income statement data"""
    # Read the CSV file
    df = pd.read_csv(os.path.join(current_dir, DATA_FILE))
    
    # Set the Line Item as index for easier data manipulation
    df = df.set_index('Line Item')
    
    # Convert all values to millions for better readability
    df = df / 1_000_000
    
    return df

def create_revenue_and_profitability_chart(data, save_path=None):
    """Create a comprehensive chart showing revenue growth and key profitability metrics"""
    
    # Initialize PlotBuddy with style directory
    buddy = PlotBuddy(style_dir_path=os.path.join(current_dir, '../../utils/styles'))
    
    # Use the gs style context
    with buddy.get_style_context('gs'):
        fig, ax = buddy.setup_figure(figsize=(16, 10))
        
        years = data.columns.astype(int)
        
        # Get key metrics
        revenue = data.loc['Revenue']
        gross_profit = data.loc['Gross Profit']
        operating_income = data.loc['Operating Income']
        net_income = data.loc['Net Income']
        
        # Create bar chart for profitability metrics
        x_pos = np.arange(len(years))
        width = 0.2
        
        # Use professional colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars1 = ax.bar(x_pos - 1.5*width, revenue, width, label='Revenue', 
                      color=colors[0], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x_pos - 0.5*width, gross_profit, width, label='Gross Profit', 
                      color=colors[1], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars3 = ax.bar(x_pos + 0.5*width, operating_income, width, label='Operating Income', 
                      color=colors[2], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars4 = ax.bar(x_pos + 1.5*width, net_income, width, label='Net Income', 
                      color=colors[3], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars (only for the top of each bar, no overlapping)
        def add_bar_labels(bars, values, offset=0.3):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if height >= 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                           f'${value:.1f}M', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height - offset,
                           f'${value:.1f}M', ha='center', va='top', 
                           fontsize=9, fontweight='bold', color='red')
        
        add_bar_labels(bars1, revenue, 0.4)
        add_bar_labels(bars2, gross_profit, 0.3)
        add_bar_labels(bars3, operating_income, 0.2)
        add_bar_labels(bars4, net_income, 0.1)
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Amount ($ Millions)', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(years)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Set y-axis limits to prevent cramping
        max_val = max(revenue.max(), gross_profit.max(), operating_income.max(), net_income.max())
        min_val = min(0, net_income.min())
        ax.set_ylim(min_val - 1, max_val + 2)
        
        # Format y-axis as currency
        buddy.format_axis_as_currency(ax, axis='y', symbol='$', suffix='M')
        
        # Add titles using PlotBuddy
        buddy.add_titles(ax, CHART_TITLE, CHART_SUBTITLE)
        
        # Create legend using PlotBuddy
        buddy.create_legend(ax, position='bottom', ncol=4)
        
        # Add logo and source using PlotBuddy
        logo_path = os.path.join(current_dir, '../../../logos', LOGO_FILE)
        if os.path.exists(logo_path):
            buddy.add_logo(fig, logo_path)
        buddy.add_source_citation(fig, SOURCE_CITATION)
        
        # Apply layout using PlotBuddy
        buddy.apply_tight_layout(fig)
        
        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax

def create_growth_metrics_chart(data, save_path=None):
    """Create a chart showing year-over-year growth rates"""
    
    # Initialize PlotBuddy with style directory
    buddy = PlotBuddy(style_dir_path=os.path.join(current_dir, '../../utils/styles'))
    
    # Use the gs style context
    with buddy.get_style_context('gs'):
        fig, ax = buddy.setup_figure(figsize=(14, 8))
        
        years = data.columns.astype(int)
        
        # Calculate year-over-year growth rates
        revenue = data.loc['Revenue']
        gross_profit = data.loc['Gross Profit']
        net_income = data.loc['Net Income']
        
        # Calculate growth rates (skip first year)
        revenue_growth = revenue.pct_change() * 100
        gross_profit_growth = gross_profit.pct_change() * 100
        net_income_growth = net_income.pct_change() * 100
        
        # Plot growth rates (starting from 2021)
        years_growth = years[1:]  # Skip 2020 since we can't calculate growth
        
        # Use professional colors and styling
        ax.plot(years_growth, revenue_growth.iloc[1:], marker='o', linewidth=4, markersize=10, 
                label='Revenue Growth', color='#1f77b4', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor='#1f77b4')
        ax.plot(years_growth, gross_profit_growth.iloc[1:], marker='s', linewidth=4, markersize=10, 
                label='Gross Profit Growth', color='#ff7f0e', markerfacecolor='white',
                markeredgewidth=2, markeredgecolor='#ff7f0e')
        ax.plot(years_growth, net_income_growth.iloc[1:], marker='^', linewidth=4, markersize=10, 
                label='Net Income Growth', color='#2ca02c', markerfacecolor='white',
                markeredgewidth=2, markeredgecolor='#2ca02c')
        
        # Add value labels with smart positioning to avoid overlap
        growth_values = [revenue_growth.iloc[1:], gross_profit_growth.iloc[1:], net_income_growth.iloc[1:]]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ['Revenue', 'Gross Profit', 'Net Income']
        
        for i, year in enumerate(years_growth):
            # Get all values for this year
            year_values = [growth_values[j].iloc[i] for j in range(3)]
            
            # Sort by value to position labels without overlap
            sorted_indices = sorted(range(3), key=lambda x: year_values[x], reverse=True)
            
            for j, idx in enumerate(sorted_indices):
                value = year_values[idx]
                if abs(value) < 1000:  # Only show reasonable growth rates
                    # Position labels with vertical offset to prevent overlap
                    offset = 15 + (j * 25)  # Stagger labels vertically
                    ax.text(year, value + offset, f'{value:.1f}%', 
                           ha='center', va='bottom', fontweight='bold', 
                           fontsize=10, color=colors[idx],
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=colors[idx], alpha=0.8))
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Growth Rate (%)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Set y-axis limits to accommodate labels
        max_growth = max([max(growth_values[i]) for i in range(3) if max(growth_values[i]) < 1000])
        min_growth = min([min(growth_values[i]) for i in range(3) if min(growth_values[i]) > -1000])
        ax.set_ylim(min_growth - 50, max_growth + 100)
        
        # Add titles using PlotBuddy
        buddy.add_titles(ax, 'CloudNova Year-over-Year Growth Analysis', 
                        'Revenue and profitability growth trends (2021-2024)')
        
        # Create legend using PlotBuddy
        buddy.create_legend(ax, position='bottom', ncol=3)
        
        # Add logo and source using PlotBuddy
        logo_path = os.path.join(current_dir, '../../../logos', LOGO_FILE)
        if os.path.exists(logo_path):
            buddy.add_logo(fig, logo_path)
        buddy.add_source_citation(fig, SOURCE_CITATION)
        
        # Apply layout using PlotBuddy
        buddy.apply_tight_layout(fig)
        
        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax

def generate_cloudnova_charts(save_directory=OUTPUT_DIR, display=True):
    """
    Main function to generate CloudNova income statement charts
    
    Args:
        save_directory (str): Directory to save output files
        display (bool): Whether to display the charts after generation
    
    Returns:
        tuple: (main_chart, growth_chart)
    """
    # Ensure output directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Load data
    data = load_cloudnova_data()
    
    # Generate main income statement chart
    main_save_path = os.path.join(save_directory, 'cloudnova_income_statement.png')
    main_chart = create_revenue_and_profitability_chart(data, main_save_path)
    
    # Generate growth metrics chart
    growth_save_path = os.path.join(save_directory, 'cloudnova_growth_metrics.png')
    growth_chart = create_growth_metrics_chart(data, growth_save_path)
    
    if display:
        plt.show()
    
    return main_chart, growth_chart

# Example usage
if __name__ == "__main__":
    import sys
    
    # Check for --no-display flag
    display = '--no-display' not in sys.argv
    
    # Generate charts
    main_result, growth_result = generate_cloudnova_charts(display=display)
    
    print("CloudNova Income Statement charts generated successfully!")
    print("Files saved in ./outputs directory:")
    print("  - cloudnova_income_statement.png (Main Analysis)")
    print("  - cloudnova_growth_metrics.png (Growth Analysis)")
