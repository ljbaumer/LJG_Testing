"""
AI Power Demand Growth Chart Generator
Section 1.1 - Executive Summary Graphics

Generates visualization showing exponential AI power demand growth
Based on NVIDIA sellside analysis and Goldman Sachs Q1 2025 GIR estimates
"""

import matplotlib
# Use Agg backend only when running from command line, not in notebooks
if __name__ == "__main__":
    matplotlib.use('Agg')  # Non-display backend for faster execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
# STRING CONSTANTS - EDIT THESE AS NEEDED
# =============================================

# File and Directory Names
DATA_DIR = 'data'
OUTPUT_DIR = './outputs'
GPU_DATA_FILE = 'quarterly_power_requirements_nvidia_gpu.csv'
OUTPUT_CHART_FILE = 'ai_power_demand_cumulative_area.png'
LOGO_FILE = 'gs_logo.png'

# Chart Titles and Labels
CHART_TITLE = 'AI power demand is accelerating'
CHART_SUBTITLE = 'Cumulative power demand from NVIDIA datacenter GPU shipments (colors represent different models)'
X_AXIS_LABEL = 'Quarter'
Y_AXIS_LABEL = 'Cumulative Power Demand'
SOURCE_CITATION = 'Goldman Sachs Research Q1 2025 Estimates, NVIDIA Datacenter GPU Product Datasheets'

# Data Column Names
COL_QUARTER = 'Quarter'
COL_TOTAL_MW = 'Total (MW)'
COL_TOTAL_GW = 'Total (GW)'
COL_YEAR = 'year'

# GPU Model Names and Suffixes
GPU_MW_SUFFIX = ' (MW)'
GPU_GW_SUFFIX = ' (GW)'
GPU_MODELS_MW = ['A100 (MW)', 'H100 (MW)', 'H200 (MW)', 'B100 (MW)', 'GB200 (MW)', 'GB300 (MW)', 'VR200 (MW)']

# GPU Model Announcement Dates
GPU_ANNOUNCEMENT_DATES = {
    'A100': '2020',
    'H100': '2023', 
    'H200': '2024',
    'B100': '2024',
    'GB200': '2024',
    'GB300': '2025',
    'VR200': '2026E'
}

# =============================================
# EMBEDDED DATA FROM CSV - EDIT THESE AS NEEDED
# =============================================

# Complete quarterly data from quarterly_power_requirements_nvidia_gpu.csv
QUARTERS = [
    '1Q23', '2Q23', '3Q23', '4Q23', '1Q24', '2Q24', '3Q24', '4Q24',
    '1Q25', '2Q25E', '3Q25E', '4Q25E', '1Q26E', '2Q26E', '3Q26E', '4Q26E'
]

# GPU Power Data in MW
A100_MW = [
    95.04, 206.688, 202.554, 182.299, 164.069, 16.407, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

H100_MW = [
    97.2, 279.029, 453.980, 635.572, 872.005, 828.404, 782.842, 665.416,
    166.354, 83.177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

H200_MW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 289.536, 549.12, 554.611,
    654.441, 621.719, 456.964, 114.241, 28.560, 0.0, 0.0, 0.0
]

B100_MW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.112, 124.8,
    343.2, 463.32, 602.316, 692.663, 540.277, 367.389, 205.738, 77.152
]

GB200_MW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.544, 177.6,
    488.4, 659.34, 857.142, 985.713, 768.856, 522.822, 292.781, 109.793
]

GB300_MW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 79.872, 379.392, 948.48, 1498.598, 1897.226, 1992.087
]

VR200_MW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.4, 432.0
]

TOTAL_MW = [
    192.24, 485.717, 656.534, 817.871, 1036.073, 1134.347, 1351.618, 1522.427,
    1652.395, 1827.556, 1996.294, 2172.010, 2286.174, 2388.809, 2482.144, 2611.031
]

# Sample Data Arrays (kept for backwards compatibility but now unused)
SAMPLE_QUARTERS = ['1Q23', '2Q23', '3Q23', '4Q23', '1Q24', '2Q24', '3Q24', '4Q24']
SAMPLE_A100_GW = [0.095, 0.207, 0.203, 0.182, 0.164, 0.016, 0.0, 0.0]
SAMPLE_H100_GW = [0.097, 0.279, 0.454, 0.636, 0.872, 0.828, 0.783, 0.665]
SAMPLE_H200_GW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.290, 0.549, 0.555]

# Format Strings
YEAR_EXTRACT_PATTERN = r'(\d{2})$'
GW_FORMAT = '{:.1f}GW'
AXIS_FORMATTER_ZERO = '0'
AXIS_FORMATTER_GW = '{} GW'
MODEL_LABEL_FORMAT = '{} ({})'

# Messages and Warnings
WARNING_DATA_NOT_FOUND = 'Warning: Data file not found at {}, using sample data'
SUCCESS_MESSAGE = 'AI Power Demand Growth charts generated successfully!'
FILES_SAVED_MESSAGE = 'Files saved in ./outputs directory:'

# Command Line Arguments
NO_DISPLAY_FLAG = '--no-display'

# Chart Styling Constants
MW_TO_GW_CONVERTER = 1000
YEAR_BASE = 2000
CUMULATIVE_LABEL_OFFSET = 2.0
SKIP_FIRST_LABEL = 1
ALPHA_FULL = 1.0
LINE_WIDTH = 3
MARKER_SIZE = 6
CHART_DPI = 300

# =============================================
# FUNCTIONS
# =============================================

def prepare_ai_demand_data():
    """
    Prepare AI power demand growth data from embedded constants
    Uses data directly embedded in the file instead of reading CSV
    """
    # Use embedded data directly - no CSV reading needed
    df = pd.DataFrame({
        COL_QUARTER: QUARTERS,
        'A100 (MW)': A100_MW,
        'H100 (MW)': H100_MW,
        'H200 (MW)': H200_MW,
        'B100 (MW)': B100_MW,
        'GB200 (MW)': GB200_MW,
        'GB300 (MW)': GB300_MW,
        'VR200 (MW)': VR200_MW,
        COL_TOTAL_MW: TOTAL_MW
    })
    
    # Extract year from quarter (e.g., "1Q23" -> 2023)
    year_extract = df[COL_QUARTER].str.extract(YEAR_EXTRACT_PATTERN)[0]
    df[COL_YEAR] = year_extract.dropna().astype(int) + YEAR_BASE
    
    # Convert MW to GW
    for model in GPU_MODELS_MW:
        if model in df.columns:
            df[f'{model.replace(GPU_MW_SUFFIX, "")} (GW)'] = df[model] / MW_TO_GW_CONVERTER
    
    df[COL_TOTAL_GW] = df[COL_TOTAL_MW] / MW_TO_GW_CONVERTER
    
    return df

def create_cumulative_area_chart(data, save_path=None):
    """
    Create cumulative area chart showing running total power demand by GPU model
    """
    # Initialize PlotBuddy with style directory
    buddy = PlotBuddy(style_dir_path=os.path.join(current_dir, '../../utils/styles'))
    
    # Use the gs style context
    with buddy.get_style_context('gs'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Define GPU models and get colors
        gpu_models = []
        for col in data.columns:
            if GPU_GW_SUFFIX in col and col != COL_TOTAL_GW:
                model = col.replace(GPU_GW_SUFFIX, '')
                gpu_models.append(model)
        
        # Create labels with announcement dates
        gpu_labels = []
        for model in gpu_models:
            if model in GPU_ANNOUNCEMENT_DATES:
                gpu_labels.append(MODEL_LABEL_FORMAT.format(model, GPU_ANNOUNCEMENT_DATES[model]))
            else:
                gpu_labels.append(model)  # Fallback to model name only
        
        # Get colors from current matplotlib color cycle (defined in gs.mplstyle)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        model_colors = dict(zip(gpu_models, colors[:len(gpu_models)]))
        
        # Calculate cumulative data for each model
        cumulative_data = {}
        for model in gpu_models:
            values = data[f'{model}{GPU_GW_SUFFIX}'].fillna(0)
            cumulative_data[model] = values.cumsum()
        
        # Create stacked area chart with labels
        ax.stackplot(data[COL_QUARTER], 
                     *[cumulative_data[model] for model in gpu_models],
                     labels=gpu_labels,
                     colors=[model_colors[model] for model in gpu_models],
                     alpha=ALPHA_FULL)
        
        # Calculate total cumulative for labels
        total_cumulative = data[COL_TOTAL_GW].cumsum()
        
        # Add total cumulative line on top with label
        ax.plot(data[COL_QUARTER], total_cumulative, 
                color='black', linewidth=LINE_WIDTH, 
                marker='o', markersize=MARKER_SIZE,
                alpha=ALPHA_FULL)
        
        ax.set_xlabel(X_AXIS_LABEL, fontsize=buddy.standard_font_size)
        ax.set_ylabel(Y_AXIS_LABEL, fontsize=buddy.standard_font_size)
        
        # Format y-axis to show "GW" in tick labels
        from matplotlib.ticker import FuncFormatter
        def format_gw(x, pos):
            if x == 0:
                return AXIS_FORMATTER_ZERO
            else:
                return AXIS_FORMATTER_GW.format(int(x))
        
        ax.yaxis.set_major_formatter(FuncFormatter(format_gw))
        
        # Add cumulative value labels (skip the first one to avoid overlap)
        for i, cum_total in enumerate(total_cumulative):
            if i > 0:  # Skip the first label
                ax.text(i, cum_total + CUMULATIVE_LABEL_OFFSET, GW_FORMAT.format(cum_total), 
                        ha='center', va='bottom', fontweight='bold', fontsize=buddy.standard_font_size)
        
        plt.xticks(rotation=0, ha='center')
        
        # Add titles using PlotBuddy
        buddy.add_titles(ax, CHART_TITLE, CHART_SUBTITLE)
        
        # Create legend using PlotBuddy
        buddy.create_legend(ax, position='bottom')
        
        # Add logo and source using PlotBuddy
        logo_path = os.path.join('./logos', LOGO_FILE)
        buddy.add_logo(fig, logo_path)
        buddy.add_source_citation(fig, SOURCE_CITATION)
        
        # Apply layout using PlotBuddy
        buddy.apply_tight_layout(fig)
        
        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=CHART_DPI, bbox_inches='tight')
        
        return fig, ax

def generate_ai_power_demand_chart(chart_type='cumulative', save_directory=OUTPUT_DIR, display=True):
    """
    Main method to generate AI power demand growth chart from GPU data
    Creates cumulative area chart only
    
    Args:
        chart_type (str): Only 'cumulative' supported now
        save_directory (str): Directory to save output files
        display (bool): Whether to display the chart after generation
    
    Returns:
        tuple: (figure, axis)
    """
    # Ensure output directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Prepare data
    data = prepare_ai_demand_data()
    
    # Generate only the cumulative area chart
    save_path = os.path.join(save_directory, OUTPUT_CHART_FILE)
    result = create_cumulative_area_chart(data, save_path)
    if display:
        plt.show()
    return result


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check for --no-display flag
    display = NO_DISPLAY_FLAG not in sys.argv
    
    # Generate only the cumulative area chart by default
    result = generate_ai_power_demand_chart(chart_type='cumulative', display=display)
    
    print(SUCCESS_MESSAGE)
    print(FILES_SAVED_MESSAGE)
    print(f"  - {OUTPUT_CHART_FILE} (Cumulative)")
