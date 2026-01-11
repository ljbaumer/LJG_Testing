"""
Power Plant Hourly Generation Analysis - Main Charting Module
Visualizes hourly power generation data using PlotBuddy with GS wide style
Refactored to use separate constants.yaml and utils.py modules
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import sys
import argparse
from datetime import datetime
import numpy as np

# Import our modular components
from utils import (
    load_constants, load_config, get_data_file, find_target_date, 
    get_day_data, create_hour_labels, reorder_to_7am_cycle, simulate_battery_storage,
    generate_realistic_solar_data, finish_chart, create_capacity_stack_chart
)

# Add utils directory to path to find plot_buddy
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '../../utils')
sys.path.insert(0, utils_dir)
from plot_buddy import PlotBuddy

# Global variables (will be loaded from YAML and configuration) 
CONSTANTS = None
CONFIG = None
PLANT_DIR = None
PLOT_BUDDY = None


def load_and_process_data():
    """Load and process plant CSV data into pivoted format"""
    data_file = get_data_file(PLANT_DIR)
    if not data_file:
        raise FileNotFoundError(f"No data file found in {PLANT_DIR}")
    
    # Read data
    df = pd.read_csv(data_file)
    
    # Convert datetime column
    df['operating_datetime_utc'] = pd.to_datetime(df['operating_datetime_utc'])
    
    # Group by datetime and sum all units for total power
    total_data = df.groupby('operating_datetime_utc').agg({
        'gross_load_mw': 'sum'
    }).reset_index()
    
    # Create pivot table for visualization (hours vs days)
    total_data['date'] = total_data['operating_datetime_utc'].dt.date
    total_data['hour'] = total_data['operating_datetime_utc'].dt.hour
    total_data['total_mw'] = total_data['gross_load_mw']
    
    # Create pivot table with hours as index, dates as columns
    pivot_data = total_data.pivot_table(
        index='hour', 
        columns='date', 
        values='total_mw',
        fill_value=0
    )
    
    return pivot_data, total_data


def create_daily_peaks_chart(pivot_data, total_data, plot_buddy):
    """Create chart showing daily peak generation throughout the year"""
    fig, ax = plot_buddy.setup_figure(figsize=plot_buddy.wide_figure)
    
    # Calculate daily peaks (max generation per day)
    daily_peaks = total_data.groupby('date')['total_mw'].max().reset_index()
    daily_peaks = daily_peaks.sort_values('date')
    
    # Convert dates for plotting
    dates = pd.to_datetime(daily_peaks['date'])
    peaks = daily_peaks['total_mw']
    
    # Create the line chart
    ax.plot(dates, peaks, linewidth=2, color='#1f77b4', alpha=0.8)
    ax.fill_between(dates, 0, peaks, alpha=0.3, color='#1f77b4')
    
    # Add statistics annotation
    max_peak = peaks.max()
    min_peak = peaks.min()  
    avg_peak = peaks.mean()
    
    stats_text = f"""Max Daily Peak: {max_peak:.0f} MW
Min Daily Peak: {min_peak:.0f} MW  
Avg Daily Peak: {avg_peak:.0f} MW"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format axes
    ax.set_xlabel(CONSTANTS['chart_config']['x_axis_label'])
    ax.set_ylabel(CONSTANTS['chart_config']['y_axis_label'])
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show months
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # Get title and subtitle
    titles = CONSTANTS['chart_titles_subtitles']['daily_peaks']
    title = titles[0]
    subtitle = titles[1].format(plant_name=CONFIG['plant']['name'])
    
    finish_chart(fig, ax, plot_buddy, title=title, subtitle=subtitle, 
                source_citation=CONSTANTS['chart_config']['source_citation'])
    
    return fig, ax


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate power plant analysis charts from CSV data using dynamic configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python main_analysis.py plant_55286_Oleander/
  python main_analysis.py plant_8042_Belews_Creek/
  '''
    )
    parser.add_argument(
        'plant_directory',
        help='Directory containing plant data (CSV file)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Skip displaying charts (useful for batch processing)'
    )
    return parser.parse_args()


def main():
    """Main function to generate the power plant analysis"""
    global CONSTANTS, CONFIG, PLANT_DIR, PLOT_BUDDY
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load constants from YAML
    print("Loading constants...")
    CONSTANTS = load_constants()
    
    # Set up global variables
    PLANT_DIR = args.plant_directory
    
    # Calculate dynamic configuration (eliminates need for config.yaml)
    print(f"Calculating dynamic configuration for {PLANT_DIR}...")
    CONFIG = load_config(PLANT_DIR, CONSTANTS)
    
    # Initialize shared PlotBuddy instance
    PLOT_BUDDY = PlotBuddy(style_dir_path=os.path.join(utils_dir, 'styles'))
    
    # Set up output directory
    output_dir = os.path.join(PLANT_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames based on plant prefix
    chart_prefix = CONFIG['output']['chart_prefix']
    output_files = {
        'daily_peaks': f'1_{chart_prefix}_daily_peaks.png',
        'daily_cycle': f'2_{chart_prefix}_power_daily_cycle.png',
        'slack_capacity': f'3_{chart_prefix}_slack_capacity.png',
        'solar_potential': f'4_{chart_prefix}_solar_potential.png',
        'combined_capacity': f'5_{chart_prefix}_combined_capacity.png',
        'battery_storage': f'6_{chart_prefix}_battery_storage.png',
        'backup_generators': f'7_{chart_prefix}_backup_generators.png'
    }
    
    # Load and process data once
    print("Loading power plant data...")
    pivot_data, total_data = load_and_process_data()
    
    # Find target date (95th percentile peak generation day) for all daily analysis charts
    print("Finding target date for analysis (95th percentile peak generation day)...")
    target_date = find_target_date(total_data)
    
    # Create the daily peaks chart (doesn't need target date)
    print("Creating daily peaks chart...")
    fig, ax = create_daily_peaks_chart(pivot_data, total_data, PLOT_BUDDY)
    
    # Save the chart
    output_path = os.path.join(output_dir, output_files['daily_peaks'])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    
    # Extract day data for target date and prepare common derived data
    print(f"Preparing data for target date analysis...")
    day_data = get_day_data(total_data, target_date)
    peak_capacity = day_data['total_mw'].max()
    slack_capacity = peak_capacity - day_data['total_mw']
    
    print(f"Generating solar and battery storage data...")
    solar_generation = generate_realistic_solar_data(
        target_date=target_date,
        latitude=CONFIG['plant']['latitude'],
        longitude=CONFIG['plant']['longitude'], 
        max_capacity_mw=CONFIG['solar']['max_capacity_mw']
    )
    battery_storage = simulate_battery_storage(slack_capacity, solar_generation, CONSTANTS)
    
    # Natural gas generation calculation  
    natgas_generation = np.zeros(24)
    backup_gas_units_count = CONFIG['backup_generators']['units_count']
    backup_gas_total_capacity = CONSTANTS['generator_config']['unit_capacity_mw'] * backup_gas_units_count
    target_mw = CONFIG['datacenter']['target_power_mw']
    
    for i in range(24):
        current_total = slack_capacity.values[i] + solar_generation[i] + battery_storage[i]
        remaining_gap = max(0, target_mw - current_total)
        if remaining_gap > 0:
            natgas_generation[i] = min(remaining_gap, backup_gas_total_capacity)
    
    # Reorder all hourly data to 7AM-6AM cycle for proper chart display alignment
    print("Reordering hourly data to 7AM-6AM cycle for chart display...")
    day_data_7am = day_data.copy()
    day_data_7am['total_mw'] = reorder_to_7am_cycle(day_data['total_mw'].values)
    slack_capacity_7am = reorder_to_7am_cycle(slack_capacity.values)  
    solar_generation_7am = reorder_to_7am_cycle(solar_generation)
    battery_storage_7am = reorder_to_7am_cycle(battery_storage)
    natgas_generation_7am = reorder_to_7am_cycle(natgas_generation)
    
    # Calculate costs for enhanced legend labels (all dynamic now)
    solar_capacity_mw = CONFIG['solar']['max_capacity_mw']
    battery_capacity_mwh = target_mw * CONSTANTS['battery_config']['hours_backup']
    natgas_capacity_mw = backup_gas_total_capacity
    
    solar_cost_raw = solar_capacity_mw * CONSTANTS['costs']['solar_cost_per_mw']
    battery_cost_raw = battery_capacity_mwh * CONSTANTS['costs']['battery_cost_per_mwh']
    natgas_cost_raw = natgas_capacity_mw * CONSTANTS['costs']['natgas_generator_cost_per_mw']
    
    # Convert to millions for legend display
    solar_cost_millions = solar_cost_raw / 1_000_000
    battery_cost_millions = battery_cost_raw / 1_000_000
    natgas_cost_millions = natgas_cost_raw / 1_000_000
    
    # Define layer configurations for each chart type with enhanced labels
    # Common variables for subtitle formatting
    date_range = target_date.strftime('%B %d, %Y')
    plant_name = CONFIG['plant']['name']
    
    chart_configs = [
        # Chart 2: Daily cycle (plant generation only)
        {
            'name': 'daily_cycle',
            'layers': [{'data': day_data_7am['total_mw'], 'label': 'Total Generation', 'color': '#1f77b4'}],
            'title': CONSTANTS['chart_titles_subtitles']['daily_cycle'][0],
            'subtitle': CONSTANTS['chart_titles_subtitles']['daily_cycle'][1].format(plant_name=plant_name, date_range=date_range),
            'show_target_analysis': False
        },
        # Chart 3: Slack capacity 
        {
            'name': 'slack_capacity',
            'layers': [{'data': slack_capacity_7am, 'label': 'Available Slack Capacity', 'color': '#2ca02c'}],
            'title': CONSTANTS['chart_titles_subtitles']['slack_capacity'][0],
            'subtitle': CONSTANTS['chart_titles_subtitles']['slack_capacity'][1].format(plant_name=plant_name, date_range=date_range),
            'show_target_analysis': False
        },
        # Chart 4: Solar potential
        {
            'name': 'solar_potential', 
            'layers': [{'data': solar_generation_7am, 'label': f'Solar Generation Potential ({solar_capacity_mw} MW, ${solar_cost_millions:.0f}M)', 'color': '#FFD700'}],
            'title': CONSTANTS['chart_titles_subtitles']['solar_potential'][0],
            'subtitle': CONSTANTS['chart_titles_subtitles']['solar_potential'][1].format(plant_name=plant_name, date_range=date_range),
            'show_target_analysis': False
        },
        # Chart 5: Combined capacity (plant slack + solar)
        {
            'name': 'combined_capacity',
            'layers': [
                {'data': slack_capacity_7am, 'label': CONSTANTS['legend_labels']['plant_slack'], 'color': '#2ca02c'},
                {'data': solar_generation_7am, 'label': CONSTANTS['legend_labels']['solar'].format(capacity=solar_capacity_mw, cost=solar_cost_millions), 'color': '#FFD700'}
            ],
            'title': CONSTANTS['chart_titles_subtitles']['combined_capacity'][0],
            'subtitle': CONSTANTS['chart_titles_subtitles']['combined_capacity'][1].format(plant_name=plant_name, date_range=date_range)
        },
        # Chart 6: Battery storage (plant slack + solar + battery)
        {
            'name': 'battery_storage',
            'layers': [
                {'data': slack_capacity_7am, 'label': CONSTANTS['legend_labels']['plant_slack'], 'color': '#2ca02c'},
                {'data': solar_generation_7am, 'label': CONSTANTS['legend_labels']['solar'].format(capacity=solar_capacity_mw, cost=solar_cost_millions), 'color': '#FFD700'},
                {'data': battery_storage_7am, 'label': CONSTANTS['legend_labels']['battery'].format(capacity=battery_capacity_mwh, cost=battery_cost_millions), 'color': '#FF69B4'}
            ],
            'title': CONSTANTS['chart_titles_subtitles']['battery_storage'][0],
            'subtitle': CONSTANTS['chart_titles_subtitles']['battery_storage'][1].format(plant_name=plant_name, date_range=date_range)
        },
        # Chart 7: Complete backup system (plant slack + solar + battery + natural gas)
        {
            'name': 'backup_generators',
            'layers': [
                {'data': slack_capacity_7am, 'label': CONSTANTS['legend_labels']['plant_slack'], 'color': '#2ca02c'},
                {'data': solar_generation_7am, 'label': CONSTANTS['legend_labels']['solar'].format(capacity=solar_capacity_mw, cost=solar_cost_millions), 'color': '#FFD700'},
                {'data': battery_storage_7am, 'label': CONSTANTS['legend_labels']['battery'].format(capacity=battery_capacity_mwh, cost=battery_cost_millions), 'color': '#FF69B4'},
                {'data': natgas_generation_7am, 'label': CONSTANTS['legend_labels']['backup_gas'].format(capacity=natgas_capacity_mw, cost=natgas_cost_millions), 'color': '#FF4500'}
            ],
            'title': CONSTANTS['chart_titles_subtitles']['backup_generators'][0],
            'subtitle': CONSTANTS['chart_titles_subtitles']['backup_generators'][1].format(plant_name=plant_name, date_range=date_range)
        }
    ]
    
    # Generate all capacity charts using unified function
    for config in chart_configs:
        print(f"Creating {config['name']} chart...")
        fig, ax = create_capacity_stack_chart(
            layers=config['layers'],
            target_mw=target_mw,
            title=config['title'], 
            subtitle=config['subtitle'],
            plot_buddy=PLOT_BUDDY,
            show_target_analysis=config.get('show_target_analysis', True)
        )
        
        # Save the chart
        output_path = os.path.join(output_dir, output_files[config['name']])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{config['name']} chart saved to: {output_path}")
    
    # Display the chart (optional)
    if not args.no_display:
        plt.show()
        
    print(f"\nAll charts generated successfully for {CONFIG['plant']['name']}!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()