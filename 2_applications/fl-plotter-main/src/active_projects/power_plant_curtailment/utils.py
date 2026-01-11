"""
Utility functions for power plant analysis
Contains helper functions extracted from power_plant_hourly_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import yaml
from datetime import datetime
import pvlib
import numpy as np


def load_constants(constants_file='constants.yaml'):
    """Load constants from YAML file"""
    with open(constants_file, 'r') as f:
        return yaml.safe_load(f)


def calculate_plant_configuration(plant_dir, constants):
    """
    Calculate all plant configuration dynamically based on actual data analysis.
    Eliminates need for config.yaml files by computing everything from plant data.
    
    Args:
        plant_dir (str): Directory containing plant data CSV
        constants (dict): Constants loaded from YAML
        
    Returns:
        dict: Complete configuration for plant analysis
    """
    # Extract plant key from directory name
    plant_key = os.path.basename(plant_dir.rstrip('/'))
    
    plant_locations = constants['plant_locations']
    if plant_key not in plant_locations:
        raise ValueError(f"Unknown plant: {plant_key}")
    
    plant_info = plant_locations[plant_key]
    
    # Find and read the CSV file
    csv_file = None
    for file in os.listdir(plant_dir):
        if file.endswith('_hourly_data.csv'):
            csv_file = os.path.join(plant_dir, file)
            break
    
    if not csv_file:
        raise FileNotFoundError(f"No hourly data CSV file found in {plant_dir}")
    
    # Read and analyze plant data
    df = pd.read_csv(csv_file)
    df['operating_datetime_utc'] = pd.to_datetime(df['operating_datetime_utc'])
    
    # Group by hour and sum all units for total generation
    hourly_totals = df.groupby('operating_datetime_utc')['gross_load_mw'].sum()
    
    # Calculate key plant statistics
    max_generation = hourly_totals.max()
    median_generation = hourly_totals.median()
    slack_median = max_generation - median_generation if not pd.isna(median_generation) else max_generation
    
    # Calculate datacenter target (50% of median slack for reliability)
    target_power_mw = max(100, round((slack_median * 0.5) / 50) * 50)  # Round to nearest 50 MW, min 100 MW
    
    # Calculate supporting infrastructure based on datacenter target
    solar_capacity_mw = int(target_power_mw * 0.4)  # 40% of target for renewable complement
    backup_units_count = max(10, int((target_power_mw * 0.4) / constants['generator_config']['unit_capacity_mw']))  # 40% backup capacity (doubled)
    
    # Extract plant ID from directory name
    plant_id = int(plant_key.split('_')[1])
    
    # Return complete dynamic configuration
    return {
        'plant': {
            'name': plant_info['name'],
            'id': plant_id,
            'location': plant_info['location'],
            'latitude': plant_info['latitude'],
            'longitude': plant_info['longitude']
        },
        'datacenter': {
            'target_power_mw': target_power_mw
        },
        'solar': {
            'max_capacity_mw': solar_capacity_mw
        },
        'backup_generators': {
            'units_count': backup_units_count
        },
        'output': {
            'chart_prefix': plant_info['chart_prefix']
        }
    }


def load_config(plant_directory, constants):
    """Load configuration - now uses dynamic calculation instead of YAML files"""
    return calculate_plant_configuration(plant_directory, constants)


def get_data_file(plant_directory):
    """Find the CSV data file in plant directory"""
    for file in os.listdir(plant_directory):
        if file.endswith('_hourly_data.csv'):
            return os.path.join(plant_directory, file)
    return None


def find_target_date(total_data):
    """
    Find the target analysis date - the 95th percentile peak generation day
    Strategy: Find days with consistently high generation for realistic analysis scenarios
    """
    print("  Analyzing daily generation peaks across full dataset...")
    
    # Group by date and get the maximum generation for each day
    total_data['date'] = total_data['operating_datetime_utc'].dt.date
    daily_peaks = total_data.groupby('date')['gross_load_mw'].max().reset_index()
    daily_peaks = daily_peaks.sort_values('gross_load_mw', ascending=False)
    
    # Calculate 95th percentile to find high-generation days
    percentile_95 = daily_peaks['gross_load_mw'].quantile(0.95)
    print(f"  95th percentile peak: {percentile_95:.0f} MW")
    
    # Filter to days at or above 95th percentile
    high_generation_days = daily_peaks[daily_peaks['gross_load_mw'] >= percentile_95]
    print(f"  Found {len(high_generation_days)} days at/above 95th percentile")
    
    if len(high_generation_days) == 0:
        # Fallback to highest single day if no days meet threshold
        target_date = daily_peaks.iloc[0]['date'] 
        target_peak = daily_peaks.iloc[0]['gross_load_mw']
        print(f"  Fallback to highest day: {target_date} (peak: {target_peak:.0f} MW)")
    else:
        # Select the first (highest) day from the qualifying high-generation days
        target_date = high_generation_days.iloc[0]['date']
        target_peak = high_generation_days.iloc[0]['gross_load_mw'] 
        print(f"  Selected target date: {target_date} (peak: {target_peak:.0f} MW)")
    
    return target_date


def get_day_data(total_data, target_date):
    """Extract 24-hour data for the target date"""
    day_data = total_data[total_data['operating_datetime_utc'].dt.date == target_date].copy()
    day_data = day_data.sort_values('operating_datetime_utc')
    
    print(f"  Extracted {len(day_data)} hourly records for {target_date}")
    
    if len(day_data) != 24:
        print(f"  Warning: Expected 24 hours of data, got {len(day_data)} records")
        # Check if we have partial day data and fill missing hours with 0
        if len(day_data) > 0:
            print(f"  Data range: {day_data['operating_datetime_utc'].min().strftime('%H:%M')} to {day_data['operating_datetime_utc'].max().strftime('%H:%M')}")
    
    return day_data


def create_hour_labels():
    """Create 24-hour labels starting from 7 AM (7AM-7AM cycle)"""
    hours = []
    for hour in range(7, 31):  # 7 AM to 6 AM next day
        if hour <= 24:
            display_hour = hour
        else:
            display_hour = hour - 24
        
        if display_hour == 0:
            hours.append("12AM")  # Midnight
        elif display_hour > 12:
            hours.append(f"{display_hour - 12}PM")
        elif display_hour == 12:
            hours.append("12PM")
        else:
            hours.append(f"{display_hour}AM")
    
    return hours


def reorder_to_7am_cycle(hourly_data):
    """
    Reorder hourly data from natural 0-23 (midnight-11PM) to 7AM-6AM cycle.
    
    Args:
        hourly_data: Array-like data with 24 elements indexed 0-23 (hour 0 = midnight)
        
    Returns:
        Reordered data where index 0 = 7 AM, index 1 = 8 AM, ..., index 17 = midnight, etc.
    """
    if len(hourly_data) != 24:
        raise ValueError(f"Expected 24 hours of data, got {len(hourly_data)}")
    
    # Convert to numpy array for easier indexing
    data_array = np.array(hourly_data)
    
    # Reorder: hours 7-23 followed by hours 0-6
    # Index mapping: 0->7AM, 1->8AM, ..., 16->11PM, 17->12AM, ..., 23->6AM
    reordered = np.concatenate([data_array[7:24], data_array[0:7]])
    
    return reordered


def simulate_battery_storage(slack_capacity, solar_generation, constants):
    """
    Simulate battery storage system that charges from excess and discharges during shortfalls
    
    Args:
        slack_capacity: Hourly plant slack capacity
        solar_generation: Hourly solar generation
        constants: Constants from YAML including battery configuration
        
    Returns:
        list: Hourly battery contribution (positive = discharge, negative = charge)
    """
    battery_config = constants['battery_config']
    
    # Get datacenter target from global CONFIG (this needs to be passed or handled differently)
    # For now, we'll calculate it based on typical patterns
    max_slack = max(slack_capacity) if isinstance(slack_capacity, list) else slack_capacity.max()
    target_mw = max_slack * 0.5  # Conservative estimate
    
    # Calculate dynamic battery parameters based on datacenter target
    battery_capacity_mwh = target_mw * battery_config['hours_backup']
    max_charge_rate = target_mw * battery_config['charge_rate_ratio'] 
    max_discharge_rate = target_mw * battery_config['discharge_rate_ratio']
    
    battery_storage = [0] * 24
    battery_level = battery_capacity_mwh * 0.5  # Start at 50% charge
    
    for i in range(24):
        # Current available capacity before battery
        current_capacity = (slack_capacity[i] if isinstance(slack_capacity, list) 
                          else slack_capacity.iloc[i]) + solar_generation[i]
        
        # Determine if we need to charge or discharge
        gap = target_mw - current_capacity
        
        if gap > 0:  # Need more power - discharge battery
            discharge_amount = min(gap, max_discharge_rate, battery_level)
            battery_storage[i] = discharge_amount
            battery_level -= discharge_amount
        elif gap < -20:  # Excess power available - charge battery (only if significant excess)
            available_space = battery_capacity_mwh - battery_level
            charge_amount = min(abs(gap), max_charge_rate, available_space)
            battery_storage[i] = -charge_amount  # Negative indicates charging
            battery_level += charge_amount
        
        # Apply hourly degradation
        battery_level = max(0, battery_level - battery_config['degradation_mw_per_hour'])
    
    # Convert charging (negative values) to zero for stacked area chart
    # Only show discharge contribution in the chart
    return [max(0, x) for x in battery_storage]


def generate_realistic_solar_data(target_date, latitude, longitude, max_capacity_mw):
    """
    Generate realistic solar generation data using pvlib for accurate solar modeling.
    Uses clear-sky Global Horizontal Irradiance (GHI) with location-specific sun position.
    
    Args:
        target_date: Date for solar analysis
        latitude: Plant latitude 
        longitude: Plant longitude
        max_capacity_mw: Maximum solar farm capacity
        
    Returns:
        list: 24 hourly solar generation values (MW)
    """
    # Convert target_date to datetime if it's a date object
    if hasattr(target_date, 'strftime'):
        start_time = datetime.combine(target_date, datetime.min.time())
    else:
        start_time = target_date
    
    # Create location object for solar calculations
    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz='UTC')
    
    # Generate 24 hours of time data
    times = pd.date_range(start=start_time, periods=24, freq='h', tz='UTC')
    
    # Get clear-sky irradiance data
    clearsky = location.get_clearsky(times)
    
    # Extract Global Horizontal Irradiance (GHI) in W/m²
    ghi = clearsky['ghi']  
    
    # Convert GHI to capacity factors (0-1 range)
    # Typical solar panel efficiency under standard test conditions is ~1000 W/m²
    capacity_factors = ghi / 1000.0
    capacity_factors = capacity_factors.clip(0, 1)  # Ensure 0-1 range
    
    # Convert to actual generation values
    solar_generation = (capacity_factors * max_capacity_mw).tolist()
    
    return solar_generation


def finish_chart(fig, ax, plot_buddy, title=None, subtitle=None, source_citation=None):
    """Common chart finishing operations"""
    if title:
        plot_buddy.add_titles(ax, title, subtitle)
    
    if source_citation:
        plot_buddy.add_source_citation(fig, source_citation)
    
    # Add logo if available
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../logos/gs_logo.png')
    if os.path.exists(logo_path):
        plot_buddy.add_logo(fig, logo_path)
    
    plot_buddy.apply_tight_layout(fig)


def create_capacity_stack_chart(layers, target_mw, title, subtitle, plot_buddy, annotations=None, show_target_analysis=True):
    """
    Create a unified stacked area chart for capacity visualization
    
    Args:
        layers: List of layer dictionaries with 'data', 'label', and 'color' keys
        target_mw: Target capacity line
        title: Chart title
        subtitle: Chart subtitle  
        plot_buddy: PlotBuddy instance
        annotations: Optional list of annotation dictionaries
        show_target_analysis: Whether to show target capacity analysis
        
    Returns:
        tuple: (figure, axis) objects
    """
    x_values = list(range(24))
    hour_labels = create_hour_labels()
    
    fig, ax = plot_buddy.setup_figure(figsize=plot_buddy.wide_figure)
    
    # Create stacked areas
    bottom = np.zeros(24)
    for layer in layers:
        ax.fill_between(x_values, bottom, bottom + layer['data'], 
                       alpha=0.7, label=layer['label'], color=layer['color'])
        bottom += layer['data']
    
    # Calculate and plot total capacity line
    total_capacity = bottom  # Sum of all layers
    ax.plot(x_values, total_capacity, 
           color='black', linewidth=3, 
           marker='o', markersize=6,
           alpha=1.0, label='Total Available Capacity')
    
    # Add target capacity line
    ax.axhline(y=target_mw, color='red', linestyle='--', linewidth=2, 
               label=f'Datacenter Target Capacity: {target_mw:.0f} MW')
    
    # Minimum capacity coverage analysis (worst-case scenario)
    if show_target_analysis:
        min_capacity = min(total_capacity)
        min_coverage_percentage = round((min_capacity / target_mw) * 100)
        
        if min_coverage_percentage >= 70:
            analysis_text = f"{min_coverage_percentage}% Worst Case Capacity"
            color = 'green'
        elif min_coverage_percentage >= 30:
            analysis_text = f"{min_coverage_percentage}% Worst Case Capacity"
            color = 'orange' 
        else:
            analysis_text = f"{min_coverage_percentage}% Worst Case Capacity"
            color = 'red'
        
        ax.text(0.02, 0.98, analysis_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color=color, weight='bold')
    
    # Add any custom annotations
    if annotations:
        for ann in annotations:
            ax.annotate(ann['text'], xy=ann['xy'], xytext=ann['xytext'],
                       bbox=ann.get('bbox', dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)),
                       arrowprops=ann.get('arrowprops', dict(arrowstyle='->', color='black')))
    
    # Configure chart
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Power Generation (MW)')
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([hour_labels[i] for i in range(0, 24, 3)])
    ax.grid(True, alpha=0.3)
    
    # Add legend at bottom using PlotBuddy's create_legend method
    total_items = len(layers) + 2  # +2 for total capacity line and target line
    if total_items <= 3:
        ncol = total_items
    elif total_items <= 6:  
        ncol = 3
    else:
        ncol = 4
    
    plot_buddy.create_legend(ax, ncol=ncol, position='bottom')
    
    finish_chart(fig, ax, plot_buddy, title=title, subtitle=subtitle, 
                source_citation='EPA Clean Air Markets Program Data')
    
    return fig, ax