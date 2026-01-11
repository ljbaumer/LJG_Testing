
"""
Power Plant Hourly Generation Analysis
Visualizes hourly power generation data using PlotBuddy with GS wide style
Configurable for any power plant via YAML CONFIGuration files
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import sys
import yaml
import argparse
from datetime import datetime
import pvlib
import numpy as np

# Add utils directory to path to find plot_buddy
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '../../utils')
sys.path.insert(0, utils_dir)
from plot_buddy import PlotBuddy

# Global CONFIG variable (will be loaded from YAML) 
CONFIG = None
PLANT_DIR = None
PLOT_BUDDY = None

# Load constants from YAML file
with open(os.path.join(current_dir, 'constants.yaml'), 'r') as f:
    constants = yaml.safe_load(f)

# Assign constants to global variables
Y_AXIS_LABEL = constants['chart_config']['y_axis_label']
X_AXIS_LABEL = constants['chart_config']['x_axis_label']
SOURCE_CITATION = constants['chart_config']['source_citation']
CHART_TITLES_SUBTITLES = constants['chart_titles_subtitles']
LEGEND_LABELS = constants['legend_labels']
COSTS = constants['costs']
BATTERY_CONFIG = constants['battery_config']
BACKUP_GENERATOR_CONFIG = constants['backup_generator_config']
PLANT_LOCATIONS = constants['plant_locations']
DATACENTER_CONFIG = constants['datacenter_config']
SOLAR_CONFIG = constants['solar_config']

# Extract specific cost values
SOLAR_COST_PER_MW = COSTS['solar_cost_per_mw']
BATTERY_COST_PER_MWH = COSTS['battery_cost_per_mwh']
NATGAS_GENERATOR_COST_PER_MW = COSTS['natgas_generator_cost_per_mw']
NATGAS_FUEL_COST_PER_MWH = COSTS['natgas_fuel_cost_per_mwh']

# Extract specific battery values
BATTERY_HOURS_BACKUP = BATTERY_CONFIG['hours_backup']
BATTERY_CHARGE_RATE_RATIO = BATTERY_CONFIG['charge_rate_ratio']
BATTERY_DISCHARGE_RATE_RATIO = BATTERY_CONFIG['discharge_rate_ratio']
BATTERY_DEGRADATION_MW_PER_HOUR = BATTERY_CONFIG['degradation_mw_per_hour']

# Extract specific generator values
GAS_BACKUP_UNIT_CAPACITY_MW = BACKUP_GENERATOR_CONFIG['unit_capacity_mw']
BACKUP_GENERATOR_CAPACITY_PERCENT_TARGET = BACKUP_GENERATOR_CONFIG['capacity_percent_target']

# Extract specific datacenter values
DATACENTER_TARGET_PERCENT_MAX = DATACENTER_CONFIG['target_percent_max']

# Extract specific solar values
SOLAR_CAPACITY_PERCENT_TARGET = SOLAR_CONFIG['capacity_percent_target']

# Chart configuration now loaded from constants.yaml
# (Legacy constants removed - using centralized configuration)
# All constants now loaded from constants.yaml

def calculate_plant_configuration(plant_dir):
    """
    Calculate all plant configuration dynamically based on actual data analysis.
    Eliminates need for config.yaml files by computing everything from plant data.
    
    Args:
        plant_dir (str): Directory containing plant data CSV
        
    Returns:
        dict: Complete configuration for plant analysis
    """
    # Extract plant key from directory name
    plant_key = os.path.basename(plant_dir.rstrip('/'))
    
    if plant_key not in PLANT_LOCATIONS:
        raise ValueError(f"Unknown plant: {plant_key}")
    
    plant_info = PLANT_LOCATIONS[plant_key]
    
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
    
    # Calculate datacenter target (X% of median slack for reliability from the constant page)
    target_power_mw = max(100, round((slack_median * DATACENTER_TARGET_PERCENT_MAX) / 50) * 50)  # Round to nearest 50 MW, min 100 MW
    
    # Calculate supporting infrastructure based on datacenter target
    solar_capacity_mw = 0 # Will be calculated later
    backup_units_count = 0 # Will be calculated later
    
    # Extract plant ID from directory name
    plant_id = int(plant_key.split('_')[1])
    
    return {
        'plant': {
            'name': plant_info['name'],
            'id': plant_id,
            'location': plant_info['location'],
            'latitude': plant_info['latitude'],
            'longitude': plant_info['longitude']
        },
        'solar': {
            'max_capacity_mw': solar_capacity_mw
        },
        'backup_generators': {
            'units_count': backup_units_count
        },
        'output': {
            'chart_prefix': plant_info['chart_prefix']
        },
        # Include analysis stats for reference
        '_analysis_stats': {
            'max_generation': max_generation,
            'median_generation': median_generation,
            'slack_median': slack_median
        }
    }

def load_config(plant_directory):
    """Load configuration from YAML file"""
    config_path = os.path.join(plant_directory, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate solar timing
    sunrise = config['solar_timing']['sunrise_hour']
    sunset = config['solar_timing']['sunset_hour']
    
    if sunrise >= sunset:
        raise ValueError(f"Invalid solar timing: sunrise ({sunrise}) must be before sunset ({sunset})")
    if sunrise < 0 or sunrise > 24:
        raise ValueError(f"Invalid sunrise hour: {sunrise} (must be between 0 and 24)")
    if sunset < 0 or sunset > 24:
        raise ValueError(f"Invalid sunset hour: {sunset} (must be between 0 and 24)")
    
    return config

def get_data_file(plant_directory):
    """Find the CSV data file in the plant directory"""
    for file in os.listdir(plant_directory):
        if file.endswith('_hourly_data.csv'):
            return os.path.join(plant_directory, file)
    raise FileNotFoundError(f"No hourly data CSV file found in {plant_directory}")


# Helper functions for reducing chart code repetition
def find_target_date(total_data):
    """Find the target date for analysis (95th percentile peak generation day)
    
    This function analyzes the entire dataset to find the day with peak generation
    that represents high-demand conditions for datacenter analysis. Prefers 
    summer (June-August) or winter (Dec-Feb) months when there are ties.
    
    Args:
        total_data: DataFrame with total generation data
    
    Returns:
        datetime.date: The target date for analysis
    """
    print("  Analyzing daily generation peaks across full dataset...")
    
    # Calculate daily peak (maximum hourly generation for each day)
    daily_peaks = total_data.set_index('operating_datetime_utc').resample('D')['total_mw'].max()
    
    # Find the 95th percentile peak value
    peak_95th = daily_peaks.quantile(0.95)
    
    # Find days that are at or above the 95th percentile
    high_peak_days = daily_peaks[daily_peaks >= peak_95th]
    
    print(f"  95th percentile peak: {peak_95th:.0f} MW")
    print(f"  Found {len(high_peak_days)} days at/above 95th percentile")
    
    # If multiple days have the same highest peak, prefer summer or winter months
    max_peak_value = high_peak_days.max()
    max_peak_days = high_peak_days[high_peak_days == max_peak_value]
    
    if len(max_peak_days) > 1:
        print(f"  Found {len(max_peak_days)} days with peak value {max_peak_value:.0f} MW")
        
        # Create season preference weights: summer (Jun-Aug) and winter (Dec-Feb) get higher priority
        seasonal_scores = {}
        for date_idx in max_peak_days.index:
            month = date_idx.month
            if month in [6, 7, 8]:  # Summer months
                seasonal_scores[date_idx] = 3  # High priority
            elif month in [12, 1, 2]:  # Winter months  
                seasonal_scores[date_idx] = 3  # High priority
            elif month in [9, 10, 11] or month in [3, 4, 5]:  # Fall/Spring
                seasonal_scores[date_idx] = 2  # Medium priority
            else:
                seasonal_scores[date_idx] = 1  # Low priority
        
        # Sort by seasonal preference, then by date (latest first as tiebreaker)
        best_date = max(seasonal_scores.keys(), key=lambda x: (seasonal_scores[x], x))
        target_date = best_date.date()
        
        season_name = ""
        if best_date.month in [6, 7, 8]:
            season_name = "summer"
        elif best_date.month in [12, 1, 2]:
            season_name = "winter"
        elif best_date.month in [9, 10, 11]:
            season_name = "fall"
        else:
            season_name = "spring"
            
        print(f"  Selected {season_name} date: {target_date} (peak: {max_peak_value:.0f} MW)")
    else:
        target_date = high_peak_days.idxmax().date()
        print(f"  Selected target date: {target_date} (peak: {max_peak_value:.0f} MW)")
    
    return target_date

def get_day_data(total_data, target_date):
    """Extract 24-hour data slice for a specific date (7 AM to 7 AM next day)
    
    Args:
        total_data: DataFrame with total generation data
        target_date: datetime.date object for the target analysis date
    
    Returns:
        pd.DataFrame: Filtered data for the 24-hour period
    """
    from datetime import datetime, timedelta
    
    # Create 24-hour range from 7 AM to 7 AM next day
    start_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=7)
    end_datetime = start_datetime + timedelta(hours=24)
    
    # Filter data for the 24-hour period
    day_mask = (total_data['operating_datetime_utc'] >= start_datetime) & \
               (total_data['operating_datetime_utc'] < end_datetime)
    day_data = total_data[day_mask].copy()
    
    if len(day_data) == 0:
        print(f"  Warning: No data found for {target_date} in 7AM-7AM range")
    else:
        print(f"  Extracted {len(day_data)} hourly records for {target_date}")
    
    return day_data

def create_hour_labels():
    """Create standard hour labels for 7 AM to 7 AM display
    
    Returns:
        list: Hour labels for 24-hour period starting at 7 AM
    """
    hour_labels = []
    for i in range(24):
        hour = (i + 7) % 24
        if hour == 0:
            hour_labels.append('12 AM')
        elif hour < 12:
            hour_labels.append(f'{hour} AM')
        elif hour == 12:
            hour_labels.append('12 PM')
        else:
            hour_labels.append(f'{hour - 12} PM')
    
    return hour_labels

def simulate_battery_storage(slack_capacity, solar_generation):
    """Centralized battery storage simulation logic
    
    Returns:
        numpy.ndarray: Battery discharge capacity for each hour (MW)
    """
    import numpy as np
    
    # Calculate battery configuration dynamically based on datacenter target
    target_power = CONFIG['datacenter']['target_power_mw']
    max_battery_capacity_mwh = target_power * BATTERY_HOURS_BACKUP
    max_charge_rate_mw = target_power * BATTERY_CHARGE_RATE_RATIO
    max_discharge_rate_mw = target_power * BATTERY_DISCHARGE_RATE_RATIO
    battery_degradation_mw_per_hour = BATTERY_DEGRADATION_MW_PER_HOUR
    
    # Initialize battery simulation
    battery_storage = np.zeros(24)
    battery_charge_level_mwh = 0
    
    # Simulate battery charge/discharge for each hour
    for i in range(24):
        current_capacity = slack_capacity.values[i] + solar_generation[i]
        gap_to_target = max(0, target_power - current_capacity)
        
        # Charge battery when there's excess capacity
        if current_capacity > target_power and battery_charge_level_mwh < max_battery_capacity_mwh:
            excess_capacity = current_capacity - target_power
            charge_rate_mw = min(max_charge_rate_mw, excess_capacity)
            charge_energy_mwh = charge_rate_mw
            battery_charge_level_mwh = min(max_battery_capacity_mwh, battery_charge_level_mwh + charge_energy_mwh)
        
        # Discharge battery when capacity is below target
        elif gap_to_target > 0 and battery_charge_level_mwh > 0:
            max_discharge_mw = min(gap_to_target, max_discharge_rate_mw, battery_charge_level_mwh)
            discharge_energy_mwh = max_discharge_mw
            battery_charge_level_mwh = max(0, battery_charge_level_mwh - discharge_energy_mwh)
            battery_storage[i] = max_discharge_mw
        
        # Apply natural battery degradation
        degradation_mwh = battery_degradation_mw_per_hour
        battery_charge_level_mwh = max(0, battery_charge_level_mwh - degradation_mwh)
    
    return battery_storage

def finish_chart(fig, ax, plot_buddy, title=None, subtitle=None, source_citation=None):
    """Unified chart finishing: titles, logo, source, layout (legends handled separately)
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axis
        plot_buddy: PlotBuddy instance
        title: Chart title (optional)
        subtitle: Chart subtitle (optional) 
        source_citation: Custom source citation (optional, defaults to CONFIG citation)
    """
    # Add titles if provided
    if title and subtitle:
        plot_buddy.add_titles(ax, title, subtitle)
    
    # Add logo if available
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../logos/gs_logo.png')
    if os.path.exists(logo_path):
        plot_buddy.add_logo(fig, logo_path)
    
    # Add source citation
    citation = source_citation or SOURCE_CITATION
    plot_buddy.add_source_citation(fig, citation)
    
    # Apply tight layout
    plot_buddy.apply_tight_layout(fig)

def create_capacity_stack_chart(layers, target_mw, title, subtitle, plot_buddy, annotations=None, show_target_analysis=True):
    """Unified capacity stack chart for all 24-hour capacity visualizations
    
    Args:
        layers: List of dicts with keys: data (array), label (str), color (str)
        target_mw: Target datacenter capacity to draw as red line
        title: Chart title
        subtitle: Chart subtitle
        plot_buddy: PlotBuddy instance
        annotations: Optional dict with custom annotations
        show_target_analysis: If True, show target line and worst case analysis
    
    Returns:
        tuple: (fig, ax)
    """
    import numpy as np
    
    buddy = plot_buddy
    
    # Create standard chart elements
    x_values = list(range(24))
    hour_labels = create_hour_labels()
    
    # Use the gs_wide style
    with buddy.get_style_context('gs_wide'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Extract data arrays and labels from layers
        layer_data = [layer['data'] for layer in layers]
        layer_labels = [layer['label'] for layer in layers]
        layer_colors = [layer['color'] for layer in layers]
        
        # Create stacked area chart or line chart depending on number of layers
        if len(layers) == 1:
            # Single layer - create filled line chart
            ax.plot(x_values, layer_data[0], 
                   label=layer_labels[0], 
                   color=layer_colors[0], 
                   linewidth=3, marker='o', markersize=8, alpha=0.9)
            ax.fill_between(x_values, 0, layer_data[0], alpha=0.2, color=layer_colors[0])
        else:
            # Multiple layers - create stacked area chart
            ax.stackplot(x_values, *layer_data, labels=layer_labels, colors=layer_colors, alpha=0.7)
        
        # Calculate total capacity
        total_capacity = np.sum(layer_data, axis=0)
        
        # Add outline for total capacity (if multiple layers) - make it more visually distinct
        if len(layers) > 1:
            ax.plot(x_values, total_capacity, 
                   color='black', linewidth=3, 
                   marker='o', markersize=6,
                   alpha=1.0, label='Total Available Capacity')
        
        # Add target line and worst case analysis (optional)
        if show_target_analysis:
            ax.axhline(y=target_mw, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Target: {target_mw} MW')
            
            # Find worst case (minimum capacity) and mark it
            min_capacity_idx = np.argmin(total_capacity)
            min_capacity_value = total_capacity[min_capacity_idx]
            min_capacity_percentage = (min_capacity_value / target_mw) * 100
            
            ax.annotate(f'Worst Case: {min_capacity_value:.0f} MW\n({min_capacity_percentage:.1f}% of target)',
                       xy=(min_capacity_idx, min_capacity_value),
                       xytext=(min_capacity_idx + 3, min_capacity_value - 50),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add custom annotations if provided
        if annotations:
            for annotation in annotations:
                ax.annotate(annotation['text'], 
                           xy=annotation['xy'], 
                           xytext=annotation['xytext'],
                           arrowprops=annotation.get('arrowprops', dict(arrowstyle='->', lw=2)),
                           fontsize=annotation.get('fontsize', 12),
                           fontweight=annotation.get('fontweight', 'bold'),
                           color=annotation.get('color', 'black'))
        
        # Set labels and formatting
        ax.set_xlabel('Hour of Day', fontsize=buddy.standard_font_size)
        ax.set_ylabel('Available Capacity (MW)', fontsize=buddy.standard_font_size)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([hour_labels[i] for i in range(0, 24, 2)], rotation=45, ha='right')
        ax.set_xlim(-0.5, 23.5)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Create legend with appropriate columns to force 2 rows
        if len(layers) == 1:  # Single layer charts
            buddy.create_legend(ax, position='bottom', ncol=1)  # 1 item in 1 row
        elif len(layers) == 2:  # Chart 5: Combined capacity (2 layers)
            buddy.create_legend(ax, position='bottom', ncol=1)  # 2 items in 2 rows = 1 column
        elif len(layers) == 3:  # Chart 6: Battery storage (3 layers)  
            buddy.create_legend(ax, position='bottom', ncol=2)  # 3 items in 2 rows = 2 columns (2 top, 1 bottom)
        elif len(layers) == 4:  # Chart 7: Backup generators (4 layers)
            buddy.create_legend(ax, position='bottom', ncol=2)  # 4 items in 2 rows = 2 columns (2 per row)
        else:  # Default case
            buddy.create_legend(ax, position='bottom', ncol=2)
        
        # Finish chart with unified function  
        finish_chart(fig, ax, buddy, title=title, subtitle=subtitle)
        
        return fig, ax

# don't love the dependency injection here, lets have this just take in sunrise and sunset hour so it's more atomic

def generate_realistic_solar_data(target_date, latitude, longitude, max_capacity_mw):
    """
    Generate realistic solar generation data using pvlib for accurate solar modeling.
    Returns data aligned to 7AM-7AM display (not midnight-midnight)
    
    Args:
        target_date: datetime.date object for the analysis date
        latitude: Plant latitude for solar calculations  
        longitude: Plant longitude for solar calculations
        max_capacity_mw: Maximum solar farm capacity (MW)
        
    Returns:
        tuple: (solar_generation_7am, max_capacity_mw)
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create pvlib Location object
    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz='UTC')
    
    # Create 24-hour time range for the target date (midnight to midnight UTC)
    start_time = datetime.combine(target_date, datetime.min.time())
    times = pd.date_range(start=start_time, periods=24, freq='h', tz='UTC')
    
    # Calculate clear-sky irradiance using pvlib
    clearsky = location.get_clearsky(times)
    
    # Get Global Horizontal Irradiance (GHI) in W/m²
    ghi = clearsky['ghi']
    
    # Convert irradiance to solar generation capacity
    # Standard Test Conditions (STC) for solar panels are at 1000 W/m²
    # Scale the GHI relative to STC to get capacity factor
    capacity_factors = ghi / 1000.0  # Normalize to 0-1 scale
    capacity_factors = capacity_factors.clip(0, 1)  # Ensure no values above 1
    
    # Calculate actual solar generation in MW
    solar_generation_full = capacity_factors * max_capacity_mw
    
    # Convert from midnight-midnight to 7AM-7AM to match chart display
    # Chart shows hours 7,8,9...23,0,1,2,3,4,5,6
    # So we need solar data for hours [7,8,9,...,23,0,1,2,3,4,5,6]
    solar_generation_7am = np.zeros(24)
    for i in range(24):
        chart_hour = (i + 7) % 24  # Convert chart index to actual hour
        solar_generation_7am[i] = solar_generation_full.iloc[chart_hour]
    
    return solar_generation_7am, max_capacity_mw

def load_and_process_data():
    """Load and process the power plant data"""
    # Find and read the CSV file
    data_file = get_data_file(PLANT_DIR)
    df = pd.read_csv(data_file)
    
    # Convert datetime column to datetime type
    df['operating_datetime_utc'] = pd.to_datetime(df['operating_datetime_utc'])
    
    # Group by datetime and emissions unit, summing the gross load
    # This handles the multiple units per hour
    hourly_by_unit = df.groupby(['operating_datetime_utc', 'emissions_unit_id_epa'])['gross_load_mw'].sum().reset_index()
    
    # Pivot to get units as columns
    pivot_data = hourly_by_unit.pivot(index='operating_datetime_utc', 
                                      columns='emissions_unit_id_epa', 
                                      values='gross_load_mw').fillna(0)
    
    # Also calculate total generation across all units
    df_total = df.groupby('operating_datetime_utc')['gross_load_mw'].sum().reset_index()
    df_total.columns = ['operating_datetime_utc', 'total_mw']
    
    return pivot_data, df_total

# TODO we can probably just use one plotboddy across the entier thing, and pass it into each of these methods as we need it, make it optional also
def create_daily_peaks_chart(pivot_data, total_data, plot_buddy):
    """Create line chart showing daily peak generation throughout the year"""
    buddy = plot_buddy
    
    # Calculate daily peak generation
    daily_peaks = total_data.set_index('operating_datetime_utc').resample('D')['total_mw'].max()
    
    # Use the gs_wide style for wider charts
    with buddy.get_style_context('gs_wide'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Plot daily peaks as a clean line
        ax.plot(daily_peaks.index, daily_peaks.values,
                color='#1f77b4',  # Nice blue color
                linewidth=2.5,
                marker='.',
                markersize=4,
                alpha=0.8,
                label='Daily Peak Generation')
        
        # Removed fill_between to make it a clean line chart
        
        # Set labels
        ax.set_xlabel('Date', fontsize=buddy.standard_font_size)
        ax.set_ylabel(Y_AXIS_LABEL, fontsize=buddy.standard_font_size)
        
        # Format x-axis for better date display
        from matplotlib.dates import DateFormatter, MonthLocator
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add annotations for key insights
        max_peak = daily_peaks.max()
        max_peak_date = daily_peaks.idxmax()
        min_peak = daily_peaks.min()
        min_peak_date = daily_peaks.idxmin()
        avg_peak = daily_peaks.mean()
        
        # Add summary statistics box
        summary_text = f'Max Daily Peak: {max_peak:.0f} MW\nMin Daily Peak: {min_peak:.0f} MW\nAvg Daily Peak: {avg_peak:.0f} MW'
        
        ax.text(0.02, 0.98, summary_text,
               transform=ax.transAxes,
               fontsize=11,
               fontweight='bold',
               ha='left',
               va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Finish chart with unified function
        peaks_title, peaks_subtitle_template = CHART_TITLES_SUBTITLES['daily_peaks']
        peaks_subtitle = peaks_subtitle_template.format(
            plant_name=CONFIG['plant']['name']
        )
        finish_chart(fig, ax, buddy, 
                    title=peaks_title, 
                    subtitle=peaks_subtitle)
        
        return fig, ax

def create_daily_cycle_chart(pivot_data, total_data, target_date, plot_buddy):
    """Create line chart showing a single day's generation cycle (7 AM to 7 AM)
    
    Args:
        target_date: datetime.date object for the analysis date
        plot_buddy: PlotBuddy instance
    """
    buddy = plot_buddy
    
    # Create datetime range from 7 AM on target date to 7 AM next day
    from datetime import datetime, timedelta
    start_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=7)
    end_datetime = start_datetime + timedelta(hours=24)
    
    # Filter data for the 24-hour period from 7 AM to 7 AM
    day_mask = (total_data['operating_datetime_utc'] >= start_datetime) & \
               (total_data['operating_datetime_utc'] < end_datetime)
    day_total_data = total_data[day_mask].copy()
    
    # Filter pivot data for the same period
    day_pivot_mask = (pivot_data.index >= start_datetime) & (pivot_data.index < end_datetime)
    day_pivot_data = pivot_data[day_pivot_mask].copy()
    
    # Use the gs_wide style
    with buddy.get_style_context('gs_wide'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Create custom x-axis labels for 7 AM to 7 AM display
        # Create hour labels that wrap around from 7 AM
        hour_labels = []
        for i in range(24):
            hour = (i + 7) % 24
            if hour == 0:
                hour_labels.append('12 AM')
            elif hour < 12:
                hour_labels.append(f'{hour} AM')
            elif hour == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{hour - 12} PM')
        
        # Plot total generation with x-axis as sequential hours (0-23)
        x_values = list(range(len(day_total_data)))
        ax.plot(x_values, day_total_data['total_mw'],
                label='Total Generation',
                color='#1f77b4',  # Nice blue color
                linewidth=3,
                marker='o',
                markersize=8,
                alpha=0.9)
        
        # Add fill under the curve for visual effect
        ax.fill_between(x_values, 0, day_total_data['total_mw'], 
                       alpha=0.2, color='#1f77b4')
        
        # Set labels
        ax.set_xlabel('Hour of Day', fontsize=buddy.standard_font_size)
        ax.set_ylabel(Y_AXIS_LABEL, fontsize=buddy.standard_font_size)
        
        # Set x-axis with custom labels
        ax.set_xticks(range(0, 24, 2))  # Show every 2 hours
        ax.set_xticklabels([hour_labels[i] for i in range(0, 24, 2)], rotation=45, ha='right')
        ax.set_xlim(-0.5, 23.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add titles with the specific date range
        next_date = target_date + timedelta(days=1)
        daily_title = f'Daily Power Generation Cycle'
        daily_subtitle = f"{CONFIG['plant']['name']} - {target_date.strftime('%B %d, %Y')} 7 AM to {next_date.strftime('%B %d, %Y')} 7 AM"
        buddy.add_titles(ax, daily_title, daily_subtitle)
        
        # Add annotations for key points
        if len(day_total_data) > 0:
            # Find peak hour
            peak_idx = day_total_data['total_mw'].idxmax()
            peak_x_position = list(day_total_data.index).index(peak_idx)
            peak_value = day_total_data.loc[peak_idx, 'total_mw']
            peak_hour = day_total_data.loc[peak_idx, 'operating_datetime_utc'].hour
            
            # Annotate peak
            ax.annotate(f'Peak: {peak_value:.0f} MW',
                       xy=(peak_x_position, peak_value),
                       xytext=(peak_x_position + 2, peak_value + 20),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12,
                       fontweight='bold',
                       color='red')
            
        
        # Create custom legend (filter out shaded regions)
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = [h for h, l in zip(handles, labels) if l == 'Total Generation']
        filtered_labels = [l for l in labels if l == 'Total Generation']
        ax.legend(filtered_handles, filtered_labels, loc='upper left', fontsize=buddy.standard_font_size)
        
        # Finish chart with unified function (no title/subtitle - handled by main loop)
        finish_chart(fig, ax, buddy)
        
        return fig, ax

def create_slack_capacity_chart(pivot_data, total_data, target_date, plot_buddy):
    """Create line chart showing available slack power generation capacity (inverse of generation)
    
    Args:
        target_date: datetime.date object for the analysis date
        plot_buddy: PlotBuddy instance
    """
    buddy = plot_buddy
    
    # Create datetime range from 7 AM on target date to 7 AM next day
    from datetime import datetime, timedelta
    start_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=7)
    end_datetime = start_datetime + timedelta(hours=24)
    
    # Filter data for the 24-hour period from 7 AM to 7 AM
    day_mask = (total_data['operating_datetime_utc'] >= start_datetime) & \
               (total_data['operating_datetime_utc'] < end_datetime)
    day_total_data = total_data[day_mask].copy()
    
    # Calculate slack capacity as inverse of generation
    # Use the peak value for this specific day as the maximum capacity
    peak_capacity = day_total_data['total_mw'].max()
    day_total_data['slack_mw'] = peak_capacity - day_total_data['total_mw']
    
    # Use the gs_wide style
    with buddy.get_style_context('gs_wide'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Create custom x-axis labels for 7 AM to 7 AM display
        hour_labels = []
        for i in range(24):
            hour = (i + 7) % 24
            if hour == 0:
                hour_labels.append('12 AM')
            elif hour < 12:
                hour_labels.append(f'{hour} AM')
            elif hour == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{hour - 12} PM')
        
        # Plot slack capacity with x-axis as sequential hours (0-23)
        x_values = list(range(len(day_total_data)))
        ax.plot(x_values, day_total_data['slack_mw'],
                label='Available Slack Capacity',
                color='#2ca02c',  # Green color for available capacity
                linewidth=3,
                marker='o',
                markersize=8,
                alpha=0.9)
        
        # Add fill under the curve for visual effect
        ax.fill_between(x_values, 0, day_total_data['slack_mw'], 
                       alpha=0.2, color='#2ca02c')
        
        # Set labels
        ax.set_xlabel('Hour of Day', fontsize=buddy.standard_font_size)
        ax.set_ylabel('Available Capacity (MW)', fontsize=buddy.standard_font_size)
        
        # Set x-axis with custom labels
        ax.set_xticks(range(0, 24, 2))  # Show every 2 hours
        ax.set_xticklabels([hour_labels[i] for i in range(0, 24, 2)], rotation=45, ha='right')
        ax.set_xlim(-0.5, 23.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add titles with the specific date range
        next_date = target_date + timedelta(days=1)
        slack_title = f'Available Slack Power Generation Capacity'
        slack_subtitle = f"{CONFIG['plant']['name']} - {target_date.strftime('%B %d, %Y')} 7 AM to {next_date.strftime('%B %d, %Y')} 7 AM"
        buddy.add_titles(ax, slack_title, slack_subtitle)
        
        # Add annotations for key points
        if len(day_total_data) > 0:
            # Find minimum slack (which corresponds to peak generation)
            min_slack_idx = day_total_data['slack_mw'].idxmin()
            min_slack_x_position = list(day_total_data.index).index(min_slack_idx)
            min_slack_value = day_total_data.loc[min_slack_idx, 'slack_mw']
            
            # Find maximum slack (which corresponds to minimum generation)
            max_slack_idx = day_total_data['slack_mw'].idxmax()
            max_slack_x_position = list(day_total_data.index).index(max_slack_idx)
            max_slack_value = day_total_data.loc[max_slack_idx, 'slack_mw']
            
            # Annotate minimum slack
            if min_slack_value < 10:  # Only annotate if very low
                ax.annotate(f'Min: {min_slack_value:.0f} MW',
                           xy=(min_slack_x_position, min_slack_value),
                           xytext=(min_slack_x_position - 3, min_slack_value + 30),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=12,
                           fontweight='bold',
                           color='red')
            
            # Annotate maximum slack
            ax.annotate(f'Max: {max_slack_value:.0f} MW',
                       xy=(max_slack_x_position, max_slack_value),
                       xytext=(max_slack_x_position + 2, max_slack_value - 30),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=12,
                       fontweight='bold',
                       color='green')
            
            # Add note about peak capacity
            ax.text(0.98, 0.98, f'Peak Capacity: {peak_capacity:.0f} MW',
                   transform=ax.transAxes,
                   fontsize=12,
                   fontweight='bold',
                   ha='right',
                   va='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Create legend
        ax.legend(loc='upper left', fontsize=buddy.standard_font_size)
        
        # Finish chart with unified function
        finish_chart(fig, ax, buddy)
        
        return fig, ax

def create_solar_potential_chart(pivot_data, total_data, target_date, plot_buddy):
    """Create line chart showing solar generation potential using dummy data
    
    Args:
        target_date: datetime.date object for the analysis date
        plot_buddy: PlotBuddy instance
    """
    buddy = plot_buddy
    
    # Create datetime range from 7 AM on target date to 7 AM next day
    from datetime import datetime, timedelta
    import numpy as np
    start_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=7)
    end_datetime = start_datetime + timedelta(hours=24)
    
    # Generate realistic solar data using pvlib for accurate location and date
    solar_generation, max_solar_capacity = generate_realistic_solar_data(
        target_date=target_date,
        latitude=CONFIG['plant']['latitude'],
        longitude=CONFIG['plant']['longitude'],
        max_capacity_mw=CONFIG['solar']['max_capacity_mw']
    )
    
    # Use the gs_wide style
    with buddy.get_style_context('gs_wide'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Create custom x-axis labels for 7 AM to 7 AM display
        hour_labels = []
        for i in range(24):
            hour = (i + 7) % 24
            if hour == 0:
                hour_labels.append('12 AM')
            elif hour < 12:
                hour_labels.append(f'{hour} AM')
            elif hour == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{hour - 12} PM')
        
        # Plot solar generation potential
        x_values = list(range(24))
        ax.plot(x_values, solar_generation,
                label='Solar Generation Potential',
                color='#FFD700',  # Gold/yellow color for solar
                linewidth=3,
                marker='o',
                markersize=8,
                alpha=0.9)
        
        # Add fill under the curve for visual effect
        ax.fill_between(x_values, 0, solar_generation, 
                       alpha=0.3, color='#FFD700')
        
        # Set labels
        ax.set_xlabel('Hour of Day', fontsize=buddy.standard_font_size)
        ax.set_ylabel('Solar Generation Potential (MW)', fontsize=buddy.standard_font_size)
        
        # Set x-axis with custom labels
        ax.set_xticks(range(0, 24, 2))  # Show every 2 hours
        ax.set_xticklabels([hour_labels[i] for i in range(0, 24, 2)], rotation=45, ha='right')
        ax.set_xlim(-0.5, 23.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add titles with the specific date range
        next_date = target_date + timedelta(days=1)
        solar_title = f'Solar Generation Potential'
        solar_subtitle = f"{CONFIG['plant']['name']} - {target_date.strftime('%B %d, %Y')} 7 AM to {next_date.strftime('%B %d, %Y')} 7 AM (Dummy Data)"
        buddy.add_titles(ax, solar_title, solar_subtitle)
        
        # Add annotations for key points
        peak_solar_idx = np.argmax(solar_generation)
        peak_solar_value = solar_generation[peak_solar_idx]
        
        # Annotate peak solar
        ax.annotate(f'Peak: {peak_solar_value:.0f} MW',
                   xy=(peak_solar_idx, peak_solar_value),
                   xytext=(peak_solar_idx + 2, peak_solar_value + 15),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                   fontsize=12,
                   fontweight='bold',
                   color='orange')
        
        # Add note about solar capacity
        ax.text(0.98, 0.98, f'Max Solar Capacity: {max_solar_capacity:.0f} MW',
               transform=ax.transAxes,
               fontsize=12,
               fontweight='bold',
               ha='right',
               va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # Add sunrise/sunset indicators using location-specific times
        # Convert actual hours to our 7AM-7AM scale
        sunrise_hour = CONFIG['solar_timing']['sunrise_hour']
        sunset_hour = CONFIG['solar_timing']['sunset_hour']
        sunrise_index = sunrise_hour - 7
        if sunrise_index < 0:
            sunrise_index += 24
        sunset_index = sunset_hour - 7
        
        ax.axvline(x=sunrise_index, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=sunset_index, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        
        # Format sunrise/sunset times
        sunrise_hr = int(sunrise_hour)
        sunrise_min = int((sunrise_hour - sunrise_hr) * 60)
        sunset_hr = int((sunset_hour - sunset_hr) * 60)
        sunset_min = int((sunset_hour - sunset_hr) * 60)
        sunrise_str = f'{sunrise_hr}:{sunrise_min:02d} AM' if sunrise_hr < 12 else f'{sunrise_hr-12}:{sunrise_min:02d} PM'
        sunset_str = f'{sunset_hr-12}:{sunset_min:02d} PM' if sunset_hr > 12 else f'{sunset_hr}:{sunset_min:02d} AM'
        
        ax.text(sunrise_index + 0.5, max_solar_capacity * 0.9, f'Sunrise\n({sunrise_str})', 
               fontsize=10, ha='left', va='top', color='orange', fontweight='bold')
        ax.text(sunset_index - 0.5, max_solar_capacity * 0.9, f'Sunset\n({sunset_str})', 
               fontsize=10, ha='right', va='top', color='orange', fontweight='bold')
        
        # Create legend
        ax.legend(loc='upper left', fontsize=buddy.standard_font_size)
        
        # Finish chart with unified function
        solar_source = f"{CONFIG['output']['source_citation']} + Solar timing: {CONFIG['plant']['location']}"
        finish_chart(fig, ax, buddy, source_citation=solar_source)
        
        return fig, ax

def create_investment_scenario_chart(day_data, investment_type, plot_buddy):
    """Create a chart showing the impact of a fixed investment on capacity.

    Args:
        day_data (pd.DataFrame): The 24-hour data for the target day.
        investment_type (str): The type of investment ('solar', 'battery', or 'gas').
        plot_buddy (PlotBuddy): The PlotBuddy instance.

    Returns:
        tuple: (fig, ax)
    """
    INVESTMENT_BUDGET = 200_000_000  # $200M

    # Calculate slack capacity
    peak_capacity = day_data['total_mw'].max()
    slack_capacity = peak_capacity - day_data['total_mw']

    # Calculate the capacity purchased with the investment
    if investment_type == 'solar':
        purchased_capacity_mw = INVESTMENT_BUDGET / SOLAR_COST_PER_MW
        additional_capacity, _ = generate_realistic_solar_data(
            target_date=day_data['operating_datetime_utc'].iloc[0].date(),
            latitude=CONFIG['plant']['latitude'],
            longitude=CONFIG['plant']['longitude'],
            max_capacity_mw=purchased_capacity_mw
        )
        layer_label = f'Solar Capacity (${INVESTMENT_BUDGET/1_000_000:.0f}M Investment)'
        layer_color = '#FFD700'
    elif investment_type == 'battery':
        purchased_capacity_mwh = INVESTMENT_BUDGET / BATTERY_COST_PER_MWH
        # Simulate battery discharge to meet the gap
        additional_capacity = np.zeros(24)
        battery_charge_level_mwh = purchased_capacity_mwh
        for i in range(24):
            gap_to_target = max(0, CONFIG['datacenter']['target_power_mw'] - (slack_capacity.values[i]))
            if gap_to_target > 0 and battery_charge_level_mwh > 0:
                discharge_mw = min(gap_to_target, battery_charge_level_mwh, CONFIG['datacenter']['target_power_mw'] * BATTERY_DISCHARGE_RATE_RATIO)
                additional_capacity[i] = discharge_mw
                battery_charge_level_mwh -= discharge_mw
        layer_label = f'Battery Capacity (${INVESTMENT_BUDGET/1_000_000:.0f}M Investment)'
        layer_color = '#FF69B4'
    elif investment_type == 'gas':
        purchased_capacity_mw = INVESTMENT_BUDGET / NATGAS_GENERATOR_COST_PER_MW
        additional_capacity = np.zeros(24)
        for i in range(24):
            gap_to_target = max(0, CONFIG['datacenter']['target_power_mw'] - (slack_capacity.values[i]))
            if gap_to_target > 0:
                additional_capacity[i] = min(gap_to_target, purchased_capacity_mw)
        layer_label = f'Backup Gas Capacity (${INVESTMENT_BUDGET/1_000_000:.0f}M Investment)'
        layer_color = '#FF4500'

    layers = [
        {'data': slack_capacity.values, 'label': 'Available Plant Slack', 'color': '#2ca02c'},
        {'data': additional_capacity, 'label': layer_label, 'color': layer_color}
    ]

    title = f'Impact of ${INVESTMENT_BUDGET/1_000_000:.0f}M Investment in {investment_type.capitalize()} Capacity'
    subtitle = f"{CONFIG['plant']['name']} - {day_data['operating_datetime_utc'].iloc[0].strftime('%B %d, %Y')}"

    return create_capacity_stack_chart(
        layers=layers,
        target_mw=CONFIG['datacenter']['target_power_mw'],
        title=title,
        subtitle=subtitle,
        plot_buddy=plot_buddy
    )

def create_battery_storage_chart(pivot_data, total_data, target_date, plot_buddy):
    """Create stacked area chart with battery storage charging/discharging cycle
    
    Args:
        target_date: datetime.date object for the analysis date
        plot_buddy: PlotBuddy instance
    """
    buddy = plot_buddy
    
    # Create datetime range from 7 AM on target date to 7 AM next day
    from datetime import datetime, timedelta
    import numpy as np
    start_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=7)
    end_datetime = start_datetime + timedelta(hours=24)
    
    # Filter data for the 24-hour period from 7 AM to 7 AM
    day_mask = (total_data['operating_datetime_utc'] >= start_datetime) & \
               (total_data['operating_datetime_utc'] < end_datetime)
    day_total_data = total_data[day_mask].copy()
    
    # Calculate slack capacity as inverse of generation (reuse logic)
    peak_capacity = day_total_data['total_mw'].max()
    slack_capacity = peak_capacity - day_total_data['total_mw']
    
    # Generate realistic solar data using pvlib for accurate location and date
    solar_generation, max_solar_capacity = generate_realistic_solar_data(
        target_date=target_date,
        latitude=CONFIG['plant']['latitude'],
        longitude=CONFIG['plant']['longitude'],
        max_capacity_mw=CONFIG['solar']['max_capacity_mw']
    )
    
    # Battery storage simulation using CONFIG values
    battery_storage = np.zeros(24)  # Available battery capacity for discharge (MW)
    battery_charge_level_mwh = 0  # Track battery state of charge in MWh
    
    # Calculate battery configuration dynamically based on datacenter target
    target_total_capacity_mw = CONFIG['datacenter']['target_power_mw']
    max_battery_capacity_mwh = target_total_capacity_mw * BATTERY_HOURS_BACKUP
    max_charge_rate_mw = target_total_capacity_mw * BATTERY_CHARGE_RATE_RATIO
    max_discharge_rate_mw = target_total_capacity_mw * BATTERY_DISCHARGE_RATE_RATIO
    battery_degradation_mw_per_hour = BATTERY_DEGRADATION_MW_PER_HOUR
    
    # Calculate how much battery we need each hour to reach target
    for i in range(24):
        current_capacity = slack_capacity.values[i] + solar_generation[i]
        gap_to_target = max(0, target_total_capacity_mw - current_capacity)
        
        # Charge battery when there's excess capacity (above target)
        if current_capacity > target_total_capacity_mw and battery_charge_level_mwh < max_battery_capacity_mwh:
            excess_capacity = current_capacity - target_total_capacity_mw
            charge_rate_mw = min(max_charge_rate_mw, excess_capacity)
            charge_energy_mwh = charge_rate_mw  # 1 hour * MW = MWh
            battery_charge_level_mwh = min(max_battery_capacity_mwh, battery_charge_level_mwh + charge_energy_mwh)
        
        # Discharge battery when capacity is below target
        elif gap_to_target > 0 and battery_charge_level_mwh > 0:
            # Discharge exactly what's needed to reach target (up to available battery and discharge rate)
            max_discharge_mw = min(gap_to_target, max_discharge_rate_mw, battery_charge_level_mwh)  # Can't discharge more MWh than we have
            discharge_energy_mwh = max_discharge_mw  # 1 hour * MW = MWh
            battery_charge_level_mwh = max(0, battery_charge_level_mwh - discharge_energy_mwh)
            battery_storage[i] = max_discharge_mw
        
        # Natural battery degradation
        degradation_mwh = battery_degradation_mw_per_hour  # 1 hour * MW = MWh
        battery_charge_level_mwh = max(0, battery_charge_level_mwh - degradation_mwh)
    
    return battery_storage

def create_backup_generators_chart(pivot_data, total_data, target_date, plot_buddy):
    """Create stacked area chart with battery storage + backup natural gas generators
    
    Args:
        target_date: datetime.date object for the analysis date
        plot_buddy: PlotBuddy instance
    """
    buddy = plot_buddy
    
    # Create datetime range from 7 AM on target date to 7 AM next day
    from datetime import datetime, timedelta
    import numpy as np
    start_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=7)
    end_datetime = start_datetime + timedelta(hours=24)
    
    # Filter data for the 24-hour period from 7 AM to 7 AM
    day_mask = (total_data['operating_datetime_utc'] >= start_datetime) & \
               (total_data['operating_datetime_utc'] < end_datetime)
    day_total_data = total_data[day_mask].copy()
    
    # Calculate slack capacity as inverse of generation (reuse logic)
    peak_capacity = day_total_data['total_mw'].max()
    slack_capacity = peak_capacity - day_total_data['total_mw']
    
    # Generate realistic solar data using pvlib for accurate location and date
    solar_generation, max_solar_capacity = generate_realistic_solar_data(
        target_date=target_date,
        latitude=CONFIG['plant']['latitude'],
        longitude=CONFIG['plant']['longitude'],
        max_capacity_mw=CONFIG['solar']['max_capacity_mw']
    )
    
    # Battery storage simulation using CONFIG values (reuse logic from chart 6)
    battery_storage = np.zeros(24)  # Available battery capacity for discharge (MW)
    battery_charge_level_mwh = 0  # Track battery state of charge in MWh
    
    # Calculate battery configuration dynamically based on datacenter target
    target_power = CONFIG['datacenter']['target_power_mw']
    max_battery_capacity_mwh = target_power * BATTERY_HOURS_BACKUP
    max_charge_rate_mw = target_power * BATTERY_CHARGE_RATE_RATIO
    max_discharge_rate_mw = target_power * BATTERY_DISCHARGE_RATE_RATIO
    battery_degradation_mw_per_hour = BATTERY_DEGRADATION_MW_PER_HOUR
    
    # Calculate battery charge/discharge (same logic as chart 6)
    for i in range(24):
        current_capacity = slack_capacity.values[i] + solar_generation[i]
        gap_to_target = max(0, target_power - current_capacity)
        
        # Charge battery when there's excess capacity (above target)
        if current_capacity > target_power and battery_charge_level_mwh < max_battery_capacity_mwh:
            excess_capacity = current_capacity - target_power
            charge_rate_mw = min(max_charge_rate_mw, excess_capacity)
            charge_energy_mwh = charge_rate_mw  # 1 hour * MW = MWh
            battery_charge_level_mwh = min(max_battery_capacity_mwh, battery_charge_level_mwh + charge_energy_mwh)
        
        # Discharge battery when capacity is below target
        elif gap_to_target > 0 and battery_charge_level_mwh > 0:
            max_discharge_mw = min(gap_to_target, max_discharge_rate_mw, battery_charge_level_mwh)
            discharge_energy_mwh = max_discharge_mw  # 1 hour * MW = MWh
            battery_charge_level_mwh = max(0, battery_charge_level_mwh - discharge_energy_mwh)
            battery_storage[i] = max_discharge_mw
        
        # Natural battery degradation
        degradation_mwh = battery_degradation_mw_per_hour  # 1 hour * MW = MWh
        battery_charge_level_mwh = max(0, battery_charge_level_mwh - degradation_mwh)
    
    # Backup natural gas generators - activate when still below target after battery
    natgas_generation = np.zeros(24)
    backup_gas_units_count = CONFIG['backup_generators']['units_count']
    backup_gas_total_capacity = GAS_BACKUP_UNIT_CAPACITY_MW * backup_gas_units_count
    
    for i in range(24):
        current_total = slack_capacity.values[i] + solar_generation[i] + battery_storage[i]
        remaining_gap = max(0, target_power - current_total)
        
        # Deploy natural gas generators to fill remaining gap (up to their total capacity)
        if remaining_gap > 0:
            natgas_generation[i] = min(remaining_gap, backup_gas_total_capacity)
    
    # Use the gs_wide style
    with buddy.get_style_context('gs_wide'):
        fig, ax = buddy.setup_figure(figsize=buddy.wide_figure)
        
        # Create custom x-axis labels for 7 AM to 7 AM display (reuse logic)
        hour_labels = []
        for i in range(24):
            hour = (i + 7) % 24
            if hour == 0:
                hour_labels.append('12 AM')
            elif hour < 12:
                hour_labels.append(f'{hour} AM')
            elif hour == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{hour - 12} PM')
        
        # Create x-axis values
        x_values = list(range(24))
        
        # Create stacked area chart with four layers
        # Bottom layer: Slack capacity (existing plant capacity not being used)
        # Second layer: Solar potential (additional renewable capacity)
        # Third layer: Battery discharge (stored energy being released)
        # Top layer: Natural gas generators (fast-response backup)
        
        ax.stackplot(x_values, 
                     slack_capacity.values, 
                     solar_generation,
                     battery_storage,
                     natgas_generation,
                     labels=['Available Plant Slack Capacity', 'Solar Generation Potential', 'Battery Discharge', 'Backup Gas Generators'],
                     colors=['#2ca02c', '#FFD700', '#FF69B4', '#FF4500'],  # Green, yellow, pink, orange-red
                     alpha=0.7)
        
        # Add outline for total combined capacity including all backup systems
        total_combined = slack_capacity.values + solar_generation + battery_storage + natgas_generation
        ax.plot(x_values, total_combined,
                color='black',
                linewidth=2,
                linestyle='-',
                alpha=0.8,
                label='Total Available Capacity')
        
        # Set labels
        ax.set_xlabel('Hour of Day', fontsize=buddy.standard_font_size)
        ax.set_ylabel('Available Capacity (MW)', fontsize=buddy.standard_font_size)
        
        # Set x-axis with custom labels
        ax.set_xticks(range(0, 24, 2))  # Show every 2 hours
        ax.set_xticklabels([hour_labels[i] for i in range(0, 24, 2)], rotation=45, ha='right')
        ax.set_xlim(-0.5, 23.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add titles with the specific date range
        next_date = target_date + timedelta(days=1)
        backup_title = f'Complete Backup Generation System - Peak Demand Day'
        backup_subtitle = f'{plant_name} - {target_date.strftime("%B %d, %Y")} 7 AM to {next_date.strftime("%B %d, %Y")} 7 AM (Battery + Natural Gas Backup)'
        buddy.add_titles(ax, backup_title, backup_subtitle)
        
        # Add target capacity line
        ax.axhline(y=target_power, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Target: {target_power} MW')
        
        # Add annotations for key insights
        max_combined_idx = np.argmax(total_combined)
        max_combined_value = total_combined[max_combined_idx]
        
        # Find peak natural gas usage
        max_natgas_idx = np.argmax(natgas_generation)
        max_natgas_value = natgas_generation[max_natgas_idx]
        
        # Calculate total daily usage
        total_battery_discharge_mwh = np.sum(battery_storage)
        total_natgas_usage_mwh = np.sum(natgas_generation)
        
        # Find worst deterioration point (minimum total capacity)
        min_combined_idx = np.argmin(total_combined)
        min_combined_value = total_combined[min_combined_idx]
        min_capacity_percentage = (min_combined_value / target_power) * 100
        
        # Annotate peak combined capacity
        ax.annotate(f'Peak Total: {max_combined_value:.0f} MW',
                   xy=(max_combined_idx, max_combined_value),
                   xytext=(max_combined_idx + 2, max_combined_value + 30),
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                   fontsize=12,
                   fontweight='bold',
                   color='darkgreen')
        
        # Annotate peak natural gas usage (if significant)
        if max_natgas_value > 5:
            ax.annotate(f'Max Natural Gas: {max_natgas_value:.0f} MW\n({backup_gas_units_count} units)',
                       xy=(max_natgas_idx, total_combined[max_natgas_idx]),
                       xytext=(max_natgas_idx - 4, total_combined[max_natgas_idx] + 50),
                       arrowprops=dict(arrowstyle='->', color='darkorange', lw=2),
                       fontsize=12,
                       fontweight='bold',
                       color='darkorange')
        
        # Annotate worst case (if still below target)
        if min_combined_value < target_power:
            ax.annotate(f'Worst Case: {min_combined_value:.0f} MW\n({min_capacity_percentage:.1f}% of target)',
                       xy=(min_combined_idx, min_combined_value),
                       xytext=(min_combined_idx + 3, min_combined_value - 50),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12,
                       fontweight='bold',
                       color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add comprehensive summary statistics box
        summary_text = f'Plant: {peak_capacity:.0f} MW\nSolar: {max_solar_capacity:.0f} MW\nBattery: {max_battery_capacity_mwh:.0f} MWh\nBackup Gas: {backup_gas_total_capacity:.0f} MW ({backup_gas_units_count} units)\nDaily Battery: {total_battery_discharge_mwh:.0f} MWh\nDaily Natural Gas: {total_natgas_usage_mwh:.0f} MWh\nTarget: {target_power:.0f} MW\nWorst Case: {min_combined_value:.0f} MW ({min_capacity_percentage:.1f}%)'
        
        ax.text(0.02, 0.98, summary_text,
               transform=ax.transAxes,
               fontsize=10,
               fontweight='bold',
               ha='left',
               va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Finish chart with unified function (legend handled by create_capacity_stack_chart)
        backup_source = f"{CONFIG['output']['source_citation']} + Solar: {CONFIG['plant']['location']} + Battery + Backup Gas"
        finish_chart(fig, ax, buddy, source_citation=backup_source)
        
        return fig, ax

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate power plant analysis charts from CSV data and YAML CONFIG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python power_plant_hourly_analysis.py plant_55286_Oleander/
  python power_plant_hourly_analysis.py plant_8042_Belews_Creek/
  '''
    )
    parser.add_argument(
        'plant_directory',
        help='Directory containing plant data (CSV file and config.yaml)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Skip displaying charts (useful for batch processing)'
    )
    return parser.parse_args()

def main():
    """Main function to generate the power plant analysis"""
    global CONFIG, PLANT_DIR, PLOT_BUDDY
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up global variables
    PLANT_DIR = args.plant_directory
    
    # Calculate dynamic configuration (eliminates need for config.yaml)
    print(f"Calculating dynamic configuration for {PLANT_DIR}...")
    CONFIG = calculate_plant_configuration(PLANT_DIR)
    
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
        'investment_solar': f'5a_{chart_prefix}_investment_solar.png',
        'investment_battery': f'5b_{chart_prefix}_investment_battery.png',
        'investment_gas': f'5c_{chart_prefix}_investment_gas.png',
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
    
    # Calculate target_mw based on peak slack capacity of the target day
    max_slack_value = slack_capacity.max()
    target_mw = max(100, round((max_slack_value * DATACENTER_TARGET_PERCENT_MAX) / 50) * 50)
    CONFIG['datacenter'] = {'target_power_mw': target_mw}
    
    # Recalculate solar and backup generator capacities based on the new target_mw
    solar_capacity_mw = int(target_mw * SOLAR_CAPACITY_PERCENT_TARGET)
    backup_gas_units_count = max(10, int((target_mw * BACKUP_GENERATOR_CAPACITY_PERCENT_TARGET) / GAS_BACKUP_UNIT_CAPACITY_MW))
    backup_gas_total_capacity = GAS_BACKUP_UNIT_CAPACITY_MW * backup_gas_units_count
    
    print(f"Generating solar and battery storage data...")
    solar_generation, max_solar_capacity = generate_realistic_solar_data(
        target_date=target_date,
        latitude=CONFIG['plant']['latitude'],
        longitude=CONFIG['plant']['longitude'], 
        max_capacity_mw=solar_capacity_mw
    )
    battery_storage = simulate_battery_storage(slack_capacity, solar_generation)
    
    # Natural gas generation calculation  
    import numpy as np
    natgas_generation = np.zeros(24)
    
    for i in range(24):
        current_total = slack_capacity.values[i] + solar_generation[i] + battery_storage[i]
        remaining_gap = max(0, target_mw - current_total)
        if remaining_gap > 0:
            natgas_generation[i] = min(remaining_gap, backup_gas_total_capacity)
    
    # Calculate costs for enhanced legend labels (all dynamic now)
    battery_capacity_mwh = target_mw * BATTERY_HOURS_BACKUP
    natgas_capacity_mw = backup_gas_total_capacity
    
    solar_cost = (solar_capacity_mw * SOLAR_COST_PER_MW) / 1_000_000
    battery_cost = (battery_capacity_mwh * BATTERY_COST_PER_MWH) / 1_000_000
    natgas_cost = (natgas_capacity_mw * NATGAS_GENERATOR_COST_PER_MW) / 1_000_000
    
    # Define layer configurations for each chart type with enhanced labels
    # Common variables for subtitle formatting
    date_range = target_date.strftime('%B %d, %Y')
    plant_name = CONFIG['plant']['name']
    
    chart_configs = [
        # Chart 2: Daily cycle (plant generation only)
        {
            'name': 'daily_cycle',
            'layers': [{'data': day_data['total_mw'].values, 'label': 'Total Generation', 'color': '#1f77b4'}],
            'title': CHART_TITLES_SUBTITLES['daily_cycle'][0].format(plant_name=plant_name),
            'subtitle': CHART_TITLES_SUBTITLES['daily_cycle'][1].format(plant_name=plant_name, date_range=date_range),
            'show_target_analysis': False
        },
        # Chart 3: Slack capacity 
        {
            'name': 'slack_capacity',
            'layers': [{'data': slack_capacity.values, 'label': 'Available Slack Capacity', 'color': '#2ca02c'}],
            'title': CHART_TITLES_SUBTITLES['slack_capacity'][0],
            'subtitle': CHART_TITLES_SUBTITLES['slack_capacity'][1].format(plant_name=plant_name, date_range=date_range),
            'show_target_analysis': False
        },
        # Chart 4: Solar potential
        {
            'name': 'solar_potential', 
            'layers': [{'data': solar_generation, 'label': LEGEND_LABELS['solar'].format(capacity=solar_capacity_mw, cost=solar_cost), 'color': '#FFD700'}],
            'title': CHART_TITLES_SUBTITLES['solar_potential'][0],
            'subtitle': CHART_TITLES_SUBTITLES['solar_potential'][1].format(plant_name=plant_name, date_range=date_range),
            'show_target_analysis': False
        },
        # Chart 5a: Solar Investment
        {
            'name': 'investment_solar',
            'investment_type': 'solar'
        },
        # Chart 5b: Battery Investment
        {
            'name': 'investment_battery',
            'investment_type': 'battery'
        },
        # Chart 5c: Gas Investment
        {
            'name': 'investment_gas',
            'investment_type': 'gas'
        },
        # Chart 6: Battery storage (plant slack + solar + battery)
        {
            'name': 'battery_storage',
            'layers': [
                {'data': slack_capacity.values, 'label': LEGEND_LABELS['plant_slack'], 'color': '#2ca02c'},
                {'data': solar_generation, 'label': LEGEND_LABELS['solar'].format(capacity=solar_capacity_mw, cost=solar_cost), 'color': '#FFD700'},
                {'data': battery_storage, 'label': LEGEND_LABELS['battery'].format(hours=BATTERY_HOURS_BACKUP, capacity=battery_capacity_mwh, cost=battery_cost), 'color': '#FF69B4'}
            ],
            'title': CHART_TITLES_SUBTITLES['battery_storage'][0],
            'subtitle': CHART_TITLES_SUBTITLES['battery_storage'][1].format(plant_name=plant_name, date_range=date_range)
        },
        # Chart 7: Complete backup system (plant slack + solar + battery + natural gas)
        {
            'name': 'backup_generators',
            'layers': [
                {'data': slack_capacity.values, 'label': LEGEND_LABELS['plant_slack'], 'color': '#2ca02c'},
                {'data': solar_generation, 'label': LEGEND_LABELS['solar'].format(capacity=solar_capacity_mw, cost=solar_cost), 'color': '#FFD700'},
                {'data': battery_storage, 'label': LEGEND_LABELS['battery'].format(hours=BATTERY_HOURS_BACKUP, capacity=battery_capacity_mwh, cost=battery_cost), 'color': '#FF69B4'},
                {'data': natgas_generation, 'label': LEGEND_LABELS['backup_gas'].format(capacity=natgas_capacity_mw, cost=natgas_cost), 'color': '#FF4500'}
            ],
            'title': CHART_TITLES_SUBTITLES['backup_generators'][0],
            'subtitle': CHART_TITLES_SUBTITLES['backup_generators'][1].format(plant_name=plant_name, date_range=date_range)
        }
    ]
    
    # Generate all capacity charts using unified function
    for config in chart_configs:
        print(f"Creating {config['name']} chart...")
        if 'investment_type' in config:
            fig, ax = create_investment_scenario_chart(
                day_data=day_data,
                investment_type=config['investment_type'],
                plot_buddy=PLOT_BUDDY
            )
        else:
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
