# IPO Pricing Tool

A Python tool to price shares in an Initial Public Offering (IPO) based on bid and ask data from investors.

## Overview

This tool simulates the traditional bookbuilding process used by investment banks to price IPOs. It generates mock investor data, determines an optimal share price, allocates shares to investors, and visualizes the results.

Key features:
- Generate realistic mock investor data for tech IPOs
- Implement traditional bookbuilding pricing methodology
- Visualize demand curves, allocation distributions, and key metrics
- Analyze IPO performance metrics like oversubscription and "money left on table"

## Installation

### Prerequisites

- Python 3.7+
- Required packages: numpy, pandas, matplotlib

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ipo-pricing-tool
   ```

2. Install dependencies:
   ```
   pip install numpy pandas matplotlib
   ```

## Usage

### Basic Usage

Run the tool with default parameters:

```bash
python -m ipo_pricing_tool.main
```

This will:
1. Generate mock data for 100 investors
2. Run the IPO pricing process
3. Display the results and visualizations

### Command Line Options

The tool supports various command line options:

```
usage: main.py [-h] [--num-investors NUM_INVESTORS] [--min-price MIN_PRICE]
               [--max-price MAX_PRICE] [--total-shares TOTAL_SHARES]
               [--seed SEED] [--output-dir OUTPUT_DIR] [--save-data]
               [--save-plots] [--no-display]

IPO Pricing Tool - Price shares in an IPO based on investor bids

options:
  -h, --help            show this help message and exit
  --num-investors NUM_INVESTORS
                        Number of investors to generate (default: 100)
  --min-price MIN_PRICE
                        Minimum price in the IPO range (default: $20.00)
  --max-price MAX_PRICE
                        Maximum price in the IPO range (default: $25.00)
  --total-shares TOTAL_SHARES
                        Total number of shares available in the IPO (default: 10,000,000)
  --seed SEED           Random seed for reproducible results (default: None)
  --output-dir OUTPUT_DIR
                        Directory to save output files (default: None)
  --save-data           Save investor data to CSV file
  --save-plots          Save plots to output directory
  --no-display          Do not display plots (useful for batch processing)
```

### Examples

Generate an IPO with 200 investors and a higher price range:

```bash
python -m ipo_pricing_tool.main --num-investors 200 --min-price 30 --max-price 35
```

Save all data and plots to an output directory:

```bash
python -m ipo_pricing_tool.main --output-dir ipo_results --save-data --save-plots
```

Run with a specific random seed for reproducible results:

```bash
python -m ipo_pricing_tool.main --seed 42
```

## Output

The tool produces:

1. **Console Output**: Detailed IPO pricing results including:
   - Final IPO price
   - Capital raised
   - Oversubscription ratios
   - Allocation metrics
   - Pricing analysis

2. **Visualizations**:
   - Demand curve showing cumulative demand at different price points
   - Distribution of share allocations
   - Distribution of investor bids
   - Summary of key IPO metrics

3. **Optional Data Files** (when using `--save-data`):
   - `investor_data.csv`: Original investor bid data
   - `allocation_results.csv`: Final share allocations

## Project Structure

- `data_generator.py`: Generates mock investor data
- `pricing_engine.py`: Implements the IPO pricing algorithm
- `visualizer.py`: Creates visualizations of the results
- `main.py`: Command-line interface and entry point

## How It Works

1. **Data Generation**: Creates mock investor data with realistic bid prices and quantities
2. **Pricing Process**: 
   - Calculates the demand curve
   - Determines the clearing price
   - Applies traditional IPO pricing adjustments
3. **Share Allocation**: Allocates shares to eligible investors
4. **Analysis**: Calculates key metrics and visualizes the results

## License

[MIT License](LICENSE)
