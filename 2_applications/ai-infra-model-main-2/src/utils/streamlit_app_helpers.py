import inspect
from dataclasses import is_dataclass
from typing import Any, Dict, Sequence, TypeVar, cast

import pandas as pd

import streamlit as st
from src.constants.gpu_dataclass import ALL_GPU_LIST, GPUDataclass


def clear_session_state_on_first_load(initialization_key: str, key_substrings_to_clear: Sequence[str]) -> None:
    """Clear matching Streamlit session state keys once per app load to avoid stale values."""
    if initialization_key not in st.session_state:
        keys_to_remove = [
            key
            for key in list(st.session_state.keys())
            if any(substring in key for substring in key_substrings_to_clear)
        ]
        for key in keys_to_remove:
            del st.session_state[key]
        st.session_state[initialization_key] = True


_GPU_NAME_OPTIONS = [gpu.name for gpu in ALL_GPU_LIST]
_GPU_NAME_TO_GPU = {gpu.name: gpu for gpu in ALL_GPU_LIST}
_GPU_NAME_TO_INDEX = {name: idx for idx, name in enumerate(_GPU_NAME_OPTIONS)}
_QUANTIZATION_OPTIONS = ("fp16", "fp8", "int8", "int4")
_QUANTIZATION_TO_INDEX = {value: idx for idx, value in enumerate(_QUANTIZATION_OPTIONS)}


def _select_gpu_model(description: str, default: GPUDataclass) -> GPUDataclass:
    """Render a GPU model dropdown and return the selected dataclass instance."""
    if not _GPU_NAME_OPTIONS:
        raise ValueError("No GPU models configured for selection")

    default_name = default.name if isinstance(default, GPUDataclass) else None
    if default_name not in _GPU_NAME_TO_INDEX:
        default_name = _GPU_NAME_OPTIONS[0]

    selected_name = st.selectbox(
        description,
        options=_GPU_NAME_OPTIONS,
        index=_GPU_NAME_TO_INDEX[default_name],
    )

    return _GPU_NAME_TO_GPU[selected_name]


def _select_quantization_format(description: str, default: str) -> str:
    """Render a quantization format dropdown and return the selected option."""
    default_value = default if default in _QUANTIZATION_TO_INDEX else _QUANTIZATION_OPTIONS[0]
    return st.selectbox(
        description,
        options=_QUANTIZATION_OPTIONS,
        index=_QUANTIZATION_TO_INDEX[default_value],
    )


def format_number_to_string(amount, is_currency: bool = False, escape_markdown: bool = False):
    """
    Format a number to a human-readable string with appropriate suffixes (K, M, B, T, Q, etc).
    For numbers >= 1 quadrillion, includes exponential notation for clarity.

    Args:
        amount: The number to format
        is_currency: Whether to include a dollar sign prefix
        escape_markdown: Whether to escape dollar signs for Markdown (True for Streamlit text, False for Plotly charts)

    Returns:
        A formatted string representation of the number
    """
    if is_currency:
        prefix = '\\$' if escape_markdown else '$'
    else:
        prefix = ''

    # Handle negative numbers
    is_negative = amount < 0
    abs_amount = abs(amount)

    # For extremely large numbers (>= 1 quintillion), use exponential notation
    if abs_amount >= 1_000_000_000_000_000_000:  # 1 quintillion (10^18)
        exponent = len(str(int(abs_amount))) - 1
        mantissa = abs_amount / (10 ** exponent)
        formatted = f"{prefix}{mantissa:.1f}e{exponent}"
    # Quadrillion (1,000 T)
    elif abs_amount >= 1_000_000_000_000_000:  # 1 quadrillion (10^15)
        quad_value = abs_amount / 1_000_000_000_000_000
        formatted = f"{prefix}{quad_value:.1f}Q (10^15)"
    # Trillion
    elif abs_amount >= 1_000_000_000_000:  # 1 trillion (10^12)
        formatted = f"{prefix}{abs_amount/1_000_000_000_000:.1f}T"
    # Billion
    elif abs_amount >= 1_000_000_000:  # 1 billion (10^9)
        formatted = f"{prefix}{abs_amount/1_000_000_000:.1f}B"
    # Million
    elif abs_amount >= 1_000_000:  # 1 million (10^6)
        formatted = f"{prefix}{abs_amount/1_000_000:.0f}M"
    # Thousand
    elif abs_amount >= 1_000:  # 1 thousand (10^3)
        formatted = f"{prefix}{abs_amount/1_000:.1f}K"
    # If it's a currency amount less than 10 cents, show in cents
    elif is_currency and abs_amount < 0.1:
        formatted = f"{abs_amount * 100:.1f}¢"
    else:
        formatted = f"{prefix}{abs_amount:.1f}"

    # Add negative sign if needed
    return f"-{formatted}" if is_negative else formatted

# Define a generic type variable for dataclasses
T = TypeVar('T')

# Define a generic type variable for dataclasses
T = TypeVar('T')


## Sidebar Generation Helpers

def generate_sidebar_from_dataclass(dataclass_instance: T) -> T:
    """
    Dynamically generate a Streamlit sidebar configuration using an existing dataclass instance.
    
    Args:
        dataclass_instance: A dataclass instance with values to default for the sidebar
    
    Returns:
        updated_instance: A new dataclass instance with values from the sidebar
    """
    # Ensure the input is a dataclass
    if not is_dataclass(dataclass_instance):
        raise TypeError("Input must be a dataclass instance")


    # Get the class and its initialization parameters
    cls = type(dataclass_instance)
    signature = inspect.signature(cls.__init__)
    parameters = {k: v for k, v in signature.parameters.items() if k != 'self'}

    # Group parameters by category
    categorized_params = {}
    for param_name, param in parameters.items():
        # Get category and description
        category, description = _get_parameter_description(param)

        if category not in categorized_params:
            categorized_params[category] = []

        categorized_params[category].append((param_name, param, description))

    config: Dict[str, Any] = {}
    with st.sidebar:
        st.markdown(f"## {cls.__name__} Configuration")

        # Create an expander for each category
        for category, params in sorted(categorized_params.items()):
            if category is None:
                category = "Other Parameters"

            with st.expander(f"{category}", expanded=True):
                # Process enable_second_lease first if it exists in this category
                enable_second_lease_value = None
                params_to_process = []

                for param_name, param, description in sorted(params, key=lambda x: x[0]):
                    if param_name == "enable_second_lease":
                        default = getattr(dataclass_instance, param_name)
                        enable_second_lease_value = st.toggle(description, value=default)
                        config[param_name] = enable_second_lease_value
                    else:
                        params_to_process.append((param_name, param, description))

                # Process other parameters, conditionally showing second lease params
                for param_name, param, description in params_to_process:
                    # Get the default value from the dataclass instance
                    default = getattr(dataclass_instance, param_name)

                    # Hide second lease parameters if second lease is disabled
                    if param_name in ["second_lease_term_months", "second_lease_discount_multiplier"]:
                        if enable_second_lease_value is False:
                            config[param_name] = default  # Use default but don't show input
                            continue

                    # Generate appropriate input based on parameter type
                    config[param_name] = _create_sidebar_input(param_name, description, default)

    # Create the new instance with the values from the sidebar
    updated_instance = cls(**config)  # type: ignore

    # Cast to ensure type safety for the return value
    return cast(T, updated_instance)

def _get_parameter_description(param):
    """Extract category and human-readable description for a parameter."""
    from typing import Annotated, get_args, get_origin

    # Check if the annotation is an Annotated type
    if get_origin(param.annotation) is Annotated:
        args = get_args(param.annotation)
        # The second argument is the category
        category = args[1]
        # The third argument is the description
        description = args[2] if len(args) > 2 else None
        return category, description

    # Fallback to default description
    return None, param.name


def _get_parameter_default(calculator_instance, param_name):
    """Retrieve the default value for a parameter."""
    return (getattr(calculator_instance, param_name.upper(), None) or
            getattr(calculator_instance, param_name, None))

def _create_sidebar_input(param_name, description, default):
    """Create an appropriate Streamlit input based on the parameter type."""
    # Boolean input
    if isinstance(default, bool):
        return st.toggle(description, value=default)

    # Numeric input
    if isinstance(default, (int, float)):
        return _create_numeric_input(description, default)

    # String input
    if isinstance(default, str):
        if param_name == "quantization_format":
            return _select_quantization_format(description, default)
        return st.text_input(description, value=default)

    if isinstance(default, GPUDataclass):
        return _select_gpu_model(description, default)

    # Fallback for complex types
    st.write(f"Unable to generate input for {description}")
    return default

def _create_numeric_input(description, default):
    """Create a numeric input with appropriate constraints."""
    min_val = 0.0
    max_val = 1.797e+308
    step = 1.0 if isinstance(default, int) else 0.1

    # Ensure values are float and within bounds
    default = float(min(max(default, min_val), max_val))

    return st.number_input(
        description,
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step
    )

# Common formatting helpers
def format_currency(amount: float, show_cents: bool = False) -> str:
    """Format amount as currency with $ sign."""
    if show_cents and amount < 1.0:
        return f"${amount:.2f}"
    return f"${amount:,.0f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimal_places}f}%"


def calculate_chart_tick_intervals(max_value: float, base_interval: float = 5_000_000_000, target_ticks: int = 8) -> list[float]:
    """Calculate clean tick interval values for Plotly chart y-axes.

    Creates evenly spaced tick marks at multiples of base_interval to avoid cluttered
    or inconsistent axis labels on financial charts.

    Args:
        max_value: The maximum value to display on the chart
        base_interval: Base interval for tick spacing (default: 5 billion)
        target_ticks: Target number of ticks to display (default: 8)

    Returns:
        List of tick values for Plotly yaxis tickvals configuration

    Example:
        >>> calculate_chart_tick_intervals(42_000_000_000)
        [0, 5000000000, 10000000000, 15000000000, 20000000000, 25000000000, 30000000000, 35000000000, 40000000000, 45000000000]
    """
    import math

    rough_interval = max_value / target_ticks
    multiplier = math.ceil(rough_interval / base_interval)
    tick_interval = multiplier * base_interval

    max_tick = math.ceil(max_value / tick_interval) * tick_interval
    num_ticks = int(max_tick / tick_interval) + 1
    tick_values = [i * tick_interval for i in range(num_ticks)]

    return tick_values


# Common UI patterns
# Note: Explicit col1, col2, col3 = st.columns(3) is preferred over generic helpers
# Each app should control its own metric layout explicitly for clarity


def create_styled_dataframe(df: pd.DataFrame, highlight_keys: list = None, title: str = None, highlight_column: str = None) -> None:
    """Display a styled dataframe with optional highlighting, matching income statement styling."""

    # Apply styling to highlight key rows
    def highlight_rows(row):
        if not highlight_keys:
            return [''] * len(row)

        # If highlight_column is specified, only check that column
        if highlight_column and highlight_column in row.index:
            target_value = str(row[highlight_column])
            if any(str(key) in target_value for key in highlight_keys):
                return ['background-color: #e6f3ff; font-weight: bold'] * len(row)
        else:
            # Check all values in the row
            for value in row.values:
                if any(str(key) in str(value) for key in highlight_keys):
                    return ['background-color: #e6f3ff; font-weight: bold'] * len(row)

        return [''] * len(row)

    if title:
        st.subheader(title)

    styled_df = df.style.apply(highlight_rows, axis=1)
    st.dataframe(styled_df, width='stretch', hide_index=True)
