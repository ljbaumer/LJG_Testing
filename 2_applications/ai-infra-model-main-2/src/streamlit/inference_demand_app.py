import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.constants.inference_demand_scenario_dataclass import (
    SCENARIOS,
    TOKENS_PER_PAGE,
    TOKENS_PER_PARAGRAPH,
    TOKENS_PER_SENTENCE,
    TOKENS_PER_WORD,
    InferenceDemandPerUser,
)
from src.data_fetchers.TokenPriceFetcher import TokenPriceFetcher
from src.models.InferenceDemandModel import (
    AGENTIC_RATIOS,
    CHAT_RATIOS,
    InferenceCalculator,
    TokenRatios,
)
from src.utils.inference_demand_helpers import (
    align_token_cost_revenue_table,
)
from src.utils.streamlit_app_helpers import create_styled_dataframe, format_number_to_string

# Token conversion mapping
TOKEN_CONVERSION = {
    'word': TOKENS_PER_WORD,
    'sentence': TOKENS_PER_SENTENCE,
    'paragraph': TOKENS_PER_PARAGRAPH,
    'page': TOKENS_PER_PAGE
}

def setup_sidebar_config(prices_df):
    """Handle all sidebar model configuration inputs"""
    # Add scenario selector
    st.sidebar.title("Configuration")
    selected_scenario_name = st.sidebar.selectbox(
        "Select Scenario",
        options=list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index("Default")
    )
    selected_scenario = SCENARIOS[selected_scenario_name]

    # Convert scenario patterns to config format
    model_config = {}
    for pattern in selected_scenario.patterns:
        model_config[pattern.category] = {
            "num_prompts": pattern.num_prompts,
            "avg_input_tokens": pattern.avg_input_tokens,
            "avg_output_tokens": pattern.avg_output_tokens,
            "model_used": pattern.model_used
        }

    # Allow customization of the selected scenario
    with st.sidebar.expander("Customize Scenario", expanded=True):
        for category in model_config.keys():
            with st.sidebar.expander(f"{category.title()} Model Settings", expanded=True):
                model_config[category] = create_model_settings(
                    category,
                    model_config[category],
                    prices_df['model_id'].tolist()
                )

    return model_config

def display_comprehensive_metrics(calculator, daily_cost_per_user: float, yearly_cost_data: dict, monthly_subscription_price: float, num_users: float):
    """Display the Token Economics Dashboard"""
    # Calculate revenue metrics
    monthly_revenue = monthly_subscription_price * num_users
    yearly_revenue = monthly_revenue * 12

    # Get daily per-user and all-users data
    daily_per_user_data = calculator.daily_cost_per_user()
    daily_all_users_data = calculator.total_daily_cost_all_users()

    # Get the yearly cost from the data
    yearly_cost = yearly_cost_data['total_cost']
    yearly_profit = yearly_revenue - yearly_cost
    profit_margin = (yearly_profit / yearly_revenue) * 100 if yearly_revenue > 0 else 0

    # Note: workdays_only is a property of the calculator
    days_per_year = calculator.days_per_year

    # Create the integrated token economics dashboard that includes costs, revenue, and profit
    comprehensive_table = align_token_cost_revenue_table(
        daily_per_user_data,
        daily_all_users_data,
        yearly_cost_data,
        monthly_revenue,
        yearly_revenue,
        yearly_profit,
        profit_margin,
        int(num_users),
        days_per_year
    )
    st.markdown(comprehensive_table)

def display_token_assumption_table(model_config):
    """Create and display the text length usage breakdown table"""
    usage_breakdown = []
    for category, config in model_config.items():
        usage_breakdown.append({
            "Category": f"{category.title()} ({config['model_used']})",
            "Daily Prompts per User": config['num_prompts'],
            "Input Length": f"{config['avg_input_tokens']:,} tokens",
            "Output Length": f"{config['avg_output_tokens']:,} tokens"
        })
    create_styled_dataframe(
        pd.DataFrame(usage_breakdown),
        title="Token Length Assumptions"
    )

def text_unit_input(label: str, default_tokens: int, key: str, default_unit_index: int = 0) -> Tuple[float, str]:
    """
    Create an input for text length with unit selection.
    """
    units = ['page', 'paragraph', 'sentence', 'word']

    # Convert default tokens to the default unit
    default_unit = units[default_unit_index]
    default_value = default_tokens / TOKEN_CONVERSION[default_unit]

    # Round to 1 decimal place for better display
    default_value = round(default_value, 1)

    # Adjust column ratio for better balance
    col1, col2 = st.columns([3, 2])
    with col1:
        value = st.number_input(
            label,
            min_value=0.1,
            value=default_value,
            step=0.1,
            key=f"{key}_value"
        )
    with col2:
        unit = st.selectbox(
            "Unit",
            options=units,
            index=default_unit_index,
            key=f"{key}_unit"
        )

    return value, unit

def create_model_settings(category, config, available_models):
    """
    Create the settings interface for a single model category.
    
    Args:
        category: The model category
        config: The default configuration for this category
        available_models: List of available model IDs
    
    Returns:
        Updated configuration dictionary for this category
    """
    # Create a copy of the config to modify
    updated_config = config.copy()

    # Model selection dropdown
    default_model = config['model_used']
    updated_config['model_used'] = st.selectbox(
        "Select Model",
        options=available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
        key=f"{category}_model"
    )

    # Number of prompts input
    updated_config['num_prompts'] = st.number_input(
        "Daily Prompts per User",
        min_value=1,
        value=config['num_prompts'],
        key=f"{category}_prompts"
    )

    # Set default unit based on category
    if category == 'agent':
        default_unit_index = 0  # 'page'
    elif category == 'chat':
        default_unit_index = 1  # 'paragraph'
    elif category in ['small', 'tiny']:
        default_unit_index = 2  # 'sentence'

    # Input length with unit selection
    input_value, input_unit = text_unit_input(
        "Input Length",
        config['avg_input_tokens'],
        f"{category}_input",
        default_unit_index
    )

    # Output length with unit selection
    output_value, output_unit = text_unit_input(
        "Output Length",
        config['avg_output_tokens'],
        f"{category}_output",
        default_unit_index
    )

    # Convert to tokens for internal use
    input_tokens = input_value * TOKEN_CONVERSION[input_unit]
    output_tokens = output_value * TOKEN_CONVERSION[output_unit]

    updated_config['avg_input_tokens'] = round(input_tokens)
    updated_config['avg_output_tokens'] = round(output_tokens)

    # Store original units for display
    updated_config['input_display'] = f"{input_value} {input_unit}s"
    updated_config['output_display'] = f"{output_value} {output_unit}s"

    return updated_config

def config_to_inference_patterns(model_config) -> List[InferenceDemandPerUser]:
    """Convert model config to InferenceDemandPerUser objects"""
    inference_patterns = []
    for category, config in model_config.items():
        inference_patterns.append(
            InferenceDemandPerUser(
                category=category,
                num_prompts=config['num_prompts'],
                avg_input_tokens=config['avg_input_tokens'],
                avg_output_tokens=config['avg_output_tokens'],
                model_used=config['model_used']
            )
        )
    return inference_patterns

def calculate_cost_breakdown(calculator, model_config, num_users):
    """Calculate the cost breakdown for each model category."""
    cost_breakdown = []

    # Get cost breakdown from calculator
    user_costs = calculator.daily_cost_per_user()
    total_daily_cost = user_costs['total_cost'] * num_users

    for category, config in model_config.items():
        model_name = config['model_used']
        # Find corresponding model in the usage breakdown
        for model_id, details in user_costs['usage_breakdown'].items():
            if model_id == model_name and details['category'] == category:
                cost_per_user = details['total_category_cost']
                total_cost = cost_per_user * num_users

                cost_breakdown.append({
                    "Category": f"{category.title()} ({config['model_used']})",
                    "Daily Cost per User": f"${cost_per_user:.3f}",
                    "Total Inference Spend": format_number_to_string(total_cost, is_currency=True),
                    "Percentage": f"{(total_cost / total_daily_cost * 100):.1f}%"
                })
                break

    return cost_breakdown

def display_cost_breakdown_table(cost_breakdown):
    """Display the cost breakdown table."""
    create_styled_dataframe(
        pd.DataFrame(cost_breakdown),
        highlight_keys=["Total", "Cost"],
        title="Cost Breakdown by Model Category"
    )

def display_cost_breakdown_pie_chart(cost_breakdown):
    """Create and display the cost breakdown pie chart."""
    def convert_cost_string_to_float(cost_str):
        cost_str = cost_str.replace("$", "")
        if "T" in cost_str:
            return float(cost_str.replace("T", "")) * 1_000_000_000_000
        elif "B" in cost_str:
            return float(cost_str.replace("B", "")) * 1_000_000_000
        elif "M" in cost_str:
            return float(cost_str.replace("M", "")) * 1_000_000
        elif "K" in cost_str:
            return float(cost_str.replace("K", "")) * 1_000
        return float(cost_str)

    fig = go.Figure(data=[go.Pie(
        labels=[item["Category"] for item in cost_breakdown],
        values=[convert_cost_string_to_float(item["Total Inference Spend"])
               for item in cost_breakdown],
        hole=.3,
        textfont=dict(size=20),
        marker=dict(colors=['blue', 'red', 'yellow', 'green'])
    )])

    fig.update_layout(
        title="Cost Breakdown by Model Category",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def format_price_dataframe(prices_df):
    """Format the prices dataframe for display."""
    # Convert prices to dollars per million tokens and format
    def format_price_per_million(price):
        price_in_millions = price * 1_000_000
        if price_in_millions < 0.90:
            return f"{price_in_millions * 100:.2f}¢"  # Convert to cents
        return f"${price_in_millions:.2f}"

    # Create a copy to avoid modifying the original
    formatted_df = prices_df.copy()

    # Format price columns
    price_cols = ['prompt_price', 'completion_price']
    for col in price_cols:
        formatted_df[col] = formatted_df[col].apply(format_price_per_million)

    # Reorder and rename columns
    formatted_df = formatted_df.rename(columns={
        'model_id': 'Model',
        'prompt_price': 'Input (Prompt) Price per 1M tokens',
        'completion_price': 'Output (Completion) Price per 1M tokens'
    })

    # Sort by completion price (high to low)
    def price_to_float(price_str):
        if '¢' in price_str:
            return float(price_str.replace('¢', '')) / 100
        return float(price_str.replace('$', ''))

    formatted_df['sort_value'] = formatted_df['Output (Completion) Price per 1M tokens'].apply(price_to_float)
    formatted_df = formatted_df.sort_values('sort_value', ascending=False).drop('sort_value', axis=1)

    # Reset index to get numbered rows starting from 1
    formatted_df = formatted_df.reset_index(drop=True)
    formatted_df.index = formatted_df.index + 1

    return formatted_df

def display_latest_model_pricing(price_csv_path, prices_df=None):
    """
    Display the model pricing table.
    
    Args:
        price_csv_path: Path to the CSV file with model pricing data
        prices_df: Optional DataFrame with model pricing data. If None, will read from CSV.
    """
    # Use the provided DataFrame if available, otherwise read from CSV
    if prices_df is None:
        prices_df = pd.read_csv(price_csv_path)

    formatted_df = format_price_dataframe(prices_df)
    # get date from filepath
    date_str = os.path.basename(price_csv_path).split('_')[2].split('.')[0]
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    formatted_date = date_obj.strftime("%B %-d, %Y")
    st.subheader(f"Current Model Pricing as of {formatted_date}")

    st.dataframe(
        formatted_df[['Model', 'Input (Prompt) Price per 1M tokens', 'Output (Completion) Price per 1M tokens']],
        use_container_width=True,
        column_config={
            "Model": st.column_config.TextColumn(width="medium"),
            "Input (Prompt) Price per 1M tokens": st.column_config.TextColumn(width="medium"),
            "Output (Completion) Price per 1M tokens": st.column_config.TextColumn(width="medium")
        }
    )

def generate_cost_narrative(num_queries, input_tokens, output_tokens, model_name, price_fetcher):
    """
    Generate an ad-libs style narrative about inference costs based on user inputs.
    
    Args:
        num_queries: Number of queries per year (in trillions)
        input_tokens: Average input tokens
        output_tokens: Average output tokens
        model_name: Name of the model used
        price_fetcher: TokenPriceFetcher instance with loaded prices
    
    Returns:
        A formatted narrative string
    """
    # Calculate cost per query using the token prices directly
    input_price, output_price = price_fetcher.get_model_token_prices(model_name)
    cost_per_query = (input_tokens * input_price) + (output_tokens * output_price)

    # Calculate total annual cost
    total_annual_cost = cost_per_query * num_queries

    # Format the narrative
    narrative = f"""
    Model API pricing would cost approximately **{format_number_to_string(cost_per_query, is_currency=True)}** per query (converting the above figures to tokens we get an average input length of 
    **{input_tokens} tokens** and average output length of **{output_tokens} tokens**) using the **{model_name}** model. 
    
    This gives us a general sense of the "underlying cost" of the model in terms of infrastructure and amortized R&D.
    
    Google Search receives roughly 1.8 trillion queries per year, If ChatGPT were to get **{num_queries / 1_000_000_000_000} trillion queries per year**,
    it would cost **{format_number_to_string(total_annual_cost, is_currency=True)}** in inference to run every year.
    """

    return narrative

def main():
    st.set_page_config(layout="wide")

    # Add custom CSS to make sidebar wider
    st.markdown("""
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 450px;
            max-width: 450px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("LLM Token Usage Calculator")

    # Add OpenAI tokenizer link and Google scale reference
    st.info("Use [OpenAI Tokenizer](https://platform.openai.com/tokenizer) for token calculations | [Google processed nearly 1 quadrillion tokens in June](https://www.datacenterdynamics.com/en/news/google-processed-nearly-one-quadrillion-tokens-in-june-deepminds-demis-hassabis-says/)")

    # Mode selector
    calculation_mode = st.radio(
        "What do you want to calculate?",
        ["Total Annual Cost", "Required Token Volume (Chip Lifecycle)"],
        index=1  # Default to the chip lifecycle mode
    )

    # First, get the latest price CSV path
    temp_fetcher = TokenPriceFetcher()
    price_csv_path = temp_fetcher.get_model_prices()

    # Load the prices DataFrame directly
    prices_df = pd.read_csv(price_csv_path)

    # Now create a TokenPriceFetcher with the loaded prices to use throughout the app
    price_fetcher = TokenPriceFetcher(price_csv_path)

    # Sidebar configuration
    st.sidebar.title("Configuration")

    # Add workdays toggle
    workdays_only = st.sidebar.toggle(
        "Calculate costs based on workdays only (260 days/year)",
        value=True,
        help="If enabled, costs will be calculated based on 260 workdays per year. If disabled, costs will be based on 365 calendar days."
    )

    # Conditional inputs based on calculation mode
    if calculation_mode == "Total Annual Cost":
        num_users = st.sidebar.number_input("Number of Users (in millions)",
                                          min_value=0.1,
                                          value=10.0,
                                          step=0.1) * 1_000_000
        chip_capex = None
        years = None
        annual_price_decline = None
        blended_ratios = None

    else:  # Required Token Volume (Chip Lifecycle)
        chip_capex_billions = st.sidebar.number_input("GPU Server CAPEX for Inference (billions $)",
                                                     min_value=0.001,
                                                     value=60.0,
                                                     step=0.1)
        chip_capex = chip_capex_billions * 1_000_000_000  # Convert to actual dollars

        model_name = st.sidebar.selectbox("Model",
                                        ["openai/gpt-5", "anthropic/claude-opus-4", "anthropic/claude-sonnet-4", "openai/gpt-4", "openai/gpt-3.5-turbo"],
                                        index=0)

        years = st.sidebar.number_input("Time Horizon (years)",
                                      min_value=1,
                                      max_value=10,
                                      value=5)

        annual_price_decline = st.sidebar.number_input("Annual Price Decline (%)",
                                                      min_value=0.0,
                                                      max_value=50.0,
                                                      value=30.0,
                                                      step=1.0) / 100

        # Workload Mix with percentage sliders (remove dropdown abstraction)
        agentic_percent = st.sidebar.slider("Agentic Workload %", 0, 100, 50)
        chat_percent = 100 - agentic_percent
        st.sidebar.write(f"Chat Workload: {chat_percent}%")

        # Calculate blended ratios
        blended_ratios = TokenRatios(
            input_ratio=agentic_percent/100 * AGENTIC_RATIOS.input_ratio + chat_percent/100 * CHAT_RATIOS.input_ratio,
            output_ratio=agentic_percent/100 * AGENTIC_RATIOS.output_ratio + chat_percent/100 * CHAT_RATIOS.output_ratio,
            cache_hit_rate=AGENTIC_RATIOS.cache_hit_rate  # Keep constant at 94%
        )

        st.sidebar.info(f"Blended ratios: {blended_ratios.input_ratio:.1%} input, {blended_ratios.output_ratio:.1%} output, 94% cache hit rate")

        num_users = None

    # Only show scenario selection for "Total Annual Cost" mode
    if calculation_mode == "Total Annual Cost":
        selected_scenario_name = st.sidebar.selectbox(
            "Select Scenario",
            options=list(SCENARIOS.keys()),
            index=list(SCENARIOS.keys()).index("Default")
        )
        selected_scenario = SCENARIOS[selected_scenario_name]

        # Monthly subscription price input
        monthly_subscription_price = st.sidebar.number_input(
            "Monthly Subscription Price ($)",
            min_value=0.0,
            value=selected_scenario.monthly_subscription_price,
            step=1.0
        )
    else:
        # Use default scenario for other modes
        selected_scenario = SCENARIOS["Default"]
        monthly_subscription_price = selected_scenario.monthly_subscription_price

    # Convert scenario patterns to config format
    model_config = {}
    for pattern in selected_scenario.patterns:
        model_config[pattern.category] = {
            "num_prompts": pattern.num_prompts,
            "avg_input_tokens": pattern.avg_input_tokens,
            "avg_output_tokens": pattern.avg_output_tokens,
            "model_used": pattern.model_used
        }

    # Allow customization of the selected scenario
    with st.sidebar.expander("Customize Scenario", expanded=True):
        for category in model_config.keys():
            with st.sidebar.expander(f"{category.title()} Model Settings", expanded=True):
                model_config[category] = create_model_settings(
                    category,
                    model_config[category],
                    prices_df['model_id'].tolist()
                )

    # Convert model config to inference patterns
    inference_patterns = config_to_inference_patterns(model_config)

    # Initialize calculator with default user count for calculation purposes
    default_users = 1 if calculation_mode != "Total Annual Cost" else int(num_users)
    calculator = InferenceCalculator(price_fetcher, inference_patterns, default_users, workdays_only=workdays_only)

    # Handle different calculation modes
    if calculation_mode == "Total Annual Cost":
        # Bottom-up calculation: users → total cost
        result = calculator.calculate_total_cost(int(num_users))

        st.header("Total Annual Cost Calculation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Users", f"{num_users:,.0f}")
        with col2:
            st.metric("Cost per User per Year", f"${result['cost_per_user_per_year']:,.2f}")
        with col3:
            st.metric("Total Annual Cost", f"${result['total_annual_cost']:,.0f}")

        # Display comprehensive metrics for bottom-up calculation
        daily_cost_data = calculator.daily_cost_per_user()
        yearly_cost_data = calculator.yearly_cost_all_users()
        display_comprehensive_metrics(calculator, daily_cost_data['total_cost'], yearly_cost_data, monthly_subscription_price, num_users)

    else:  # Required Token Volume (Chip Lifecycle)
        # Chip lifecycle calculation: CAPEX + years → required token sales per year
        result = calculator.calculate_breakeven_tokens_lifecycle(
            chip_capex, model_name, blended_ratios, years, annual_price_decline
        )

        st.header("Chip Lifecycle Cost Recovery Analysis")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Chip CAPEX", format_number_to_string(result['chip_capex'], is_currency=True))
        with col2:
            st.metric("Time Horizon", f"{result['years']} years")
        with col3:
            st.metric("Annual Recovery Target", format_number_to_string(result['annual_capex_recovery'], is_currency=True))
        with col4:
            st.metric("Annual Per Token Price Decline", f"{result['annual_price_decline']:.0%}")

        # Base Pricing & Workload Details (moved up as assumptions)
        st.subheader("Model Pricing & Workload Assumptions")
        col1, col2 = st.columns(2)
        with col1:
            base_pricing = result['base_pricing']
            st.write(f"**Model:** {base_pricing['model_name']}")
            st.write(f"**List Price (Input):** ${base_pricing['base_input_price'] * 1_000_000:.2f} per 1M tokens")
            st.write(f"**List Price (Output):** ${base_pricing['base_output_price'] * 1_000_000:.2f} per 1M tokens")
        with col2:
            ratios = result['token_ratios']
            st.write(f"**Workload Mix:** {agentic_percent}% agentic, {chat_percent}% chat")
            st.write(f"**Blended Input Ratio:** {ratios.input_ratio:.1%}")
            st.write(f"**Blended Output Ratio:** {ratios.output_ratio:.1%}")

        # Year-by-year breakdown table
        st.subheader("Year-by-Year Token Sales Requirements")

        breakdown_data = []
        for year_data in result['yearly_breakdown']:
            total_annual = year_data['total_tokens_needed']
            total_monthly = total_annual // 12
            breakdown_data.append({
                'Year': year_data['year'],
                'Blended Cost per 1M Tokens': f"${year_data['blended_cost_per_token'] * 1_000_000:.2f}",
                'Total Tokens (Monthly)': format_number_to_string(total_monthly),
                'Total Tokens (Annual)': format_number_to_string(total_annual),
                'Input Tokens': format_number_to_string(year_data['input_tokens_needed']),
                'Output Tokens': format_number_to_string(year_data['output_tokens_needed'])
            })

        st.dataframe(breakdown_data, use_container_width=True)
        st.caption("*Blended cost per 1M tokens is weighted average of input/output costs based on your workload mix proportions")

        # Simple explanation above the detailed breakdown
        total_tokens = result['total_tokens_over_lifecycle']
        avg_price_per_token = chip_capex / total_tokens if total_tokens > 0 else 0
        avg_price_per_million = avg_price_per_token * 1_000_000

        st.info("""
        **Bottom Line:** To recover your ${} investment, you need to monetize **{} tokens** at an average rate of **${:.2f} per million tokens** over the 5-year lifecycle. Monetization rate per token is heavily dependent on business model.
        """.format(
            format_number_to_string(chip_capex, is_currency=True),
            format_number_to_string(int(total_tokens)),
            avg_price_per_million
        ))


    # Show additional sections only for "Total Annual Cost" mode
    if calculation_mode == "Total Annual Cost":
        # Add the ad-libs narrative generator
        st.header("Cost Overview")
        with st.expander("Generate a cost narrative", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                num_queries = st.number_input("Queries per year (trillions)",
                                             min_value=0.001,
                                             value=5.0,
                                             step=0.1) * 1_000_000_000_000

                input_tokens = st.number_input(
                    "Average input paragraphs",
                    min_value=0.1,
                    value=10.0,
                    step=0.1
                ) * TOKENS_PER_PARAGRAPH

            with col2:
                available_models = prices_df['model_id'].tolist()
                narrative_model_name = st.selectbox(
                    "Model to use",
                    options=available_models,
                    index=available_models.index("openai/gpt-4o")
                        if "openai/gpt-4o" in available_models else 0
                )

                output_tokens = st.number_input(
                    "Average output paragraphs",
                    min_value=0.1,
                    value=3.0,
                    step=0.1
                ) * TOKENS_PER_PARAGRAPH

            narrative = generate_cost_narrative(
                num_queries,
                input_tokens,
                output_tokens,
                narrative_model_name,
                price_fetcher
            )

            st.markdown(narrative)

        # Display sections
        display_token_assumption_table(model_config)
        cost_breakdown = calculate_cost_breakdown(calculator, model_config, num_users)
        display_cost_breakdown_table(cost_breakdown)
        display_cost_breakdown_pie_chart(cost_breakdown)

    # Always show model pricing
    display_latest_model_pricing(price_csv_path, prices_df)

    st.markdown("""
    **Hard Questions & Caveats**:
    * Consumer subscription revenue is complex to account for
    * Enterprise revenue models have significant variability
    * Different providers have varying gross margins, capex, and overhead, making true cost estimation challenging
    * Tokens per GPU hour involve intricate tradeoffs between throughput and interactivity so it's hard to know the cost structure
    * This calculator provides estimates based on average token usage patterns, there is no data on this so I am just giving my best guess
    """)

    st.markdown("""
    **ChatGPT Plus Message Limits**:
    * GPT-4o: 80 messages per 3 hours
    * GPT-4: 40 messages per 3 hours
    * OpenAI o1-preview: 50 messages per week
    * OpenAI o1-mini: 50 messages per day
    * [Source: OpenAI Help Center](https://help.openai.com/en/articles/6950777-what-is-chatgpt-plus)
    """)

if __name__ == "__main__":
    main()
