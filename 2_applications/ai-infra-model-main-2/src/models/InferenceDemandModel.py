from typing import Dict, List, NamedTuple, Union

from src.constants.inference_demand_scenario_dataclass import (
    DEFAULT_SCENARIO,
    TOKENS_PER_PARAGRAPH,
    InferenceDemandPerUser,
)
from src.data_fetchers.TokenPriceFetcher import TokenPriceFetcher


class TokenRatios(NamedTuple):
    input_ratio: float
    output_ratio: float
    cache_hit_rate: float

class TokenQuantities(NamedTuple):
    input_tokens: int
    output_tokens: int

# Named constants for different workload types
AGENTIC_RATIOS = TokenRatios(input_ratio=0.95, output_ratio=0.05, cache_hit_rate=0.94)  # Monster input from file reads
CHAT_RATIOS = TokenRatios(input_ratio=0.80, output_ratio=0.20, cache_hit_rate=0.94)     # Conversational back-and-forth

class InferenceCalculator:
    def __init__(self, price_fetcher: TokenPriceFetcher, inference_demand_patterns: Union[List[InferenceDemandPerUser], InferenceDemandPerUser], num_users: int = 1, workdays_only: bool = True):
        """
        Initialize InferenceCalculator with a price fetcher and inference demand patterns.
        
        Args:
            price_fetcher: TokenPriceFetcher instance for getting model prices
            inference_demand_patterns: Either a single InferenceDemandPerUser object or a list of them
            num_users: Number of users for cost calculation
            workdays_only: Whether to calculate costs based on workdays only (260 days) or all days (365 days)
        """
        # Convert single pattern to list if needed
        if isinstance(inference_demand_patterns, InferenceDemandPerUser):
            self.inference_demand_patterns = [inference_demand_patterns]
        else:
            self.inference_demand_patterns = inference_demand_patterns

        self.num_users = num_users
        self.price_fetcher = price_fetcher
        self.workdays_only = workdays_only
        self.days_per_year = 260 if workdays_only else 365  # 260 workdays or 365 calendar days

    def calculate_query_cost(self, model_name: str, input_tokens: float, output_tokens: float) -> float:
        """
        Calculate cost for a specific query based on model and token counts.
        
        Args:
            model_name: Name of the model to use
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            float: Total cost of the query
        """
        # Get the token prices from the price fetcher
        input_price, output_price = self.price_fetcher.get_model_token_prices(model_name)

        # Calculate costs
        input_cost = input_tokens * input_price
        output_cost = output_tokens * output_price
        total_cost = input_cost + output_cost

        return total_cost

    def daily_cost_per_user(self) -> Dict:
        """Calculate the total daily token cost per user across all model categories with breakdown."""
        total_daily_cost = 0.0
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        usage_costs = {}

        for inference_demand_pattern in self.inference_demand_patterns:
            # Unpack the inference demand pattern
            model_name = inference_demand_pattern.model_used
            input_tokens = inference_demand_pattern.avg_input_tokens
            output_tokens = inference_demand_pattern.avg_output_tokens
            num_prompts = inference_demand_pattern.num_prompts

            # Calculate cost for this inference demand pattern
            category_cost = self.calculate_query_cost(model_name, input_tokens, output_tokens) * num_prompts

            # Calculate total tokens for this pattern
            daily_input_tokens = input_tokens * num_prompts
            daily_output_tokens = output_tokens * num_prompts
            daily_total_tokens = daily_input_tokens + daily_output_tokens

            total_input_tokens += daily_input_tokens
            total_output_tokens += daily_output_tokens

            usage_costs[inference_demand_pattern.model_used] = {
                'category': inference_demand_pattern.category,
                'total_category_cost': category_cost,
                'input_tokens': daily_input_tokens,
                'output_tokens': daily_output_tokens,
                'total_tokens': daily_total_tokens
            }
            total_daily_cost += category_cost

        return {
            'total_cost': total_daily_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'usage_breakdown': usage_costs
        }

    def total_daily_cost_all_users(self) -> Dict:
        """Calculate the total daily cost across all users by multiplying per-user cost."""
        per_user_costs = self.daily_cost_per_user()

        # Create a new dictionary with the same structure but multiplied costs
        total_costs = {
            'total_cost': per_user_costs['total_cost'] * self.num_users,
            'total_input_tokens': per_user_costs['total_input_tokens'] * self.num_users,
            'total_output_tokens': per_user_costs['total_output_tokens'] * self.num_users,
            'total_tokens': per_user_costs['total_tokens'] * self.num_users,
            'usage_breakdown': {}
        }

        # Multiply each category cost by number of users
        for model_id, details in per_user_costs['usage_breakdown'].items():
            total_costs['usage_breakdown'][model_id] = {
                'category': details['category'],
                'total_category_cost': details['total_category_cost'] * self.num_users,
                'input_tokens': details['input_tokens'] * self.num_users,
                'output_tokens': details['output_tokens'] * self.num_users,
                'total_tokens': details['total_tokens'] * self.num_users
            }

        return total_costs

    def yearly_cost_all_users(self) -> Dict:
        """Calculate the total yearly cost across all users based on workdays or all days."""
        daily_costs = self.total_daily_cost_all_users()

        # Create a new dictionary with the same structure but multiplied costs
        yearly_costs = {
            'total_cost': daily_costs['total_cost'] * self.days_per_year,
            'total_input_tokens': daily_costs['total_input_tokens'] * self.days_per_year,
            'total_output_tokens': daily_costs['total_output_tokens'] * self.days_per_year,
            'total_tokens': daily_costs['total_tokens'] * self.days_per_year,
            'usage_breakdown': {}
        }

        # Multiply each category cost by number of days
        for model_id, details in daily_costs['usage_breakdown'].items():
            yearly_costs['usage_breakdown'][model_id] = {
                'category': details['category'],
                'total_category_cost': details['total_category_cost'] * self.days_per_year,
                'input_tokens': details['input_tokens'] * self.days_per_year,
                'output_tokens': details['output_tokens'] * self.days_per_year,
                'total_tokens': details['total_tokens'] * self.days_per_year
            }

        return yearly_costs

    def yearly_cost_per_user(self) -> Dict:
        """Calculate the total yearly cost for a single user based on workdays or all days."""
        daily_costs = self.daily_cost_per_user()

        # Create a new dictionary with the same structure but multiplied costs
        yearly_costs = {
            'total_cost': daily_costs['total_cost'] * self.days_per_year,
            'total_input_tokens': daily_costs['total_input_tokens'] * self.days_per_year,
            'total_output_tokens': daily_costs['total_output_tokens'] * self.days_per_year,
            'total_tokens': daily_costs['total_tokens'] * self.days_per_year,
            'usage_breakdown': {}
        }

        # Multiply each category cost by number of days
        for model_id, details in daily_costs['usage_breakdown'].items():
            yearly_costs['usage_breakdown'][model_id] = {
                'category': details['category'],
                'total_category_cost': details['total_category_cost'] * self.days_per_year,
                'input_tokens': details['input_tokens'] * self.days_per_year,
                'output_tokens': details['output_tokens'] * self.days_per_year,
                'total_tokens': details['total_tokens'] * self.days_per_year
            }

        return yearly_costs

    # ===== EXISTING API (PRESERVE) =====
    def cost_per_user_per_year(self) -> float:
        """Calculate annual cost per user using token-based calculations"""
        return self.yearly_cost_per_user()['total_cost']

    def calculate_total_cost(self, num_users: int) -> Dict:
        """Bottom-up: users → total cost"""
        cost_per_user = self.cost_per_user_per_year()
        return {
            'total_annual_cost': cost_per_user * num_users,
            'cost_per_user_per_year': cost_per_user,
            'num_users': num_users,
        }

    # ===== NEW TOP-DOWN METHODS =====
    def calculate_breakeven_users(self, capex_annual_cost: float) -> Dict:
        """Top-down: CAPEX → required users"""
        cost_per_user = self.cost_per_user_per_year()

        if cost_per_user == 0:
            required_users = float('inf')
        else:
            required_users = capex_annual_cost / cost_per_user

        return {
            'required_users': required_users,
            'required_users_rounded': int(round(required_users)) if required_users != float('inf') else 0,
            'cost_per_user_per_year': cost_per_user,
            'capex_annual_cost': capex_annual_cost
        }

    def calculate_breakeven_tokens(self, inference_capex: float, model_name: str,
                                  token_ratios: TokenRatios = AGENTIC_RATIOS) -> TokenQuantities:
        """Top-down: CAPEX + model → required tokens to sell (cache-aware)
        
        Uses TokenRatios with embedded cache hit rates for realistic economics.
        Cache creates = new content (full cost), cache reads = essentially free.
        
        Returns:
            TokenQuantities: (input_tokens_needed_annually, output_tokens_needed_annually)
        """
        # Get token prices
        input_price, output_price = self.price_fetcher.get_model_token_prices(model_name)

        # Calculate effective pricing considering cache hit rates
        # Cache creates are charged full price, cache reads are essentially free
        effective_input_price = input_price * (1 - token_ratios.cache_hit_rate)
        effective_output_price = output_price * (1 - token_ratios.cache_hit_rate)

        # We need to solve: inference_capex = input_tokens * input_ratio * effective_input_price + output_tokens * output_ratio * effective_output_price
        # Where input_tokens + output_tokens = total_tokens
        # And input_tokens = total_tokens * input_ratio, output_tokens = total_tokens * output_ratio

        # Substitute to get: inference_capex = total_tokens * (input_ratio * effective_input_price + output_ratio * effective_output_price)
        cost_per_token_blended = (token_ratios.input_ratio * effective_input_price +
                                token_ratios.output_ratio * effective_output_price)

        if cost_per_token_blended == 0:
            return TokenQuantities(0, 0)

        total_tokens_needed = inference_capex / cost_per_token_blended

        input_tokens_needed = int(total_tokens_needed * token_ratios.input_ratio)
        output_tokens_needed = int(total_tokens_needed * token_ratios.output_ratio)

        return TokenQuantities(input_tokens_needed, output_tokens_needed)

    def calculate_breakeven_tokens_lifecycle(self, chip_capex: float, model_name: str,
                                            token_ratios: TokenRatios = AGENTIC_RATIOS,
                                            years: int = 5,
                                            annual_price_decline: float = 0.20) -> Dict:
        """Chip Lifecycle: CAPEX + years → required token sales per year
        
        Business Goal: "I spent $X on chips. How many tokens must I sell each year to break even?"
        
        Args:
            chip_capex: Total upfront chip investment
            model_name: Model for base pricing
            token_ratios: Workload mix (agentic/chat blend)
            years: Chip lifecycle (default 5)
            annual_price_decline: Linear price decline per year (default 20%)
        
        Returns:
            Dict with year-by-year breakdown and totals
        """
        # Get base token prices
        input_price, output_price = self.price_fetcher.get_model_token_prices(model_name)

        # Calculate effective pricing considering cache hit rates
        base_effective_input_price = input_price * (1 - token_ratios.cache_hit_rate)
        base_effective_output_price = output_price * (1 - token_ratios.cache_hit_rate)
        base_blended_cost = (token_ratios.input_ratio * base_effective_input_price +
                            token_ratios.output_ratio * base_effective_output_price)

        # Annual CAPEX recovery requirement
        annual_capex_recovery = chip_capex / years

        # Calculate year-by-year requirements
        yearly_breakdown = []
        total_tokens_lifecycle = 0

        for year in range(1, years + 1):
            # Exponential price decline: starting_price * (1 - decline_rate)^(year-1)
            price_multiplier = (1.0 - annual_price_decline) ** (year - 1)
            year_blended_cost = base_blended_cost * price_multiplier

            # Calculate total tokens needed this year
            total_tokens_this_year = annual_capex_recovery / year_blended_cost if year_blended_cost > 0 else 0

            # Break down into input/output tokens
            input_tokens_needed = int(total_tokens_this_year * token_ratios.input_ratio)
            output_tokens_needed = int(total_tokens_this_year * token_ratios.output_ratio)

            yearly_breakdown.append({
                'year': year,
                'price_multiplier': price_multiplier,
                'blended_cost_per_token': year_blended_cost,
                'input_tokens_needed': input_tokens_needed,
                'output_tokens_needed': output_tokens_needed,
                'total_tokens_needed': int(total_tokens_this_year),
                'revenue_target': annual_capex_recovery
            })

            total_tokens_lifecycle += int(total_tokens_this_year)

        return {
            'chip_capex': chip_capex,
            'years': years,
            'annual_capex_recovery': annual_capex_recovery,
            'annual_price_decline': annual_price_decline,
            'yearly_breakdown': yearly_breakdown,
            'total_tokens_over_lifecycle': total_tokens_lifecycle,
            'base_pricing': {
                'model_name': model_name,
                'base_input_price': input_price,
                'base_output_price': output_price,
                'base_effective_input_price': base_effective_input_price,
                'base_effective_output_price': base_effective_output_price,
                'base_blended_cost': base_blended_cost
            },
            'token_ratios': token_ratios
        }


# Example usage
if __name__ == "__main__":
    price_fetcher = TokenPriceFetcher()
    calculator = InferenceCalculator(price_fetcher, DEFAULT_SCENARIO.patterns, num_users=1000)
    print(calculator.daily_cost_per_user())

    # Detailed cost calculation examples
    print("\nDetailed Cost Breakdown:")

    # Print query cost for each inference demand pattern
    for inference_demand_pattern in calculator.inference_demand_patterns:
        model_name = inference_demand_pattern.model_used
        input_tokens = inference_demand_pattern.avg_input_tokens
        output_tokens = inference_demand_pattern.avg_output_tokens

        query_cost = calculator.calculate_query_cost(model_name, input_tokens, output_tokens)
        print(f"\nModel: {model_name}")
        print(f"  Query Cost: ${query_cost:.4f}")

    # Print daily cost summaries
    print(f"\nDefault config daily cost per user: {calculator.daily_cost_per_user()}")
    print(f"\nTotal Daily Cost Breakdown: {calculator.total_daily_cost_all_users()}")
    print(f"\nTotal Yearly Cost Breakdown ({calculator.days_per_year} days): {calculator.yearly_cost_all_users()}")



    large_only_pattern = InferenceDemandPerUser(
            category="large",
            num_prompts=100,
            avg_input_tokens=10 * TOKENS_PER_PARAGRAPH,
            avg_output_tokens=3 * TOKENS_PER_PARAGRAPH,
            model_used="openai/gpt-4o"
        )

    large_only_calculator = InferenceCalculator(price_fetcher, large_only_pattern, num_users=85_000_000)
    query_cost = large_only_calculator.calculate_query_cost(
        large_only_pattern.model_used,
        large_only_pattern.avg_input_tokens,
        large_only_pattern.avg_output_tokens
    )
    total_daily_cost = large_only_calculator.total_daily_cost_all_users()
    total_yearly_cost = large_only_calculator.yearly_cost_all_users()

    print("\n" + "=" * 50)
    print("🔍 Large Category Inference Cost Analysis 🔍")
    print("=" * 50)
    print(f"Price per Query:     ${query_cost:,.4f}")
    print(f"Total Daily Spend:   ${total_daily_cost['total_cost']:,.2f}")
    print(f"Yearly Estimated Spend ({large_only_calculator.days_per_year} days): ${total_yearly_cost['total_cost']:,.2f}")
    print("=" * 50)

# number of paragrpahs in, parapgrahs out
