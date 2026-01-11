
import pandas as pd
import pytest

from src.constants.inference_demand_scenario_dataclass import (
    InferenceDemandPerUser,
)
from src.data_fetchers.TokenPriceFetcher import TokenPriceFetcher
from src.models.InferenceDemandModel import InferenceCalculator


# Sample test data
@pytest.fixture
def sample_price_csv(tmp_path):
    """Create a sample price CSV file for testing."""
    df = pd.DataFrame({
        'model_id': ['openai/gpt-4o', 'anthropic/claude-3-opus', 'test-model'],
        'prompt_price': [0.00001, 0.00002, 0.00003],
        'completion_price': [0.00003, 0.00004, 0.00005]
    })
    csv_path = tmp_path / "test_prices.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def price_fetcher(sample_price_csv):
    """Create a TokenPriceFetcher with the sample price data."""
    fetcher = TokenPriceFetcher(sample_price_csv)
    return fetcher

class TestInferenceCalculatorWithDifferentPatterns:
    def test_single_inference_pattern(self, price_fetcher):
        """Test with a single inference pattern."""
        pattern = InferenceDemandPerUser(
            category="test-single",
            num_prompts=10,
            avg_input_tokens=100,
            avg_output_tokens=200,
            model_used="openai/gpt-4o"
        )

        calculator = InferenceCalculator(price_fetcher, pattern)

        # Verify pattern was properly stored
        assert len(calculator.inference_demand_patterns) == 1
        assert calculator.inference_demand_patterns[0].category == "test-single"

        # Verify cost calculation
        cost = calculator.calculate_query_cost(
            pattern.model_used,
            pattern.avg_input_tokens,
            pattern.avg_output_tokens
        )
        expected_cost = 100 * 0.00001 + 200 * 0.00003
        assert cost == pytest.approx(expected_cost)

        # Verify daily cost
        daily_cost = calculator.daily_cost_per_user()
        assert daily_cost['total_cost'] == pytest.approx(expected_cost * 10)  # 10 prompts
        assert "openai/gpt-4o" in daily_cost['usage_breakdown']
        assert daily_cost['usage_breakdown']["openai/gpt-4o"]['category'] == "test-single"

    def test_multiple_inference_patterns(self, price_fetcher):
        """Test with multiple inference patterns."""
        patterns = [
            InferenceDemandPerUser(
                category="small",
                num_prompts=5,
                avg_input_tokens=50,
                avg_output_tokens=100,
                model_used="openai/gpt-4o"
            ),
            InferenceDemandPerUser(
                category="large",
                num_prompts=2,
                avg_input_tokens=500,
                avg_output_tokens=1000,
                model_used="anthropic/claude-3-opus"
            )
        ]

        calculator = InferenceCalculator(price_fetcher, patterns)

        # Verify patterns were properly stored
        assert len(calculator.inference_demand_patterns) == 2

        # Verify cost calculation for each pattern
        small_cost = calculator.calculate_query_cost(
            patterns[0].model_used,
            patterns[0].avg_input_tokens,
            patterns[0].avg_output_tokens
        )
        expected_small_cost = 50 * 0.00001 + 100 * 0.00003
        assert small_cost == pytest.approx(expected_small_cost)

        large_cost = calculator.calculate_query_cost(
            patterns[1].model_used,
            patterns[1].avg_input_tokens,
            patterns[1].avg_output_tokens
        )
        expected_large_cost = 500 * 0.00002 + 1000 * 0.00004
        assert large_cost == pytest.approx(expected_large_cost)

        # Verify daily cost
        daily_cost = calculator.daily_cost_per_user()
        expected_total = (expected_small_cost * 5) + (expected_large_cost * 2)
        assert daily_cost['total_cost'] == pytest.approx(expected_total)
        assert len(daily_cost['usage_breakdown']) == 2

    def test_empty_pattern_list(self, price_fetcher):
        """Test with an empty list of patterns."""
        calculator = InferenceCalculator(price_fetcher, [])

        # Verify empty list was properly stored
        assert len(calculator.inference_demand_patterns) == 0

        # Verify daily cost with no patterns
        daily_cost = calculator.daily_cost_per_user()
        assert daily_cost['total_cost'] == 0
        assert len(daily_cost['usage_breakdown']) == 0

    def test_pattern_with_unknown_model(self, price_fetcher):
        """Test with a pattern that uses an unknown model."""
        pattern = InferenceDemandPerUser(
            category="unknown-model",
            num_prompts=10,
            avg_input_tokens=100,
            avg_output_tokens=200,
            model_used="nonexistent-model"
        )

        calculator = InferenceCalculator(price_fetcher, pattern)

        # Verify pattern was properly stored
        assert len(calculator.inference_demand_patterns) == 1

        # Verify cost calculation raises error for unknown model
        with pytest.raises(ValueError, match="Unknown model: nonexistent-model"):
            calculator.calculate_query_cost(
                pattern.model_used,
                pattern.avg_input_tokens,
                pattern.avg_output_tokens
            )

        # Verify daily cost calculation also raises error
        with pytest.raises(ValueError, match="Unknown model: nonexistent-model"):
            calculator.daily_cost_per_user()

    def test_pattern_with_zero_tokens(self, price_fetcher):
        """Test with a pattern that has zero tokens."""
        pattern = InferenceDemandPerUser(
            category="zero-tokens",
            num_prompts=10,
            avg_input_tokens=0,
            avg_output_tokens=0,
            model_used="openai/gpt-4o"
        )

        calculator = InferenceCalculator(price_fetcher, pattern)

        # Verify cost calculation with zero tokens
        cost = calculator.calculate_query_cost(
            pattern.model_used,
            pattern.avg_input_tokens,
            pattern.avg_output_tokens
        )
        assert cost == 0

        # Verify daily cost with zero tokens
        daily_cost = calculator.daily_cost_per_user()
        assert daily_cost['total_cost'] == 0
        assert "openai/gpt-4o" in daily_cost['usage_breakdown']
        assert daily_cost['usage_breakdown']["openai/gpt-4o"]['total_category_cost'] == 0

    # def test_pattern_with_negative_values(self, price_fetcher):
    #     """Test with a pattern that has negative values (invalid but should be handled)."""
    #     pattern = InferenceDemandPerUser(
    #         category="negative-values",
    #         num_prompts=-5,
    #         avg_input_tokens=-100,
    #         avg_output_tokens=-200,
    #         model_used="openai/gpt-4o"
    #     )

    #     calculator = InferenceCalculator(price_fetcher, pattern)

    #     # Verify cost calculation with negative values
    #     # This should still calculate, though the result may not make practical sense
    #     cost = calculator.calculate_query_cost(
    #         pattern.model_used,
    #         pattern.avg_input_tokens,
    #         pattern.avg_output_tokens
    #     )
    #     expected_cost = -100 * 0.00001 + -200 * 0.00003
    #     assert cost == pytest.approx(expected_cost)

    #     # Verify daily cost with negative values
    #     daily_cost = calculator.daily_cost_per_user()
    #     assert daily_cost['total_cost'] == pytest.approx(expected_cost * -5)  # -5 prompts
    #     assert "openai/gpt-4o" in daily_cost['usage_breakdown']
