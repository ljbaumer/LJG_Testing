import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule
from src.constants.value_chain_market_segments import (
    BULL_CASE_MARKET_SEGMENTS,
    TODAY_MARKET_SEGMENTS,
)
from src.models.ValueChainModel import ValueChainModel


def build_minimal_schedule() -> CapexDepreciationSchedule:
    """Single year schedule for basic tests."""
    depreciation_accounting_schedule = pd.DataFrame({
        "chips": [120.0],
    }, index=[2025])
    useful_life = pd.Series({"chips": 2})
    return CapexDepreciationSchedule(
        depreciation_accounting_schedule=depreciation_accounting_schedule,
        useful_life_series=useful_life,
    )


def build_interpolation_schedule() -> CapexDepreciationSchedule:
    """Multi-year schedule for interpolation tests (3+ points)."""
    depreciation_accounting_schedule = pd.DataFrame({
        "chips": [100.0, 80.0, 60.0],  # Three years for meaningful interpolation
    }, index=[2025, 2026, 2027])
    useful_life = pd.Series({"chips": 3})
    return CapexDepreciationSchedule(
        depreciation_accounting_schedule=depreciation_accounting_schedule,
        useful_life_series=useful_life,
    )


def build_minimal_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort_name": ["Cohort A", "Cohort B"],
            "cohort_share": [0.10, 0.20],
            "arpu": [100.0, 10.0],
            "cost_to_service": [40.0, 2.0],
            "segment": ["test_segment", "test_segment"],
            "total_addressable_users": [1000, 1000],
        }
    )


def build_minimal_markups() -> pd.Series:
    return pd.Series(
        data=[1.5, 1.2, 1.1],
        index=["cloud", "model", "app"],
    )


def test_value_chain_model_tiny_scenario_math_holds():
    schedule = build_minimal_schedule()
    market_segments = build_minimal_segments()
    markups = build_minimal_markups()

    model = ValueChainModel(
        start_market_segments=market_segments,
        target_market_segments=market_segments,  # Same start/end for this test
        depreciation_schedule=schedule,
        base_markups=markups,
    )

    result = model.run_full_analysis()

    expected_timeline = pd.DataFrame(
        {
            "year": [2025, 2026],
            "depreciation": [60.0, 60.0],  # 120.0 CapEx spread over 2-year useful life
            "cloud_revenue": [90.0, 90.0],
            "model_revenue": [108.0, 108.0],
            "app_revenue": [118.8, 118.8],
            "total_profit": [58.8, 58.8],
        }
    )
    assert_frame_equal(
        result["timeline"].reset_index(drop=True),
        expected_timeline,
        atol=1e-9,
        check_like=False,
    )

    expected_breakdown = pd.DataFrame(
        {
            "cohort_name": ["Cohort A", "Cohort B"],
            "cohort_share": [0.10, 0.20],
            "arpu": [100.0, 10.0],
            "cost_to_service": [40.0, 2.0],
            "segment": ["test_segment", "test_segment"],
            "total_addressable_users": [1000, 1000],
            "users": [100, 200],
            "monthly_revenue": [100 * 100.0, 200 * 10.0],
            "monthly_cost": [100 * 40.0, 200 * 2.0],
            "annual_revenue": [100 * 100.0 * 12, 200 * 10.0 * 12],
            "annual_cost": [100 * 40.0 * 12, 200 * 2.0 * 12],
            "annual_profit": [72000.0, 19200.0],
        }
    )
    assert_frame_equal(
        result["cohort_breakdown"].reset_index(drop=True),
        expected_breakdown,
        atol=1e-9,
    )

    user_summary = result["user_economics_summary"]
    assert user_summary["monthly_user_revenue"] == pytest.approx(12000.0)
    assert user_summary["monthly_user_cost"] == pytest.approx(4400.0)
    assert user_summary["annual_user_revenue"] == pytest.approx(144000.0)
    assert user_summary["annual_user_cost"] == pytest.approx(52800.0)
    assert user_summary["annual_user_profit"] == pytest.approx(91200.0)
    assert user_summary["total_active_users"] == 300

    configuration = result["model_configuration"]
    assert configuration["num_segments"] == 1
    assert configuration["num_cohorts"] == 2
    assert configuration["base_markups"] == {
        "cloud": pytest.approx(1.5),
        "model": pytest.approx(1.2),
        "app": pytest.approx(1.1),
    }
    assert configuration["adjusted_markups"] == configuration["base_markups"]
    assert configuration["num_toggles_applied"] == 0

    capex_summary = result["capex_depreciation_summary"]
    assert capex_summary["total_capex_committed"] == pytest.approx(120.0)
    assert capex_summary["total_depreciation"] == pytest.approx(60.0)  # Only 2025 depreciation (investment period)
    assert capex_summary["period_start_year"] == 2025
    assert capex_summary["period_end_year"] == 2025  # Investment period only


def test_interpolate_market_segments_over_time():
    """Test linear interpolation of market segments over time."""
    # Create simple test segments as DataFrames
    start_segment = pd.DataFrame({
        'segment': ['test_segment', 'test_segment'],
        'total_addressable_users': [1000, 1000],
        'cohort_name': ['Cohort A', 'Cohort B'],
        'cohort_share': [0.10, 0.20],
        'arpu': [100.0, 50.0],
        'cost_to_service': [20.0, 10.0]
    })

    end_segment = pd.DataFrame({
        'segment': ['test_segment', 'test_segment'],
        'total_addressable_users': [1000, 1000],
        'cohort_name': ['Cohort A', 'Cohort B'],
        'cohort_share': [0.30, 0.40],  # Higher adoption
        'arpu': [100.0, 50.0],  # Same ARPU
        'cost_to_service': [20.0, 10.0]  # Same cost
    })

    result_df = ValueChainModel.interpolate_market_segments_over_time(
        start_segments=start_segment,
        end_segments=end_segment,
        start_year=2025,
        end_year=2027
    )

    # Verify structure
    assert len(result_df) == 6  # 2 cohorts × 3 years
    assert sorted(result_df['year'].unique()) == [2025, 2026, 2027]
    assert sorted(result_df['cohort_name'].unique()) == ['Cohort A', 'Cohort B']

    # Verify interpolation for Cohort A (0.10 → 0.30)
    cohort_a_data = result_df[result_df['cohort_name'] == 'Cohort A'].sort_values('year')
    expected_cohort_a_shares = [0.10, 0.20, 0.30]  # Linear interpolation
    assert cohort_a_data['cohort_share'].tolist() == pytest.approx(expected_cohort_a_shares)

    # Verify interpolation for Cohort B (0.20 → 0.40)
    cohort_b_data = result_df[result_df['cohort_name'] == 'Cohort B'].sort_values('year')
    expected_cohort_b_shares = [0.20, 0.30, 0.40]  # Linear interpolation
    assert cohort_b_data['cohort_share'].tolist() == pytest.approx(expected_cohort_b_shares)

    # Verify ARPU stays constant
    assert all(result_df[result_df['cohort_name'] == 'Cohort A']['arpu'] == 100.0)
    assert all(result_df[result_df['cohort_name'] == 'Cohort B']['arpu'] == 50.0)

    # Verify cost_to_service stays constant
    assert all(result_df[result_df['cohort_name'] == 'Cohort A']['cost_to_service'] == 20.0)
    assert all(result_df[result_df['cohort_name'] == 'Cohort B']['cost_to_service'] == 10.0)

    # Verify TAM stays constant
    assert all(result_df['total_addressable_users'] == 1000)


def test_interpolate_market_segments_validation():
    """Test validation in interpolate_market_segments_over_time."""
    # Test invalid years
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="start_year must be less than or equal to end_year"):
        ValueChainModel.interpolate_market_segments_over_time(empty_df, empty_df, 2030, 2025)

    # Test empty segments
    with pytest.raises(ValueError, match="Both start_segments and end_segments must be non-empty"):
        ValueChainModel.interpolate_market_segments_over_time(empty_df, empty_df, 2025, 2030)

    # Test mismatched segment names - should now handle by inserting missing cohorts with zero
    start_segment = pd.DataFrame({
        'segment': ['segment_1'],
        'total_addressable_users': [1000],
        'cohort_name': ['Cohort A'],
        'cohort_share': [0.10],
        'arpu': [100.0],
        'cost_to_service': [20.0]
    })

    end_segment = pd.DataFrame({
        'segment': ['segment_2'],  # Different name
        'total_addressable_users': [1000],
        'cohort_name': ['Cohort A'],
        'cohort_share': [0.30],
        'arpu': [100.0],
        'cost_to_service': [20.0]
    })

    # Should not raise error anymore - handles missing segments by inserting with zero cohort_share
    result = ValueChainModel.interpolate_market_segments_over_time(
        start_segment, end_segment, 2025, 2027
    )

    # Should have both segments in result
    assert set(result['segment'].unique()) == {'segment_1', 'segment_2'}

    # segment_1 should start at 0.10 and end at 0 (missing in end)
    seg1_data = result[result['segment'] == 'segment_1'].sort_values('year')
    assert seg1_data['cohort_share'].tolist() == pytest.approx([0.10, 0.05, 0.0])

    # segment_2 should start at 0 (missing in start) and end at 0.30
    seg2_data = result[result['segment'] == 'segment_2'].sort_values('year')
    assert seg2_data['cohort_share'].tolist() == pytest.approx([0.0, 0.15, 0.30])


def test_interpolate_real_market_segments():
    """Test interpolation with real market segment data."""
    result_df = ValueChainModel.interpolate_market_segments_over_time(
        start_segments=TODAY_MARKET_SEGMENTS,
        end_segments=BULL_CASE_MARKET_SEGMENTS,
        start_year=2025,
        end_year=2030
    )

    # Verify all expected segments are present
    expected_segments = {'us_canada_europe_consumers', 'apac_row_consumers', 'professional', 'programmer'}
    actual_segments = set(result_df['segment'].unique())
    assert expected_segments == actual_segments

    # Verify years
    assert sorted(result_df['year'].unique()) == list(range(2025, 2031))

    # Verify specific interpolation for programmer segment
    programmer_data = result_df[
        (result_df['segment'] == 'programmer') &
        (result_df['cohort_name'] == 'Developer Paid')
    ].sort_values('year')

    # Should interpolate from TODAY (0.75) to BULL_CASE (0.65)
    start_share = 0.75
    end_share = 0.65
    expected_shares = []
    for year in range(2025, 2031):
        t = (year - 2025) / (2030 - 2025)
        interpolated_share = start_share * (1 - t) + end_share * t
        expected_shares.append(interpolated_share)

    actual_shares = programmer_data['cohort_share'].tolist()
    assert actual_shares == pytest.approx(expected_shares, abs=1e-10)

    # Verify ARPU is interpolated between TODAY (20.0) and BULL_CASE (22.0)
    start_arpu = 20.0
    end_arpu = 22.0
    expected_arpus = []
    for year in range(2025, 2031):
        t = (year - 2025) / (2030 - 2025)
        interpolated_arpu = start_arpu * (1 - t) + end_arpu * t
        expected_arpus.append(interpolated_arpu)

    actual_arpus = programmer_data['arpu'].tolist()
    assert actual_arpus == pytest.approx(expected_arpus, abs=1e-10)


def test_value_chain_model_interpolated_timeline():
    """Test the instance method get_interpolated_timeline()."""
    schedule = build_interpolation_schedule()  # Use 3-year schedule
    markups = build_minimal_markups()

    # Create different start and target segments
    start_segments = pd.DataFrame({
        'segment': ['test_segment', 'test_segment'],
        'total_addressable_users': [1000, 1000],
        'cohort_name': ['Cohort A', 'Cohort B'],
        'cohort_share': [0.10, 0.20],
        'arpu': [100.0, 50.0],
        'cost_to_service': [20.0, 10.0]
    })

    target_segments = pd.DataFrame({
        'segment': ['test_segment', 'test_segment'],
        'total_addressable_users': [1000, 1000],
        'cohort_name': ['Cohort A', 'Cohort B'],
        'cohort_share': [0.30, 0.40],  # Higher adoption
        'arpu': [100.0, 50.0],
        'cost_to_service': [20.0, 10.0]
    })

    model = ValueChainModel(
        start_market_segments=start_segments,
        target_market_segments=target_segments,
        depreciation_schedule=schedule,
        base_markups=markups
    )

    # Test the static interpolation method directly (wrapper method removed)
    result = model.interpolate_market_segments_over_time(
        start_segments=start_segments,
        end_segments=target_segments,
        start_year=model.start_year,
        end_year=model.end_year
    )

    # Should use the model's timeline (2025-2027 from interpolation schedule)
    expected_years = [2025, 2026, 2027]
    assert sorted(result['year'].unique()) == expected_years

    # Should interpolate between start and target cohort shares
    cohort_a_data = result[result['cohort_name'] == 'Cohort A'].sort_values('year')
    assert cohort_a_data['cohort_share'].tolist() == pytest.approx([0.10, 0.20, 0.30])  # Linear interpolation

    cohort_b_data = result[result['cohort_name'] == 'Cohort B'].sort_values('year')
    assert cohort_b_data['cohort_share'].tolist() == pytest.approx([0.20, 0.30, 0.40])  # Linear interpolation


def test_interpolated_revenue_timeline_structure():
    """Test that interpolated revenue timeline has correct structure and shows growth."""
    from src.constants.value_chain_market_segments import (
        TODAY_MARKET_SEGMENTS,
        TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
    )

    schedule = build_interpolation_schedule()  # 3-year schedule for meaningful interpolation
    markups = build_minimal_markups()

    model = ValueChainModel(
        start_market_segments=TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,  # Display segments
        target_market_segments=TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,  # Target for interpolation
        depreciation_schedule=schedule,
        base_markups=markups
    )

    # Test via run_full_analysis - interpolated revenue timeline moved to display layer
    full_results = model.run_full_analysis()
    assert "interpolated_revenue_timeline" not in full_results  # Moved to display layer

    # Test interpolation separately using the static method

    interpolated_segments = model.interpolate_market_segments_over_time(
        start_segments=TODAY_MARKET_SEGMENTS,
        end_segments=TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
        start_year=model.start_year,
        end_year=model.end_year
    )
    revenue_timeline = model._calculate_revenue_from_interpolated_segments(interpolated_segments)

    # Verify structure
    expected_columns = {'year', 'segment', 'annual_revenue', 'cohort_share'}
    assert set(revenue_timeline.columns) == expected_columns

    # Should have 3 years and multiple segments (from TODAY_MARKET_SEGMENTS)
    years = sorted(revenue_timeline['year'].unique())
    assert years == [2025, 2026, 2027]

    # Should have real market segments (interpolation starts from TODAY_MARKET_SEGMENTS)
    segments = set(revenue_timeline['segment'].unique())
    expected_segments = set(TODAY_MARKET_SEGMENTS['segment'].unique())
    # Target segments should also be included
    expected_segments.update(TOTAL_ADOPTION_CASE_MARKET_SEGMENTS['segment'].unique())

    # All expected segments should be present (though some might have 0 revenue)
    assert expected_segments.issubset(segments) or segments.issubset(expected_segments)

    # Revenue should generally increase over time (TODAY → TOTAL_ADOPTION progression)
    total_revenue_by_year = {}
    for year in years:
        year_data = revenue_timeline[revenue_timeline['year'] == year]
        total_revenue_by_year[year] = year_data['annual_revenue'].sum()

    # Should show growth from TODAY baseline to TOTAL_ADOPTION target
    assert total_revenue_by_year[2025] < total_revenue_by_year[2027]


def test_interpolation_reaches_target_revenue():
    """Test that interpolated revenue at end year matches target scenario revenue exactly.

    This test addresses the bug where interpolation chart showed ~$200B at 2031
    but pie chart showed ~$520B for Total Adoption Case. The issue was that
    interpolation only interpolated cohort_share but kept TAM, ARPU, and cost_to_service
    constant from start segments.
    """
    from src.constants.value_chain_market_segments import (
        TODAY_MARKET_SEGMENTS,
        TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
    )

    # Create schedule that includes 2031 as end year
    years = list(range(2025, 2032))  # 2025 to 2031
    depreciation_accounting_schedule = pd.DataFrame({
        "chips": [100.0] * len(years),
    }, index=years)
    useful_life = pd.Series({"chips": 5})
    schedule = CapexDepreciationSchedule(
        depreciation_accounting_schedule=depreciation_accounting_schedule,
        useful_life_series=useful_life,
    )

    markups = build_minimal_markups()

    # Create model with Total Adoption Case as target
    model = ValueChainModel(
        start_market_segments=TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
        target_market_segments=TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
        depreciation_schedule=schedule,
        base_markups=markups
    )

    # Calculate target revenue (pie chart value)
    cohort_breakdown = model._calculate_cohort_distribution_and_economics()
    target_revenue = cohort_breakdown["annual_revenue"].sum()

    # Interpolate from TODAY to Total Adoption Case
    interpolated_segments = model.interpolate_market_segments_over_time(
        start_segments=TODAY_MARKET_SEGMENTS,
        end_segments=TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
        start_year=2025,
        end_year=2031
    )

    # Calculate interpolated revenue at end year (2031)
    revenue_timeline = model._calculate_revenue_from_interpolated_segments(interpolated_segments)
    end_year_revenue = revenue_timeline[revenue_timeline['year'] == 2031]['annual_revenue'].sum()

    # Should match exactly (interpolation reaches target at end year)
    assert end_year_revenue == pytest.approx(target_revenue, abs=1e-10)
