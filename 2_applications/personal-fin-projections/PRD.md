# Personal Finance Projections App - PRD

## Overview

A flexible, auditable financial projection tool that models the impact of earnings, savings, and spending decisions on net worth over 3-15 year horizons. Built with Python + Streamlit for interactive exploration and scenario comparison.

## Goals

- **Flexibility**: Adjust any variable at any level and immediately see downstream effects
- **Auditability**: Clear calculation logic, traceable from inputs to outputs
- **Scenario Analysis**: Save, load, and compare multiple input configurations
- **Time Value Awareness**: Toggle between nominal and present-value (discounted) figures

---

## Core Features

### 1. Income Module
| Input | Description |
|-------|-------------|
| Starting Total Compensation | Base annual income (salary + bonus + equity) |
| Annual Growth Rate | % increase in total comp per year |
| Tax Rate (effective) | Combined federal + state effective rate |

**Outputs**: Gross income, taxes, net income per year over projection period

### 2. Expense Module
| Category | Examples |
|----------|----------|
| Housing | Rent/mortgage, property tax, insurance, maintenance |
| Transportation | Car payment, insurance, gas, maintenance |
| Living | Food, utilities, subscriptions, healthcare |
| Discretionary | Travel, entertainment, shopping |
| Other | Miscellaneous, gifts, unexpected |

- Annual figures with category-level granularity
- Optional: growth rate per category (inflation adjustment)

**Outputs**: Total annual expenses, expense breakdown, expense growth over time

### 3. Savings Module

**Savings Rate Calculations**:
- Pre-tax savings rate = Pre-tax contributions / Gross Income
- Post-tax savings rate = Post-tax savings / Net Income (after taxes and expenses)

**Savings Vehicles**:
| Account | Annual Limit | Tax Treatment |
|---------|--------------|---------------|
| 401(k) | $23,500 (2025) | Pre-tax contributions, tax-deferred growth |
| IRA | $7,000 (2025) | Pre-tax or Roth, tax-advantaged growth |
| Brokerage | Unlimited | Post-tax contributions, taxable gains |

**Inputs per vehicle**:
- Contribution amount or % of income
- Expected annual return rate
- Employer match (for 401k)

### 4. Net Worth Projection

**Components**:
- 401(k) balance over time
- IRA balance over time
- Brokerage balance over time
- Total net worth = sum of all accounts

**Projection Logic**:
```
For each year t:
  income[t] = income[t-1] * (1 + growth_rate)
  taxes[t] = income[t] * tax_rate
  net_income[t] = income[t] - taxes[t]

  401k_contrib[t] = min(contribution_input, annual_limit)
  ira_contrib[t] = min(contribution_input, annual_limit)

  expenses[t] = sum(category_expenses) * (1 + inflation)^t
  remaining[t] = net_income[t] - 401k_contrib[t] - ira_contrib[t] - expenses[t]
  brokerage_contrib[t] = max(0, remaining[t])

  401k_balance[t] = 401k_balance[t-1] * (1 + return_rate) + 401k_contrib[t]
  ira_balance[t] = ira_balance[t-1] * (1 + return_rate) + ira_contrib[t]
  brokerage_balance[t] = brokerage_balance[t-1] * (1 + return_rate) + brokerage_contrib[t]

  net_worth[t] = 401k_balance[t] + ira_balance[t] + brokerage_balance[t]
```

### 5. Present Value Toggle

- Discount rate input (e.g., 3-5% for inflation or opportunity cost)
- Toggle to show all figures in:
  - **Nominal**: Raw projected values
  - **Present Value**: Discounted back to today's dollars

```
PV = FV / (1 + discount_rate)^t
```

### 6. Scenario / Case Management

- **Save Case**: Store current inputs as named configuration (e.g., "Base Case", "Aggressive Savings", "Early Retirement")
- **Load Case**: Restore inputs from saved configuration
- **Compare Cases**: Side-by-side view of 2-3 scenarios
- **Storage**: Local JSON file (`scenarios.json`)

```json
{
  "Base Case": {
    "income": {"starting": 200000, "growth_rate": 0.05},
    "expenses": {"housing": 36000, "transportation": 12000, ...},
    "savings": {"401k_contrib": 23000, "ira_contrib": 7000, ...},
    "assumptions": {"return_rate": 0.07, "discount_rate": 0.03}
  }
}
```

### 7. Visualizations

| Chart | Purpose |
|-------|---------|
| Net Worth Over Time | Line chart showing total net worth trajectory |
| Account Breakdown | Stacked area chart (401k, IRA, Brokerage) |
| Income vs Expenses | Bar chart comparing annual income/expenses |
| Savings Rate | Line chart of pre-tax and post-tax savings rates |
| Scenario Comparison | Multi-line chart overlaying different cases |
| Sensitivity Tornado | Show impact of +/- 10% on key variables |

---

## UI Structure (Streamlit)

### Sidebar
- Projection period slider (3-15 years)
- Nominal / Present Value toggle
- Case selector dropdown
- Save / Load buttons

### Main Tabs
1. **Inputs**: Income, Expenses, Savings configuration
2. **Projections**: Year-by-year table + charts
3. **Scenarios**: Compare saved cases side-by-side
4. **Sensitivity**: Adjust one variable, see impact on net worth

---

## Technical Architecture

```
personal-fin-projections/
├── app.py                 # Streamlit entry point
├── models/
│   ├── income.py          # Income projection logic
│   ├── expenses.py        # Expense tracking and projection
│   ├── savings.py         # Savings vehicle calculations
│   └── projection.py      # Net worth aggregation
├── utils/
│   ├── present_value.py   # Discounting utilities
│   └── scenarios.py       # Save/load case management
├── data/
│   └── scenarios.json     # Saved cases (gitignored)
├── requirements.txt
└── PRD.md
```

---

## Data Model

```python
@dataclass
class Scenario:
    name: str
    income: IncomeInputs
    expenses: ExpenseInputs
    savings: SavingsInputs
    assumptions: Assumptions

@dataclass
class IncomeInputs:
    starting_total_comp: float
    annual_growth_rate: float
    effective_tax_rate: float

@dataclass
class ExpenseInputs:
    housing: float
    transportation: float
    living: float
    discretionary: float
    other: float
    inflation_rate: float

@dataclass
class SavingsInputs:
    contrib_401k: float
    contrib_ira: float
    employer_match_pct: float
    return_rate_401k: float
    return_rate_ira: float
    return_rate_brokerage: float

@dataclass
class Assumptions:
    projection_years: int
    discount_rate: float
```

---

## MVP Scope

**Phase 1 - Core Engine**:
- [ ] Income projection with growth
- [ ] Expense categories with inflation
- [ ] 401k/IRA/Brokerage calculations
- [ ] Net worth projection table

**Phase 2 - Visualization**:
- [ ] Net worth line chart
- [ ] Account breakdown stacked chart
- [ ] Income vs expenses bar chart

**Phase 3 - Scenarios**:
- [ ] Save/load cases to JSON
- [ ] Case comparison view

**Phase 4 - Advanced**:
- [ ] Present value toggle
- [ ] Sensitivity analysis
- [ ] Tornado chart

---

## Success Criteria

1. Can input income, expenses, savings assumptions and see 15-year projection
2. Can toggle between nominal and present value
3. Can save 3+ scenarios and compare them side-by-side
4. Can adjust any input and immediately see updated projections
5. Calculations are transparent and auditable (show formulas or intermediate values)
