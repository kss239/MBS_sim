# Mortgage-Backed Securities (MBS) Analysis Package

This package offers a comprehensive suite of tools for the analysis and risk assessment of mortgage-backed securities (MBS). It includes classes for modeling different types of mortgages and their associated risks, as well as tools for aggregating and analyzing these mortgages within the context of mortgage-backed securities.

## Features

Mortgage Modeling: Supports various mortgage types including fixed rate, variable rate, deferred interest, balloon payment, and negative amortization mortgages.
Risk Assessment: Calculates Value at Risk (VaR), Conditional Value at Risk (CVaR), and expected cashflows for individual mortgages and cohorts of mortgages.
Cohort Analysis: Enables the analysis of groups of mortgages (cohorts) based on risk level and amortization type.
Cashflow Analysis: Estimates the expected cashflows from mortgage cohorts along with their variances.
Valuation Metrics: Computes present value, Macaulay and Modified Duration, and convexity for individual mortgages and cohorts.
Correlation Adjustment: Incorporates the impact of correlations between different risk levels in cohort analyses.
Customizable Mortgage Generation: Allows for the creation of custom mortgage pools with specified characteristics.

## Installation
```
Clone the repository:
git clone https://github.com/yourusername/mbs-analysis-package.git
```

Navigate to the repository directory:
```
cd mbs-analysis-package
```

##Usage
Initializing the MBS Instance
```
from MBS import MBS
mbs = MBS(discount_rate=0.05)
```

Generating Mortgages
```
mbs.generate_fixed_rate_mortgages(num_mortgages=50)
mbs.generate_variable_rate_mortgages(num_mortgages=30)
mbs.generate_deferred_interest_mortgages(num_mortgages=20)
mbs.generate_balloon_payment_mortgages(num_mortgages=10)
mbs.generate_negative_amortization_mortgages(num_mortgages=5)
```

Performing Cohort Analysis
```
fixed_rate_cohort_high = {'amortization_type': 'fixed_rate', 'risk_level': 'high'}
variable_rate_cohort_medium = {'amortization_type': 'variable_rate', 'risk_level': 'medium'}

mbs.expected_cohort_cashflows(fixed_rate_cohort_high, correlation_coefficient=0.8)
expected_cashflows_high_risk = mbs.cohort_stats[fixed_rate_cohort_high['amortization_type']][fixed_rate_cohort_high['risk_level']]['expected_cashflow']
cohort_VaR_high_risk = mbs.calculate_cohort_VaR(fixed_rate_cohort_high, confidence_interval=0.95)
```

Calculating Valuation Metrics
```
discount_rates = [0.05] * 30 * 12  # 30 years with a 5% discount rate
pv_variable_rate_medium = mbs.calculate_cohort_present_value(variable_rate_cohort_medium, discount_rates)
```

Combined Cohort Analysis
```
cohorts_for_analysis = [fixed_rate_cohort_high, variable_rate_cohort_medium]
combined_VaR = mbs.calculate_combined_cohorts_VaR(cohorts_for_analysis, confidence_interval=0.95)
```
