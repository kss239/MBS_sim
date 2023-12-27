# Mortgage-Backed Securities (MBS) Analysis Package

This package offers a comprehensive suite of tools for the analysis and risk assessment of mortgage-backed securities (MBS). It includes classes for modeling different types of mortgages and their associated risks, as well as tools for aggregating and analyzing these mortgages within the context of mortgage-backed securities.

## Features

Mortgage Modeling: Supports various mortgage types, including fixed-rate, variable-rate, deferred interest, balloon payment, and negative amortization mortgages.
Risk Assessment Metrics:
Value at Risk (VaR): Measures the potential loss in value of a mortgage portfolio over a defined period for a given confidence interval.
Conditional Value at Risk (CVaR): Estimates the expected loss exceeding the VaR, giving insight into the tail risk of the portfolio.
Duration Analysis: Includes both Macaulay and Modified Duration, useful for assessing the sensitivity of the mortgage's price to interest rate changes.
Convexity: Measures the curvature of how the duration of a mortgage changes as the interest rate changes, providing a measure of the bond's price volatility.
Cohort Analysis: Enables grouping of mortgages (cohorts) based on risk level and amortization type for collective analysis.
Cashflow Analysis: Estimates expected cashflows and their variances for mortgage cohorts.
Valuation Metrics: Includes computations of present value, duration, and convexity for both individual mortgages and cohorts.
Correlation Adjustments: Incorporates correlations between different risk levels in cohort analyses to provide more accurate risk assessments.
Customizable Mortgage Generation: Facilitates the creation of custom mortgage pools with specified risk and return characteristics.

## Installation
Clone the repository:
```
git clone https://github.com/kss239/MBS_sim.git
```

Navigate to the repository directory:
```
cd MBS_sim
```

## Usage
```
# Initializing the MBS Instance
from MBS import MBS

# Initialize the MBS instance with a specified discount rate
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
#Define cohort keys for analysis
fixed_rate_cohort_high = {'amortization_type': 'fixed_rate', 'risk_level': 'high'}
variable_rate_cohort_medium = {'amortization_type': 'variable_rate', 'risk_level': 'medium'}

#Calculate and display expected cashflows, VaR, and CVaR for cohorts
mbs.expected_cohort_cashflows(fixed_rate_cohort_high, correlation_coefficient=0.8)
expected_cashflows_high_risk = mbs.cohort_stats[fixed_rate_cohort_high['amortization_type']][fixed_rate_cohort_high['risk_level']]['expected_cashflow']
cohort_VaR_high_risk = mbs.calculate_cohort_VaR(fixed_rate_cohort_high, confidence_interval=0.95)
```

Calculating Valuation Metrics
```
# Calculate Present Value, Duration, VaR, CVaR, and/or Convexity for cohorts
discount_rates = [0.05] * 30 * 12  # 30 years with a 5% discount rate
pv_variable_rate_medium = mbs.calculate_cohort_present_value(variable_rate_cohort_medium, discount_rates)
```

Combined Cohort Analysis
```
# Perform combined analysis for specified cohorts, including combined VaR, CVaR, Duration, and Convexity
cohorts_for_analysis = [fixed_rate_cohort_high, variable_rate_cohort_medium]
combined_VaR = mbs.calculate_combined_cohorts_VaR(cohorts_for_analysis, confidence_interval=0.95)
```
