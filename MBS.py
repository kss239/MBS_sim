import random
import numpy as np
import itertools
from scipy.stats import norm
import math

class MBS:
    def __init__(self, discount_rate, correlation_matrix={
        'high': {'high': 1.0, 'medium': 0.4, 'low': 0.1},
        'medium': {'high': 0.4, 'medium': 1.0, 'low': 0.1},
        'low': {'high': 0.1, 'medium': 0.1, 'low': 1.0}
    }, risk_proportions={'high': 0.1, 'medium': 0.3, 'low': 0.6},
       mortgage_type_proportions={'fixed_rate': 0.3, 'variable_rate': 0.2, 'deferred_interest': 0.2,
                                  'balloon_payment': 0.2, 'negative_amortization': 0.1},
       default_probabilities={'high': 0.10, 'medium': 0.05, 'low': 0.02}):

        self.discount_rate = discount_rate
        self.correlation_matrix = correlation_matrix
        self.risk_proportions = risk_proportions
        self.mortgage_type_proportions = mortgage_type_proportions
        self.default_probabilities = default_probabilities
        self.mortgages = {mortgage_type: {risk_level: [] for risk_level in risk_proportions}
                          for mortgage_type in mortgage_type_proportions}
        self.cohort_stats = {mortgage_type: {risk_level: None for risk_level in risk_proportions}
                          for mortgage_type in mortgage_type_proportions}

    def generate_mortgages(self, num_mortgages, mean_principal=200000, std_dev_principal=50000):
        total_mortgages_created = 0

        for risk_level, risk_proportion in self.risk_proportions.items():
            num_risk_level_mortgages = int(num_mortgages * risk_proportion)
            risk_tuple = risk_level, self.default_probabilities[risk_level]

            for mortgage_type, type_proportion in self.mortgage_type_proportions.items():
                num_type_mortgages = int(num_risk_level_mortgages * type_proportion)

                # Adjust for rounding errors in the last iteration
                if risk_level == list(self.risk_proportions.keys())[-1] and mortgage_type == list(self.mortgage_type_proportions.keys())[-1]:
                    num_type_mortgages += num_mortgages - total_mortgages_created

                for _ in range(num_type_mortgages):
                    principal = np.random.normal(mean_principal, std_dev_principal)
                    principal = max(0, principal)  # Ensure principal is not negative

                    # Randomize year and rate variables
                    term_years = random.choice(range(5, 31, 5))
                    fixed_years = random.choice(range(5, term_years + 1, 5))
                    interest_only_years = random.choice(range(5, term_years + 1, 5))
                    annual_rate = round(random.uniform(0.04, 0.06), 4)  # Random rate between 4% and 6%
                    initial_rate = round(random.uniform(0.01, 0.03), 4)
                    subsequent_rate = round(random.uniform(0.06, 0.08), 4)

                    # Prepare schedule details
                    schedule_details = {
                        'principal': principal,
                        'annual_rate': annual_rate,
                        'term_years': term_years,
                        'fixed_years': fixed_years,
                        'interest_only_years': interest_only_years,
                        'balloon_payment': principal * 0.75,
                        'min_payment_ratio': 0.5,
                        'initial_rate': initial_rate,
                        'subsequent_rate': subsequent_rate
                    }

                    # Create Mortgage object
                    mortgage = Mortgage(current_period=2, risk_level=risk_tuple[0], estimated_default_probability=risk_tuple[1],
                                        amortization_type=mortgage_type, schedule_details=schedule_details)
                    mortgage.generate_schedule()
                    self.mortgages[mortgage_type][risk_level].append(mortgage)

                total_mortgages_created += num_type_mortgages


    def expected_cohort_cashflows(self, key_levels, correlation_coefficient):
        amortization_type = key_levels['amortization_type']
        risk_level = key_levels['risk_level']
        cohort_mortgages = self.mortgages[amortization_type][risk_level]

        cashflows_per_period = {}
        variances_per_period = {}

        # Collect cashflows and variances for each mortgage
        for mortgage in cohort_mortgages:
            expected_cashflows, cashflow_variance = mortgage.expected_cashflows()

            for period, cashflow in enumerate(expected_cashflows, start=1):
                cashflows_per_period.setdefault(period, []).append(cashflow)
                variances_per_period.setdefault(period, []).append(cashflow_variance[period - 1])

        # Calculate expected cashflows and variance for each period
        expected_cashflows = []
        variance_per_period = []

        for period in cashflows_per_period:
            # Expected cashflows for the period
            total_cashflow = sum(cashflows_per_period[period])
            expected_cashflows.append(total_cashflow)

            # Sum of variances for the period, considering correlation
            period_variances = variances_per_period[period]
            total_variance = sum(period_variances)

            num_mortgages = len(cashflows_per_period[period])
            if num_mortgages > 1:
                for i in range(num_mortgages):
                    for j in range(i + 1, num_mortgages):
                        covariance = correlation_coefficient * np.sqrt(period_variances[i]) * np.sqrt(period_variances[j])
                        total_variance += 2 * covariance

            variance_per_period.append(total_variance)
        self.cohort_stats[amortization_type][risk_level]={'expected_cashflow':expected_cashflows,'cashflow_variance':variance_per_period}

    def calculate_cohort_present_value(self, cohort_key, discount_rates):
        """
        Calculates the present value of a specific cohort's expected cash flows per period.

        :param cohort_key: A dictionary with keys 'amortization_type' and 'risk_level' to identify the cohort.
        :param discount_rates: A list of discount rates corresponding to each period.
        :return: List of present values per period for the cohort's expected cash flows.
        """
        amortization_type = cohort_key['amortization_type']
        risk_level = cohort_key['risk_level']

        if self.cohort_stats[amortization_type][risk_level] is None:
            self.expected_cohort_cashflows(cohort_key, self.correlation_matrix[risk_level][risk_level])

        cohort_data = self.cohort_stats[amortization_type][risk_level]['expected_cashflow']
        present_values = [cashflow / ((1 + discount_rates[i]) ** (i + 1)) for i, cashflow in enumerate(cohort_data) if i < len(discount_rates)]

        return present_values

    def calculate_cohort_VaR(self, cohort_key, confidence_interval):
        if self.cohort_stats[cohort_key['amortization_type']][cohort_key['risk_level']] is None:
            self.expected_cohort_cashflows(cohort_key, self.correlation_matrix[cohort_key['risk_level']][cohort_key['risk_level']])

        cohort_data = self.cohort_stats[cohort_key['amortization_type']][cohort_key['risk_level']]
        var_list = []
        for i in range(len(cohort_data['expected_cashflow'])):
            mean_cashflow = cohort_data['expected_cashflow'][i]
            std_deviation = cohort_data['cashflow_variance'][i] ** 0.5
            VaR = norm.ppf(1 - confidence_interval, mean_cashflow, std_deviation)
            var_list.append(VaR)
        if 'Var' not in cohort_data:
            cohort_data['Var'] = {}
        cohort_data['Var'][confidence_interval] = var_list
        return var_list

    def calculate_cohort_CVaR(self, cohort_key, confidence_level):
        """
        Calculate the Conditional Value at Risk (CVaR) for a specific cohort at a given confidence level.

        :param cohort_key: A dictionary with keys 'amortization_type' and 'risk_level'.
        :param confidence_level: Confidence level for CVaR calculation.
        :return: List of CVaR values per period for the cohort.
        """
        amortization_type = cohort_key['amortization_type']
        risk_level = cohort_key['risk_level']
        cohort_mortgages = self.mortgages[amortization_type][risk_level]

        total_cvar_per_period = [0] * max(len(m.stats['expected_cashflow']) for m in cohort_mortgages)
        for mortgage in cohort_mortgages:
            cvar_list = mortgage.calculate_CVaR(confidence_level)
            for i in range(len(cvar_list)):
                total_cvar_per_period[i] += cvar_list[i]

        # Average the CVaR values across all mortgages in the cohort
        average_cvar_per_period = [total / len(cohort_mortgages) for total in total_cvar_per_period]
        return average_cvar_per_period

    def calculate_cohort_duration(self, cohort_key, discount_rates, duration_type='Macaulay'):
        """
        Calculates the Macaulay or Modified Duration per period for a specific cohort.

        :param cohort_key: A dictionary with keys 'amortization_type' and 'risk_level'.
        :param discount_rates: A list of discount rates corresponding to each period.
        :param duration_type: Type of duration to calculate ('Macaulay' or 'Modified').
        :return: List of durations per period for the cohort.
        """
        amortization_type = cohort_key['amortization_type']
        risk_level = cohort_key['risk_level']
        mortgages = self.mortgages[amortization_type][risk_level]

        durations_per_period = []
        for i in range(len(discount_rates)):
            total_pv = sum(m.calculate_present_value(discount_rates[:i+1]) for m in mortgages)
            total_duration = sum(m.calculate_duration(discount_rates[:i+1], duration_type) * m.calculate_present_value(discount_rates[:i+1]) for m in mortgages)
            duration = total_duration / total_pv if total_pv != 0 else 0
            durations_per_period.append(duration)

        return durations_per_period

    def calculate_cohort_convexity(self, cohort_key, discount_rates):
        """
        Calculates the convexity per period for a specific cohort.

        :param cohort_key: A dictionary with keys 'amortization_type' and 'risk_level'.
        :param discount_rates: A list of discount rates corresponding to each period.
        :return: List of convexity values per period for the cohort.
        """
        amortization_type = cohort_key['amortization_type']
        risk_level = cohort_key['risk_level']
        mortgages = self.mortgages[amortization_type][risk_level]

        convexities_per_period = []
        for i in range(len(discount_rates)):
            total_pv = sum(m.calculate_present_value(discount_rates[:i+1]) for m in mortgages)
            total_convexity = sum(m.calculate_convexity(discount_rates[:i+1]) * m.calculate_present_value(discount_rates[:i+1]) for m in mortgages)
            convexity = total_convexity / total_pv if total_pv != 0 else 0
            convexities_per_period.append(convexity)

        return convexities_per_period


    def pad_list(self, lst, length, pad_value=0):
        """Extend the list to the given length by padding it with the specified value."""
        return lst + [pad_value] * (length - len(lst))

    def expected_cashflows(self, cohort_key_level_list):
        max_num_periods = 0

        # Determine the maximum number of periods across cohorts
        for key in cohort_key_level_list:
            if self.cohort_stats[key['amortization_type']][key['risk_level']] is None:
                self.expected_cohort_cashflows(key, self.correlation_matrix[key['risk_level']][key['risk_level']])

            cohort_data = self.cohort_stats[key['amortization_type']][key['risk_level']]
            max_num_periods = max(max_num_periods, len(cohort_data['expected_cashflow']))

        total_expected_cashflows = [0] * max_num_periods
        total_variance = [0] * max_num_periods

        # Sum expected cashflows and variances, padding if necessary
        for key in cohort_key_level_list:
            cohort_data = self.cohort_stats[key['amortization_type']][key['risk_level']]
            cohort_cashflows = self.pad_list(cohort_data['expected_cashflow'], max_num_periods)
            cohort_variances = self.pad_list(cohort_data['cashflow_variance'], max_num_periods)

            total_expected_cashflows = [sum(x) for x in zip(total_expected_cashflows, cohort_cashflows)]
            total_variance = [sum(x) for x in zip(total_variance, cohort_variances)]

        # Adjust variances for correlations
        combinations = list(itertools.combinations(cohort_key_level_list, 2))
        for (key1, key2) in combinations:
            corr = self.correlation_matrix[key1['risk_level']][key2['risk_level']]
            cohort1_data = self.cohort_stats[key1['amortization_type']][key1['risk_level']]
            cohort2_data = self.cohort_stats[key2['amortization_type']][key2['risk_level']]

            cohort1_variances = self.pad_list(cohort1_data['cashflow_variance'], max_num_periods)
            cohort2_variances = self.pad_list(cohort2_data['cashflow_variance'], max_num_periods)

            for period in range(max_num_periods):
                covariance = 2 * corr * (cohort1_variances[period] ** 0.5) * (cohort2_variances[period] ** 0.5)
                total_variance[period] += covariance

        return total_expected_cashflows, total_variance

    def calculate_combined_cohorts_present_value(self, cohort_keys, discount_rates):
        """
        Calculates the present value of combined cohorts' expected cash flows.

        :param cohort_keys: A list of dictionaries with keys 'amortization_type' and 'risk_level' for each cohort.
        :param discount_rates: A list of discount rates corresponding to each period.
        :return: Combined present value of all specified cohorts' expected cash flows.
        """
        total_present_value = 0
        for cohort_key in cohort_keys:
            cohort_present_value = self.calculate_cohort_present_value(cohort_key, discount_rates)
            total_present_value += cohort_present_value

        return total_present_value

    def calculate_combined_cohorts_VaR(self, cohort_keys, confidence_interval):
        max_periods = max(len(self.cohort_stats[ck['amortization_type']][ck['risk_level']]['expected_cashflow']) for ck in cohort_keys)
        combined_VaR = [0] * max_periods

        for period in range(max_periods):
            combined_mean = sum(self.cohort_stats[ck['amortization_type']][ck['risk_level']]['expected_cashflow'][period] if period < len(self.cohort_stats[ck['amortization_type']][ck['risk_level']]['expected_cashflow']) else 0 for ck in cohort_keys)
            combined_variance = 0

            # Adding variances and covariances
            for i, ck1 in enumerate(cohort_keys):
                for j, ck2 in enumerate(cohort_keys):
                    variance1 = self.cohort_stats[ck1['amortization_type']][ck1['risk_level']]['cashflow_variance'][period] if period < len(self.cohort_stats[ck1['amortization_type']][ck1['risk_level']]['cashflow_variance']) else 0
                    variance2 = self.cohort_stats[ck2['amortization_type']][ck2['risk_level']]['cashflow_variance'][period] if period < len(self.cohort_stats[ck2['amortization_type']][ck2['risk_level']]['cashflow_variance']) else 0

                    if i == j:  # Variance
                        combined_variance += variance1
                    else:  # Covariance
                        corr = self.correlation_matrix[ck1['risk_level']][ck2['risk_level']]
                        std_dev1 = math.sqrt(variance1)
                        std_dev2 = math.sqrt(variance2)
                        combined_variance += corr * std_dev1 * std_dev2

            combined_std_dev = combined_variance ** 0.5
            combined_VaR[period] = norm.ppf(1 - confidence_interval, combined_mean, combined_std_dev)

        return combined_VaR

    def calculate_combined_cohorts_CVaR_with_correlation(self, cohort_keys, confidence_level):
        """
        Calculate the combined Conditional Value at Risk (CVaR) for multiple cohorts considering correlations,
        at a given confidence level.

        :param cohort_keys: A list of dictionaries with keys 'amortization_type' and 'risk_level'.
        :param confidence_level: Confidence level for CVaR calculation.
        :return: List of combined CVaR values per period considering correlations.
        """
        max_periods = max(len(self.cohort_stats[ck['amortization_type']][ck['risk_level']]['expected_cashflow']) for ck in cohort_keys)
        combined_CVaR = [0] * max_periods

        for period in range(max_periods):
            combined_mean = sum(self.cohort_stats[ck['amortization_type']][ck['risk_level']]['expected_cashflow'][period] if period < len(self.cohort_stats[ck['amortization_type']][ck['risk_level']]['expected_cashflow']) else 0 for ck in cohort_keys)
            combined_variance = 0

            # Adding variances and covariances
            for i, ck1 in enumerate(cohort_keys):
                for j, ck2 in enumerate(cohort_keys):
                    variance1 = self.cohort_stats[ck1['amortization_type']][ck1['risk_level']]['cashflow_variance'][period] if period < len(self.cohort_stats[ck1['amortization_type']][ck1['risk_level']]['cashflow_variance']) else 0
                    variance2 = self.cohort_stats[ck2['amortization_type']][ck2['risk_level']]['cashflow_variance'][period] if period < len(self.cohort_stats[ck2['amortization_type']][ck2['risk_level']]['cashflow_variance']) else 0

                    if i == j:  # Variance
                        combined_variance += variance1
                    else:  # Covariance
                        corr = self.correlation_matrix[ck1['risk_level']][ck2['risk_level']]
                        std_dev1 = math.sqrt(variance1)
                        std_dev2 = math.sqrt(variance2)
                        combined_variance += corr * std_dev1 * std_dev2

            combined_std_dev = combined_variance ** 0.5
            alpha = 1 - confidence_level
            combined_cvar = (alpha ** -1) * norm.pdf(norm.ppf(alpha)) * combined_std_dev - combined_mean
            combined_CVaR[period] = combined_cvar

        return combined_CVaR

    def calculate_combined_cohorts_duration_with_correlation(self, cohort_keys, discount_rates, duration_type='Macaulay'):
        """
        Calculates the combined Macaulay or Modified Duration for multiple cohorts considering correlations,
        computed per period.

        :param cohort_keys: A list of dictionaries with keys 'amortization_type' and 'risk_level'.
        :param discount_rates: A list of discount rates corresponding to each period.
        :param duration_type: Type of duration to calculate ('Macaulay' or 'Modified').
        :return: List of combined durations per period for the specified cohorts considering correlations.
        """
        durations_per_period = []

        for period in range(len(discount_rates)):
            # Calculating individual durations and present values up to the current period
            cohort_durations = [self.calculate_cohort_duration(cohort_key, discount_rates[:period + 1], duration_type)[-1] for cohort_key in cohort_keys]
            cohort_present_values = [self.calculate_cohort_present_value(cohort_key, discount_rates[:period + 1])[-1] for cohort_key in cohort_keys]

            # Total present value of all cohorts up to the current period
            total_pv = sum(cohort_present_values)

            # Adjusting for correlations
            combined_duration = 0
            for i, key_i in enumerate(cohort_keys):
                for j, key_j in enumerate(cohort_keys):
                    correlation = self.correlation_matrix[key_i['risk_level']][key_j['risk_level']]
                    combined_duration += (cohort_durations[i] * cohort_present_values[i] * cohort_durations[j] * cohort_present_values[j] * correlation)

            combined_duration /= (total_pv ** 2) if total_pv != 0 else 0
            durations_per_period.append(combined_duration)

        return durations_per_period

    def calculate_combined_cohorts_convexity_with_correlation(self, cohort_keys, discount_rates):
        """
        Calculates the combined convexity for multiple cohorts considering correlations,
        computed per period.

        :param cohort_keys: A list of dictionaries with keys 'amortization_type' and 'risk_level'.
        :param discount_rates: A list of discount rates corresponding to each period.
        :return: List of combined convexity values per period for the specified cohorts considering correlations.
        """
        convexities_per_period = []

        for period in range(len(discount_rates)):
            # Calculating individual convexities and present values up to the current period
            cohort_convexities = [self.calculate_cohort_convexity(cohort_key, discount_rates[:period + 1])[-1] for cohort_key in cohort_keys]
            cohort_present_values = [self.calculate_cohort_present_value(cohort_key, discount_rates[:period + 1])[-1] for cohort_key in cohort_keys]

            # Total present value of all cohorts up to the current period
            total_pv = sum(cohort_present_values)

            # Adjusting for correlations
            combined_convexity = 0
            for i, key_i in enumerate(cohort_keys):
                for j, key_j in enumerate(cohort_keys):
                    correlation = self.correlation_matrix[key_i['risk_level']][key_j['risk_level']]
                    combined_convexity += (cohort_convexities[i] * cohort_present_values[i] * cohort_convexities[j] * cohort_present_values[j] * correlation)

            combined_convexity /= (total_pv ** 2) if total_pv != 0 else 0
            convexities_per_period.append(combined_convexity)

        return convexities_per_period

    def generate_fixed_rate_mortgages(self, num_mortgages, mean_principal=200000, std_dev_principal=50000,
                                      term_years_range=(5, 30), annual_rate_range=(0.04, 0.06),
                                      risk_proportions={'high': 0.1, 'medium': 0.3, 'low': 0.6},
                                      default_probabilities={'high': 0.10, 'medium': 0.05, 'low': 0.02}):
        """
        Generates a specified number of fixed rate mortgages.

        :param num_mortgages: Total number of fixed rate mortgages to generate.
        :param mean_principal: Mean value of the mortgage principal.
        :param std_dev_principal: Standard deviation of the mortgage principal.
        :param term_years_range: Range of term years as a tuple (min, max).
        :param annual_rate_range: Range of annual interest rates as a tuple (min, max).
        :param risk_proportions: Proportions of each risk category.
        :param default_probabilities: Default probabilities for each risk category.
        """
        mortgage_type = 'fixed_rate'
        total_mortgages_created = 0

        for risk_level, risk_proportion in risk_proportions.items():
            num_risk_level_mortgages = int(num_mortgages * risk_proportion)
            risk_tuple = risk_level, default_probabilities[risk_level]

            for _ in range(num_risk_level_mortgages):
                principal = np.random.normal(mean_principal, std_dev_principal)
                principal = max(0, principal)  # Ensure principal is not negative

                # Randomize year and rate variables within specified ranges
                term_years = random.choice(range(term_years_range[0], term_years_range[1] + 1, 5))
                annual_rate = round(random.uniform(*annual_rate_range), 4)

                # Prepare schedule details specific to fixed rate mortgages
                schedule_details = {
                    'principal': principal,
                    'annual_rate': annual_rate,
                    'term_years': term_years
                }

                # Create Mortgage object
                mortgage = Mortgage(current_period=2, risk_level=risk_tuple[0],
                                    estimated_default_probability=risk_tuple[1],
                                    amortization_type=mortgage_type, schedule_details=schedule_details)
                mortgage.generate_schedule()
                self.mortgages[mortgage_type][risk_level].append(mortgage)

                total_mortgages_created += 1

    def generate_variable_rate_mortgages(self, num_mortgages, mean_principal=200000, std_dev_principal=50000,
                                         term_years_range=(5, 30), initial_rate_range=(0.01, 0.03),
                                         subsequent_rate_range=(0.06, 0.08), fixed_years_range=(5, 30),
                                         risk_proportions={'high': 0.1, 'medium': 0.3, 'low': 0.6},
                                         default_probabilities={'high': 0.10, 'medium': 0.05, 'low': 0.02}):
        """
        Generates a specified number of variable rate mortgages.

        :param num_mortgages: Total number of variable rate mortgages to generate.
        :param mean_principal: Mean value of the mortgage principal.
        :param std_dev_principal: Standard deviation of the mortgage principal.
        :param term_years_range: Range of term years as a tuple (min, max).
        :param initial_rate_range: Range of initial interest rates as a tuple (min, max).
        :param subsequent_rate_range: Range of subsequent interest rates as a tuple (min, max).
        :param fixed_years_range: Range of fixed years in the beginning of the mortgage term as a tuple (min, max).
        :param risk_proportions: Proportions of each risk category.
        :param default_probabilities: Default probabilities for each risk category.
        """
        mortgage_type = 'variable_rate'
        total_mortgages_created = 0

        for risk_level, risk_proportion in risk_proportions.items():
            num_risk_level_mortgages = int(num_mortgages * risk_proportion)
            risk_tuple = risk_level, default_probabilities[risk_level]

            for _ in range(num_risk_level_mortgages):
                principal = np.random.normal(mean_principal, std_dev_principal)
                principal = max(0, principal)  # Ensure principal is not negative

                # Randomize year and rate variables within specified ranges
                term_years = random.choice(range(term_years_range[0], term_years_range[1] + 1, 5))
                fixed_years = random.choice(range(fixed_years_range[0], min(fixed_years_range[1], term_years) + 1, 5))
                initial_rate = round(random.uniform(*initial_rate_range), 4)
                subsequent_rate = round(random.uniform(*subsequent_rate_range), 4)

                # Prepare schedule details specific to variable rate mortgages
                schedule_details = {
                    'principal': principal,
                    'term_years': term_years,
                    'initial_rate': initial_rate,
                    'subsequent_rate': subsequent_rate,
                    'fixed_years': fixed_years
                }

                # Create Mortgage object
                mortgage = Mortgage(current_period=2, risk_level=risk_tuple[0],
                                    estimated_default_probability=risk_tuple[1],
                                    amortization_type=mortgage_type, schedule_details=schedule_details)
                mortgage.generate_schedule()
                self.mortgages[mortgage_type][risk_level].append(mortgage)

                total_mortgages_created += 1

    def generate_deferred_interest_mortgages(self, num_mortgages, mean_principal=200000, std_dev_principal=50000,
                                             term_years_range=(5, 30), annual_rate_range=(0.04, 0.06),
                                             interest_only_years_range=(1, 10),
                                             risk_proportions={'high': 0.1, 'medium': 0.3, 'low': 0.6},
                                             default_probabilities={'high': 0.10, 'medium': 0.05, 'low': 0.02}):
        """
        Generates a specified number of deferred interest mortgages.

        :param num_mortgages: Total number of deferred interest mortgages to generate.
        :param mean_principal: Mean value of the mortgage principal.
        :param std_dev_principal: Standard deviation of the mortgage principal.
        :param term_years_range: Range of term years as a tuple (min, max).
        :param annual_rate_range: Range of annual interest rates as a tuple (min, max).
        :param interest_only_years_range: Range of interest-only years as a tuple (min, max).
        :param risk_proportions: Proportions of each risk category.
        :param default_probabilities: Default probabilities for each risk category.
        """
        mortgage_type = 'deferred_interest'
        total_mortgages_created = 0

        for risk_level, risk_proportion in risk_proportions.items():
            num_risk_level_mortgages = int(num_mortgages * risk_proportion)
            risk_tuple = risk_level, default_probabilities[risk_level]

            for _ in range(num_risk_level_mortgages):
                principal = np.random.normal(mean_principal, std_dev_principal)
                principal = max(0, principal)  # Ensure principal is not negative

                # Randomize year and rate variables within specified ranges
                term_years = random.choice(range(term_years_range[0], term_years_range[1] + 1, 5))
                annual_rate = round(random.uniform(*annual_rate_range), 4)
                interest_only_years = random.choice(range(interest_only_years_range[0], interest_only_years_range[1] + 1))

                # Prepare schedule details specific to deferred interest mortgages
                schedule_details = {
                    'principal': principal,
                    'annual_rate': annual_rate,
                    'term_years': term_years,
                    'interest_only_years': interest_only_years
                }

                # Create Mortgage object
                mortgage = Mortgage(current_period=2, risk_level=risk_tuple[0],
                                    estimated_default_probability=risk_tuple[1],
                                    amortization_type=mortgage_type, schedule_details=schedule_details)
                mortgage.generate_schedule()
                self.mortgages[mortgage_type][risk_level].append(mortgage)
                total_mortgages_created += 1

    def generate_balloon_payment_mortgages(self, num_mortgages, mean_principal=200000, std_dev_principal=50000,
                                           term_years_range=(5, 30), annual_rate_range=(0.04, 0.06),
                                           balloon_payment_ratio_range=(0.5, 0.8),
                                           risk_proportions={'high': 0.1, 'medium': 0.3, 'low': 0.6},
                                           default_probabilities={'high': 0.10, 'medium': 0.05, 'low': 0.02}):
        """
        Generates a specified number of balloon payment mortgages.

        :param num_mortgages: Total number of balloon payment mortgages to generate.
        :param mean_principal: Mean value of the mortgage principal.
        :param std_dev_principal: Standard deviation of the mortgage principal.
        :param term_years_range: Range of term years as a tuple (min, max).
        :param annual_rate_range: Range of annual interest rates as a tuple (min, max).
        :param balloon_payment_ratio_range: Range of balloon payment ratios as a tuple (min, max).
        :param risk_proportions: Proportions of each risk category.
        :param default_probabilities: Default probabilities for each risk category.
        """
        mortgage_type = 'balloon_payment'
        total_mortgages_created = 0

        for risk_level, risk_proportion in risk_proportions.items():
            num_risk_level_mortgages = int(num_mortgages * risk_proportion)
            risk_tuple = risk_level, default_probabilities[risk_level]

            for _ in range(num_risk_level_mortgages):
                principal = np.random.normal(mean_principal, std_dev_principal)
                principal = max(0, principal)  # Ensure principal is not negative

                # Randomize year and rate variables within specified ranges
                term_years = random.choice(range(term_years_range[0], term_years_range[1] + 1, 5))
                annual_rate = round(random.uniform(*annual_rate_range), 4)
                balloon_payment_ratio = random.uniform(*balloon_payment_ratio_range)
                balloon_payment = principal * balloon_payment_ratio

                # Prepare schedule details specific to balloon payment mortgages
                schedule_details = {
                    'principal': principal,
                    'annual_rate': annual_rate,
                    'term_years': term_years,
                    'balloon_payment': balloon_payment
                }

                # Create Mortgage object
                mortgage = Mortgage(current_period=2, risk_level=risk_tuple[0],
                                    estimated_default_probability=risk_tuple[1],
                                    amortization_type=mortgage_type, schedule_details=schedule_details)
                mortgage.generate_schedule()
                self.mortgages[mortgage_type][risk_level].append(mortgage)

                total_mortgages_created += 1

    def generate_negative_amortization_mortgages(self, num_mortgages, mean_principal=200000, std_dev_principal=50000,
                                                 term_years_range=(5, 30), annual_rate_range=(0.01, 0.03),
                                                 min_payment_ratio_range=(0.4, 0.7),
                                                 risk_proportions={'high': 0.1, 'medium': 0.3, 'low': 0.6},
                                                 default_probabilities={'high': 0.10, 'medium': 0.05, 'low': 0.02}):
        """
        Generates a specified number of negative amortization mortgages.

        :param num_mortgages: Total number of negative amortization mortgages to generate.
        :param mean_principal: Mean value of the mortgage principal.
        :param std_dev_principal: Standard deviation of the mortgage principal.
        :param term_years_range: Range of term years as a tuple (min, max).
        :param annual_rate_range: Range of initial interest rates as a tuple (min, max).
        :param min_payment_ratio_range: Range of minimum payment ratios as a tuple (min, max).
        :param risk_proportions: Proportions of each risk category.
        :param default_probabilities: Default probabilities for each risk category.
        """
        mortgage_type = 'negative_amortization'
        total_mortgages_created = 0

        for risk_level, risk_proportion in risk_proportions.items():
            num_risk_level_mortgages = int(num_mortgages * risk_proportion)
            risk_tuple = risk_level, default_probabilities[risk_level]

            for _ in range(num_risk_level_mortgages):
                principal = np.random.normal(mean_principal, std_dev_principal)
                principal = max(0, principal)  # Ensure principal is not negative

                # Randomize year and rate variables within specified ranges
                term_years = random.choice(range(term_years_range[0], term_years_range[1] + 1, 5))
                annual_rate = round(random.uniform(*annual_rate_range), 4)
                min_payment_ratio = random.uniform(*min_payment_ratio_range)

                # Prepare schedule details specific to negative amortization mortgages
                schedule_details = {
                    'principal': principal,
                    'term_years': term_years,
                    'annual_rate': annual_rate,
                    'min_payment_ratio': min_payment_ratio
                }

                # Create Mortgage object
                mortgage = Mortgage(current_period=2, risk_level=risk_tuple[0],
                                    estimated_default_probability=risk_tuple[1],
                                    amortization_type=mortgage_type, schedule_details=schedule_details)
                mortgage.generate_schedule()
                self.mortgages[mortgage_type][risk_level].append(mortgage)

                total_mortgages_created += 1

