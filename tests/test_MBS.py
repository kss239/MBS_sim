import unittest
from MBS_sim import MBS  # Replace with the actual module name
import numpy as np

class TestMBS(unittest.TestCase):

    def setUp(self):
        # Setup the basic MBS object with default parameters or specified settings as needed
        self.mbs = MBS(discount_rate=0.04)
        self.amortization_types = ['fixed_rate', 'variable_rate', 'deferred_interest', 'balloon_payment', 'negative_amortization']

    def test_generate_mortgages(self):
        num_mortgages = 20  # Reduced number for quicker testing
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                total_mortgages = sum(len(self.mbs.mortgages[amortization_type][rl]) for rl in self.mbs.risk_proportions)
                self.assertEqual(total_mortgages, num_mortgages)

    def test_expected_cohort_cashflows(self):
        num_mortgages = 20
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                key_levels = {'amortization_type': amortization_type, 'risk_level': 'medium'}
                self.mbs.expected_cohort_cashflows(key_levels, 0.4)
                self.assertIsNotNone(self.mbs.cohort_stats[amortization_type]['medium'])

    def test_calculate_cohort_present_value(self):
        num_mortgages = 20
        discount_rates = [0.04 for _ in range(30 * 12)]
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                cohort_key = {'amortization_type': amortization_type, 'risk_level': 'medium'}
                present_values = self.mbs.calculate_cohort_present_value(cohort_key, discount_rates)
                self.assertTrue(all(pv > 0 for pv in present_values))

    def test_calculate_cohort_VaR(self):
        num_mortgages = 20
        confidence_interval = 0.95
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                cohort_key = {'amortization_type': amortization_type, 'risk_level': 'medium'}
                var_list = self.mbs.calculate_cohort_VaR(cohort_key, confidence_interval)
                self.assertEqual(len(var_list), len(var_list))

    def test_calculate_cohort_CVaR(self):
        num_mortgages = 20
        confidence_level = 0.95
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                cohort_key = {'amortization_type': amortization_type, 'risk_level': 'medium'}
                cvar_list = self.mbs.calculate_cohort_CVaR(cohort_key, confidence_level)
                self.assertEqual(len(cvar_list), len(cvar_list))

    def test_calculate_cohort_duration(self):
        num_mortgages = 20
        discount_rates = [0.04 for _ in range(30 * 12)]
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                cohort_key = {'amortization_type': amortization_type, 'risk_level': 'medium'}
                durations = self.mbs.calculate_cohort_duration(cohort_key, discount_rates)
                self.assertTrue(all(d > 0 for d in durations))

    def test_calculate_cohort_convexity(self):
        num_mortgages = 20
        discount_rates = [0.04 for _ in range(30 * 12)]
        for amortization_type in self.amortization_types:
            with self.subTest(amortization_type=amortization_type):
                self.mbs.generate_mortgages(num_mortgages, mortgage_type_proportions={amortization_type: 1})
                cohort_key = {'amortization_type': amortization_type, 'risk_level': 'medium'}
                convexities = self.mbs.calculate_cohort_convexity(cohort_key, discount_rates)
                self.assertTrue(all(c > 0 for c in convexities))

    def test_calculate_combined_cohorts_present_value(self):
        self.mbs.generate_mortgages(100)
        cohort_keys = [{'amortization_type': at, 'risk_level': 'medium'} for at in self.amortization_types]
        discount_rates = [0.04 for _ in range(30 * 12)]
        combined_pv = self.mbs.calculate_combined_cohorts_present_value(cohort_keys, discount_rates)
        self.assertTrue(combined_pv > 0)

    def test_calculate_combined_cohorts_VaR(self):
        self.mbs.generate_mortgages(100)
        cohort_keys = [{'amortization_type': at, 'risk_level': 'medium'} for at in self.amortization_types]
        confidence_interval = 0.95
        combined_var = self.mbs.calculate_combined_cohorts_VaR(cohort_keys, confidence_interval)
        self.assertEqual(len(combined_var), len(combined_var))

    def test_calculate_combined_cohorts_CVaR(self):
        self.mbs.generate_mortgages(100)
        cohort_keys = [{'amortization_type': at, 'risk_level': 'medium'} for at in self.amortization_types]
        confidence_level = 0.95
        combined_cvar = self.mbs.calculate_combined_cohorts_CVaR(cohort_keys, confidence_level)
        self.assertEqual(len(combined_cvar), len(combined_cvar))

    def test_calculate_combined_cohorts_duration(self):
        self.mbs.generate_mortgages(100)
        cohort_keys = [{'amortization_type': at, 'risk_level': 'medium'} for at in self.amortization_types]
        discount_rates = [0.04 for _ in range(30 * 12)]
        combined_duration = self.mbs.calculate_combined_cohorts_duration_with_correlation(cohort_keys, discount_rates)
        self.assertTrue(all(d > 0 for d in combined_duration))

    def test_calculate_combined_cohorts_convexity(self):
        self.mbs.generate_mortgages(100)
        cohort_keys = [{'amortization_type': at, 'risk_level': 'medium'} for at in self.amortization_types]
        discount_rates = [0.04 for _ in range(30 * 12)]
        combined_convexity = self.mbs.calculate_combined_cohorts_convexity_with_correlation(cohort_keys, discount_rates)
        self.assertTrue(all(c > 0 for c in combined_convexity))

    def test_pad_list(self):
        list_to_pad = [1, 2, 3]
        padded_list = self.mbs.pad_list(list_to_pad, 5)
        self.assertEqual(padded_list, [1, 2, 3, 0, 0])

if __name__ == '__main__':
    unittest.main()
