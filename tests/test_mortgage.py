import unittest
from MBS_sim import Mortgage  # Replace with the actual module name
import numpy as np

class TestMortgage(unittest.TestCase):

    def setUp(self):
        self.base_mortgage_details = {
            'current_period': 1,
            'risk_level': 'medium',
            'estimated_default_probability': 0.05,
            'amortization_type': None,
            'schedule_details': {
                'principal': 200000,
                'annual_rate': 0.05,
                'term_years': 30
            }
        }

    def create_mortgage(self, amortization_type, additional_details=None):
        mortgage_details = self.base_mortgage_details.copy()
        mortgage_details['amortization_type'] = amortization_type
        if additional_details:
            mortgage_details['schedule_details'].update(additional_details)
        return Mortgage(**mortgage_details)

    def test_fixed_rate_schedule(self):
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        self.assertIsNotNone(mortgage.schedule)
        self.assertEqual(len(mortgage.schedule), 30 * 12)

    def test_variable_rate_schedule(self):
        additional_details = {
            'initial_rate': 0.04,
            'subsequent_rate': 0.06,
            'fixed_years': 5
        }
        mortgage = self.create_mortgage('variable_rate', additional_details)
        mortgage.generate_schedule()
        self.assertIsNotNone(mortgage.schedule)
        self.assertEqual(len(mortgage.schedule), 30 * 12)

    def test_deferred_interest_schedule(self):
        additional_details = {'interest_only_years': 5}
        mortgage = self.create_mortgage('deferred_interest', additional_details)
        mortgage.generate_schedule()
        self.assertIsNotNone(mortgage.schedule)
        for payment in mortgage.schedule[:5*12]:
            self.assertEqual(payment['Principal'], 0)

    def test_balloon_payment_schedule(self):
        additional_details = {'balloon_payment': 100000}
        mortgage = self.create_mortgage('balloon_payment', additional_details)
        mortgage.generate_schedule()
        self.assertIsNotNone(mortgage.schedule)
        self.assertEqual(mortgage.schedule[-1]['Principal'], 100000)

    def test_negative_amortization_schedule(self):
        additional_details = {'min_payment_ratio': 0.5}
        mortgage = self.create_mortgage('negative_amortization', additional_details)
        mortgage.generate_schedule()
        self.assertIsNotNone(mortgage.schedule)
        initial_balance = mortgage.schedule[0]['Remaining Balance']
        later_balance = mortgage.schedule[12]['Remaining Balance']
        self.assertGreater(later_balance, initial_balance)

    def test_expected_cashflows(self):
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        expected_cashflows, cashflow_variance = mortgage.expected_cashflows()
        self.assertEqual(len(expected_cashflows), len(mortgage.schedule))
        self.assertEqual(len(cashflow_variance), len(mortgage.schedule))

    def test_calculate_present_value(self):
        discount_rates = [0.04 for _ in range(30 * 12)]
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        present_value = mortgage.calculate_present_value(discount_rates)
        self.assertGreater(present_value, 0)

    def test_calculate_VaR(self):
        confidence_interval = 0.95
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        mortgage.expected_cashflows()
        var_list = mortgage.calculate_VaR(confidence_interval)
        self.assertEqual(len(var_list), len(mortgage.schedule))

    def test_calculate_CVaR(self):
        confidence_level = 0.95
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        mortgage.expected_cashflows()
        cvar_list = mortgage.calculate_CVaR(confidence_level)
        self.assertEqual(len(cvar_list), len(mortgage.schedule))

    def test_calculate_duration(self):
        discount_rates = [0.04 for _ in range(30 * 12)]
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        duration = mortgage.calculate_duration(discount_rates)
        self.assertGreater(duration, 0)

    def test_calculate_convexity(self):
        discount_rates = [0.04 for _ in range(30 * 12)]
        mortgage = self.create_mortgage('fixed_rate')
        mortgage.generate_schedule()
        convexity = mortgage.calculate_convexity(discount_rates)
        self.assertGreater(convexity, 0)

    def test_invalid_amortization_type(self):
        mortgage_details = self.base_mortgage_details.copy()
        mortgage_details['amortization_type'] = 'invalid_type'
        with self.assertRaises(ValueError):
            mortgage = Mortgage(**mortgage_details)
            mortgage.generate_schedule()

    def test_zero_principal(self):
        mortgage_details = self.base_mortgage_details.copy()
        mortgage_details['schedule_details']['principal'] = 0
        mortgage = Mortgage(**mortgage_details)
        mortgage.generate_schedule()
        self.assertEqual(len(mortgage.schedule), 0)

    def test_high_interest_rate(self):
        mortgage_details = self.base_mortgage_details.copy()
        mortgage_details['schedule_details']['annual_rate'] = 1  # 100% interest rate
        mortgage = Mortgage(**mortgage_details)
        mortgage.generate_schedule()
        self.assertGreater(mortgage.schedule[0]['Total Payment'], mortgage.schedule[0]['Principal'])

    def test_negative_principal(self):
        mortgage_details = self.base_mortgage_details.copy()
        mortgage_details['schedule_details']['principal'] = -100000
        with self.assertRaises(ValueError):
            mortgage = Mortgage(**mortgage_details)
            mortgage.generate_schedule()

if __name__ == '__main__':
    unittest.main()
