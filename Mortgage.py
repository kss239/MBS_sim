from scipy.stats import norm
import math
import numpy as np

class Mortgage:
    def __init__(self, current_period, risk_level, estimated_default_probability, amortization_type, schedule_details):
        self.current_period = current_period
        self.risk_level = risk_level
        self.estimated_default_probability = estimated_default_probability
        self.amortization_type = amortization_type
        self.schedule_details = schedule_details
        self.schedule = None
        self.stats = None

    def generate_schedule(self):

        if self.amortization_type == 'fixed_rate':
            self.schedule = self.generate_fixed_rate_schedule(self.schedule_details['principal'], self.schedule_details['annual_rate'], self.schedule_details['term_years'])
        elif self.amortization_type == 'variable_rate':
            self.schedule = self.generate_variable_rate_schedule(self.schedule_details['principal'], self.schedule_details['initial_rate'], self.schedule_details['subsequent_rate'], self.schedule_details['fixed_years'], self.schedule_details['term_years'])
        elif self.amortization_type == 'deferred_interest':
            self.schedule = self.generate_deferred_interest_schedule(self.schedule_details['principal'], self.schedule_details['annual_rate'], self.schedule_details['interest_only_years'], self.schedule_details['term_years'])
        elif self.amortization_type == 'balloon_payment':
            self.schedule = self.generate_balloon_payment_schedule(self.schedule_details['principal'], self.schedule_details['annual_rate'], self.schedule_details['term_years'], self.schedule_details['balloon_payment'])
        elif self.amortization_type == 'negative_amortization':
            self.schedule = self.generate_negative_amortization_schedule(self.schedule_details['principal'], self.schedule_details['annual_rate'], self.schedule_details['term_years'], self.schedule_details['min_payment_ratio'])
        else:
            try:
                raise Exception("No Amortization of that Type")
            except Exception as e:
                print(str(e))
                pass

    def generate_fixed_rate_schedule(self, principal, annual_rate, term_years):
        monthly_rate = annual_rate / 12
        num_payments = term_years * 12
        payment = principal * (monthly_rate / (1 - (1 + monthly_rate) ** -num_payments))

        schedule = []
        current_balance = principal
        for n in range(1, num_payments + 1):
            interest = current_balance * monthly_rate
            principal_payment = payment - interest
            current_balance -= principal_payment
            schedule.append({
                'Period': n,
                'Total Payment': payment,
                'Principal': principal_payment,
                'Interest': interest,
                'Remaining Balance': current_balance
            })

        return schedule

    def generate_variable_rate_schedule(self, principal, initial_rate, subsequent_rate, fixed_term_years, total_term_years):
        monthly_initial_rate = initial_rate / 12
        monthly_subsequent_rate = subsequent_rate / 12
        num_payments = total_term_years * 12
        fixed_term_payments = fixed_term_years * 12

        schedule = []
        current_balance = principal

        # Initial fixed-rate period
        initial_payment = principal * (monthly_initial_rate / (1 - (1 + monthly_initial_rate) ** -num_payments))
        for n in range(1, fixed_term_payments + 1):
            interest = current_balance * monthly_initial_rate
            principal_payment = initial_payment - interest
            current_balance -= principal_payment
            schedule.append({
                'Period': n,
                'Total Payment': initial_payment,
                'Principal': principal_payment,
                'Interest': interest,
                'Remaining Balance': current_balance
            })

        # Variable-rate period
        for n in range(fixed_term_payments + 1, num_payments + 1):
            remaining_terms = num_payments - n
            if monthly_subsequent_rate == 0 or remaining_terms == 0:
                adjusted_payment = current_balance  # If rate is 0 or last payment, pay off the remaining balance
            else:
                adjusted_payment = current_balance * (monthly_subsequent_rate / (1 - (1 + monthly_subsequent_rate) ** -remaining_terms))
            interest = current_balance * monthly_subsequent_rate
            principal_payment = adjusted_payment - interest
            current_balance -= principal_payment
            schedule.append({
                'Period': n,
                'Total Payment': adjusted_payment,
                'Principal': principal_payment,
                'Interest': interest,
                'Remaining Balance': current_balance
            })

        return schedule

    def generate_deferred_interest_schedule(self, principal, annual_rate, interest_only_years, total_term_years):
        monthly_rate = annual_rate / 12
        num_payments = total_term_years * 12
        interest_only_payments = interest_only_years * 12
        amortizing_payments = num_payments - interest_only_payments

        schedule = []
        current_balance = principal

        # Interest-only period
        for n in range(1, interest_only_payments + 1):
            interest = current_balance * monthly_rate
            schedule.append({
                'Period': n,
                'Total Payment': interest,
                'Principal': 0,
                'Interest': interest,
                'Remaining Balance': current_balance
            })

        # Amortizing period (if any)
        if amortizing_payments > 0:
            payment_after_interest_only = principal * (monthly_rate / (1 - (1 + monthly_rate) ** -amortizing_payments))
            for n in range(interest_only_payments + 1, num_payments + 1):
                interest = current_balance * monthly_rate
                principal_payment = payment_after_interest_only - interest
                current_balance -= principal_payment
                schedule.append({
                    'Period': n,
                    'Total Payment': payment_after_interest_only,
                    'Principal': principal_payment,
                    'Interest': interest,
                    'Remaining Balance': current_balance
                })
        else:
            # Handle the case where there are no amortizing payments
            # This could involve setting the schedule for these periods, or leaving it as is
            pass

        return schedule


    def generate_balloon_payment_schedule(self, principal, annual_rate, term_years, balloon_payment):
        monthly_rate = annual_rate / 12
        num_payments = term_years * 12
        payment_without_balloon = principal * (monthly_rate / (1 - (1 + monthly_rate) ** -num_payments))

        schedule = []
        current_balance = principal

        for n in range(1, num_payments):
            interest = current_balance * monthly_rate
            principal_payment = payment_without_balloon - interest
            current_balance -= principal_payment
            schedule.append({
                'Period': n,
                'Total Payment': payment_without_balloon,
                'Principal': principal_payment,
                'Interest': interest,
                'Remaining Balance': current_balance
            })

        # Final balloon payment
        final_interest = current_balance * monthly_rate
        schedule.append({
            'Period': num_payments,
            'Total Payment': current_balance + final_interest,
            'Principal': current_balance,
            'Interest': final_interest,
            'Remaining Balance': 0
        })

        return schedule

    def generate_negative_amortization_schedule(self, principal, annual_rate, term_years, min_payment_ratio=0.5):
        monthly_rate = annual_rate / 12
        num_payments = term_years * 12
        minimum_payment = min_payment_ratio * monthly_rate * principal # Use the provided fixed minimum payment

        schedule = []
        current_balance = principal

        for n in range(1, num_payments + 1):
            interest = current_balance * monthly_rate
            principal_payment = min(minimum_payment, interest) - interest
            current_balance -= principal_payment  # In negative amortization, the balance increases if payment < interest
            schedule.append({
                'Period': n,
                'Total Payment': minimum_payment,
                'Principal': principal_payment,
                'Interest': interest,
                'Remaining Balance': current_balance
            })

        return schedule

    def expected_cashflows(self, payment_type = 'Total Payment'):
        expected_cashflows = []
        cashflow_variance = []

        n_period = 1
        for schedule_period in self.schedule:
            value = schedule_period[payment_type]
            n_period_default_prob = self.estimated_default_probability/12

            expected_cashflow = value * (1 - n_period_default_prob) ** n_period
            expected_cashflows.append(expected_cashflow)
            cashflow_variance.append((value ** 2) * ((1 - n_period_default_prob) ** n_period) - (value * ((1 - n_period_default_prob) ** n_period))**2)
            n_period += 1

        if payment_type == 'Total Payment':
            self.stats = {'expected_cashflow': expected_cashflows, 'cashflow_variance': cashflow_variance}
        return expected_cashflows, cashflow_variance

    def calculate_present_value(self, discount_rates):
        """
        Calculates the present value of the expected cash flows of the mortgage.

        :param discount_rates: A list of discount rates corresponding to each period.
        :return: Present value of the mortgage.
        """
        if self.stats is None:
            self.expected_cashflows()

        present_value = 0
        for i, cashflow in enumerate(self.stats['expected_cashflow']):
            rate = discount_rates[i] if i < len(discount_rates) else discount_rates[-1]
            present_value += cashflow / ((1 + rate) ** (i + 1))

        return present_value

    def calculate_VaR(self, confidence_interval):
        if self.stats is None:
            self.expected_cashflows()

        var_list = []
        for i in range(len(self.stats['expected_cashflow'])):
            mean_cashflow = self.stats['expected_cashflow'][i]
            std_deviation = self.stats['cashflow_variance'][i] ** 0.5
            VaR = norm.ppf(1 - confidence_interval, mean_cashflow, std_deviation)
            var_list.append(VaR)
        if 'Var' not in self.stats:
            self.stats['Var'] = {}
        self.stats['Var'][confidence_interval] = var_list
        return var_list

    def calculate_CVaR(self, confidence_level):
        """
        Calculate the Conditional Value at Risk (CVaR) at a specified confidence level.

        :param confidence_level: Confidence level for CVaR calculation (e.g., 0.95 for 95% confidence).
        :return: List of CVaR values for each period.
        """
        if self.stats is None:
            self.expected_cashflows()

        alpha = 1 - confidence_level
        cvar_list = []
        for i in range(len(self.stats['expected_cashflow'])):
            mean_cashflow = self.stats['expected_cashflow'][i]
            std_deviation = np.sqrt(self.stats['cashflow_variance'][i])
            
            # Calculate CVaR using the adapted formula
            cvar = (alpha ** -1) * norm.pdf(norm.ppf(alpha)) * std_deviation - mean_cashflow
            cvar_list.append(cvar)

        if 'CVaR' not in self.stats:
            self.stats['CVaR'] = {}
        self.stats['CVaR'][confidence_level] = cvar_list

        return cvar_list

    def calculate_duration(self, discount_rates, duration_type='Macaulay'):
        """
        Calculates the Macaulay or Modified Duration of the mortgage.

        :param discount_rates: A list of discount rates corresponding to each period.
        :param duration_type: Type of duration to calculate ('Macaulay' or 'Modified').
        :return: Duration of the mortgage.
        """
        if self.stats is None:
            self.expected_cashflows()

        if len(discount_rates) < len(self.stats['expected_cashflow']):
            discount_rates.extend([discount_rates[-1]] * (len(self.stats['expected_cashflow']) - len(discount_rates)))


        present_values = [cf / ((1 + discount_rates[i]) ** (i + 1)) for i, cf in enumerate(self.stats['expected_cashflow'])]
        total_pv = sum(present_values)
        weighted_terms = [present_values[i] * (i + 1) for i in range(len(present_values))]
        macaulay_duration = sum(weighted_terms) / total_pv

        if duration_type == 'Modified':
            yield_per_period = discount_rates[0]  # Assuming constant yield for simplicity
            modified_duration = macaulay_duration / (1 + yield_per_period)
            return modified_duration

        return macaulay_duration

    def calculate_convexity(self, discount_rates):
        """
        Calculates the convexity of the mortgage.

        :param discount_rates: A list of discount rates corresponding to each period.
        :return: Convexity of the mortgage.
        """
        if self.stats is None:
            self.expected_cashflows()

        if len(discount_rates) < len(self.stats['expected_cashflow']):
            discount_rates.extend([discount_rates[-1]] * (len(self.stats['expected_cashflow']) - len(discount_rates)))


        present_values = [cf / ((1 + discount_rates[i]) ** (i + 1)) for i, cf in enumerate(self.stats['expected_cashflow'])]
        total_pv = sum(present_values)
        convexity_terms = [present_values[i] * (i + 1) * (i + 2) for i in range(len(present_values))]
        convexity = sum(convexity_terms) / (total_pv * (1 + discount_rates[0]) ** 2)

        return convexity
