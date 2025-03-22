# src/valuation/valuator.py
from backend.src.financials.extractor import FinancialExtractor
from typing import Dict, Any, Tuple
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os
from io import StringIO
import traceback

class CompanyValuator:
    def __init__(self, financial_data: Dict[str, Any]):
        self.financial_data = financial_data
    
    
    def get_multiples(self, industry: str) -> Tuple[float, float]:
        """
        Returns EV/EBITDA and EV/EBIT multiples for a given industry from hardcoded Damodaran data.
        Uses January 2025 data for "All firms" category.
        """
        # Normalize the input industry name
        normalized_industry = industry.lower().strip()
        
        # Hardcoded industry multiples from January 2025 Damodaran data
        # Using the "All firms" columns for EV/EBITDA and EV/EBIT
        industry_multiples = {
            "advertising": (20.07, 24.48),
            "aerospace/defense": (23.20, 34.01),
            "air transport": (9.27, 18.71),
            "apparel": (11.99, 14.71),
            "auto & truck": (49.31, 108.13),
            "auto parts": (8.57, 14.32),
            "beverage (alcoholic)": (12.67, 14.42),
            "beverage (soft)": (18.03, 19.88),
            "broadcasting": (9.07, 10.29),
            "building materials": (14.08, 17.29),
            "business & consumer services": (19.03, 23.97),
            "cable tv": (7.20, 12.52),
            "chemical (basic)": (8.15, 16.53),
            "chemical (diversified)": (8.32, 22.28),
            "chemical (specialty)": (14.51, 22.76),
            "coal & related energy": (4.94, 14.11),
            "computer services": (14.33, 21.94),
            "computers/peripherals": (26.23, 29.46),
            "construction supplies": (13.37, 16.20),
            "diversified": (8.70, 8.82),
            "drugs (biotechnology)": (62.25, None),  # NA value for EV/EBIT
            "drugs (pharmaceutical)": (18.96, 24.05),
            # Add more industries as needed
        }
        
        # Default multiples if the industry is not found
        default_ev_ebitda = 22.86  # Total Market value
        default_ev_ebit = 30.33    # Total Market value
        
        # Try to find the industry in our dictionary
        for ind, multiples in industry_multiples.items():
            if normalized_industry in ind or ind in normalized_industry:
                print(f"Found match: '{ind}' for query '{normalized_industry}'")
                ev_ebitda, ev_ebit = multiples
                
                # Handle NA values
                if ev_ebitda is None:
                    ev_ebitda = default_ev_ebitda
                if ev_ebit is None:
                    ev_ebit = default_ev_ebit
                    
                return ev_ebitda, ev_ebit
        
        # If no match is found, provide default values and log the miss
        print(f"Could not find exact match for '{industry}'. Using market average.")
        return default_ev_ebitda, default_ev_ebit
        def calculate_dcf(self):
            # Implement Discounted Cash Flow valuation
            pass
            
    
    
    def adjust_values_to_2025(self, base_year: int, amount: float) -> float:
        """
        Adjusts financial values from base_year to 2025 using compound inflation.
        
        Args:
            base_year: The year of the financial data
            amount: The amount to be adjusted
        
        Returns:
            float: Inflation-adjusted amount for 2025
        """
        #czech inflation data consider using api later
        INFLATION_DATA = {
            2019: 0.028,
            2020: 0.032,
            2021: 0.038,
            2022: 0.151,
            2023: 0.107,
            2024: 0.024 
        }
        
        def get_inflation_rate(year: int) -> float:
            if year in INFLATION_DATA:
                return INFLATION_DATA[year]
            # Use average inflation if year not available
            return sum(INFLATION_DATA.values()) / len(INFLATION_DATA)
        
        target_year = 2025
        
        # Return original amount if base_year is beyond target
        if base_year >= target_year:
            return amount
        
        # Calculate compound inflation factor
        inflation_factor = 1.0
        for year in range(base_year + 1, target_year + 1):
            rate = get_inflation_rate(year)
            inflation_factor *= (1 + rate)
        
        return amount * inflation_factor
    
    
    
    
    def calculate_multiples(self):
        try:
            # Convert period to integer with error handling
            try:
                period = int(self.financial_data['information']['accounting_period'])
            except (ValueError, TypeError):
                print(f"Warning: Invalid accounting period format: {self.financial_data['information'].get('accounting_period')}. Using current year 2024.")
                #fallback is last year
                period = 2024
            
            operating_profit = self.financial_data['income_statement']['ebit_current']
            depreciation_amortization = self.financial_data['income_statement']['depreciation_current']
            industry = self.financial_data['information']['industry']
            
            print(f"Searching for industry: {industry}")  # Debug print
            
            # Calculate EBITDA
            ebit = operating_profit
            ebitda = ebit + depreciation_amortization
            
            ebit_original = (ebit, period)
            ebitda_original = (ebitda, period)
            
            if period != 2024:
                ebit = self.adjust_values_to_2025(period, ebit)
                ebitda = self.adjust_values_to_2025(period, ebitda)
                
                 
            
            # Get multiples once
            ev_ebitda_multiple, ev_ebit_multiple = self.get_multiples(industry)

            # Calculate Enterprise Value
            enterprise_ebitda_value = ebitda * ev_ebitda_multiple
            enterprise_ebit_value = ebit * ev_ebit_multiple

            # Unpack tuples for clearer dictionary structure
            return {
                "EBIT original": {
                    "value": ebit_original[0],
                    "year": ebit_original[1]
                },
                "EBITDA original": {
                    "value": ebitda_original[0],
                    "year": ebitda_original[1]
                },
                "EBIT": ebit,
                "EBITDA": ebitda,
                "EV/EBITDA Multiple": ev_ebitda_multiple,
                "EV/EBIT Multiple": ev_ebit_multiple,
                "Enterprise Value based on EBITDA (Kč thousands)": enterprise_ebitda_value,
                "Enterprise Value based on EBIT (Kč thousands)": enterprise_ebit_value
            }
        except KeyError as e:
            print(f"Error: Missing required financial data key: {e}")
            return None

        
    def calculate_asset_based(self):
        # Implement asset-based valuation
        pass
    


if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to financial_data.json
    json_path = os.path.join(script_dir, 'financial_data.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        finance_data = json.load(f)
    
    valuator = CompanyValuator(finance_data)
    result = valuator.calculate_multiples()
    print(result)