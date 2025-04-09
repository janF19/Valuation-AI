# src/valuation/valuator.py

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
        "advertising": (11.52, 14.65),
        "aerospace/defense": (19.24, 23.33),
        "air transport": (7.82, 12.40),
        "apparel": (12.56, 15.35),
        "auto & truck": (5.50, 8.99),
        "auto parts": (5.44, 9.73),
        "beverage (alcoholic)": (9.70, 12.20),
        "beverage (soft)": (13.19, 17.16),
        "broadcasting": (7.87, 8.75),
        "building materials": (13.90, 17.57),
        "business & consumer services": (13.96, 18.16),
        "cable tv": (7.25, 34.32),
        "chemical (basic)": (15.89, 44.60),
        "chemical (diversified)": (7.93, 23.12),
        "chemical (specialty)": (15.23, 23.73),
        "coal & related energy": (6.02, 5.81),
        "computer services": (14.77, 17.42),
        "computers/peripherals": (17.37, 20.48),
        "construction supplies": (8.46, 11.31),
        "diversified": (8.98, 8.11),
        "drugs (pharmaceutical)": (12.70, 14.80),
        "education": (14.98, 18.18),
        "electrical equipment": (20.97, 25.35),
        "electronics (general)": (14.86, 18.32),
        "engineering/construction": (8.56, 12.15),
        "entertainment": (33.45, 52.74),
        "environmental & waste services": (9.41, 14.41),
        "farming/agriculture": (9.16, 13.58),
        "financial svcs. (non-bank & insurance)": (79.73, 88.79),
        "food processing": (11.38, 15.30),
        "food wholesalers": (8.18, 20.70),
        "furn/home furnishings": (9.54, 23.01),
        "green & renewable energy": (11.23, 18.44),
        "healthcare products": (18.96, 26.70),
        "healthcare support services": (13.04, 18.14),
        "heathcare information and technology": (18.83, 30.25),
        "homebuilding": (10.00, 11.20),
        "hospitals/healthcare facilities": (15.20, 29.72),
        "hotel/gaming": (13.76, 18.14),
        "household products": (15.31, 16.89),
        "information services": (5.97, 8.61),
        "insurance (general)": (8.52, 7.41),
        "insurance (life)": (11.91, 10.90),
        "insurance (prop/cas.)": (12.67, 11.00),
        "investments & asset management": (17.05, 14.70),
        "machinery": (12.80, 16.16),
        "metals & mining": (5.51, 9.01),
        "office equipment & services": (6.45, 9.08),
        "oil/gas (integrated)": (3.28, 5.96),
        "oil/gas (production and exploration)": (2.33, 4.09),
        "oil/gas distribution": (6.35, 8.82),
        "oilfield svcs/equip.": (3.19, 6.32),
        "packaging & container": (11.20, 19.42),
        "paper/forest products": (10.45, 15.15),
        "power": (6.86, 10.51),
        "precious metals": (6.84, 19.60),
        "publishing & newspapers": (10.56, 15.80),
        "r.e.i.t.": (22.98, 22.34),
        "real estate (development)": (26.73, 81.05),
        "real estate (general/diversified)": (34.05, 45.85),
        "real estate (operations & services)": (25.93, 25.65),
        "recreation": (11.17, 15.97),
        "reinsurance": (9.37, 8.91),
        "restaurant/dining": (17.29, 24.44),
        "retail (automotive)": (8.17, 19.16),
        "retail (building supply)": (8.83, 11.35),
        "retail (distributors)": (11.62, 15.75),
        "retail (general)": (22.33, 31.66),
        "retail (grocery and food)": (8.99, 13.86),
        "retail (reits)": (18.88, 17.95),
        "retail (special lines)": (15.44, 18.41),
        "rubber& tires": (5.51, 8.63),
        "semiconductor": (8.09, 14.70),
        "semiconductor equip": (24.90, 30.99),
        "shipbuilding & marine": (6.40, 9.99),
        "shoe": (21.18, 26.32),
        "software (entertainment)": (15.84, 21.64),
        "software (internet)": (10.62, 21.61),
        "software (system & application)": (32.83, 33.47),
        "steel": (4.63, 27.19),
        "telecom (wireless)": (7.44, 13.49),
        "telecom. equipment": (8.92, 12.59),
        "telecom. services": (7.11, 13.46),
        "tobacco": (7.18, 8.61),
        "transportation": (11.31, 14.95),
        "transportation (railroads)": (13.87, 21.24),
        "trucking": (8.12, 12.66),
        "utility (general)": (5.91, 9.69),
        "utility (water)": (18.11, 28.94),
        }
        
        # Default multiples if the industry is not found
        default_ev_ebitda = 18.53  # Total Market value
        default_ev_ebit = 25.16    # Total Market value
        
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
            
            # Get values with safe defaults if they're None
            operating_profit = self.financial_data['income_statement'].get('operating_profit_current')
            depreciation_amortization = self.financial_data['income_statement'].get('depreciation_current')
            industry = self.financial_data['information'].get('industry', 'Unknown')
            
            print(f"Searching for industry: {industry}")  # Debug print
            
            # Check for None values and handle them
            if operating_profit is None:
                print("Warning: Operating profit is None. Valuation may be incomplete.")
                ebit = None
            else:
                ebit = operating_profit
                
            # Calculate EBITDA only if both values are available
            if ebit is not None and depreciation_amortization is not None:
                ebitda = ebit + depreciation_amortization
            elif ebit is not None:
                print("Warning: Depreciation/amortization is None. EBITDA will equal EBIT.")
                ebitda = ebit
            else:
                ebitda = None
                print("Warning: Cannot calculate EBITDA due to missing data.")
            
            # Store original values with period
            ebit_original = (ebit, period) if ebit is not None else (None, period)
            ebitda_original = (ebitda, period) if ebitda is not None else (None, period)
            
            # Only adjust values if they're not None
            if period != 2024:
                if ebit is not None:
                    ebit = self.adjust_values_to_2025(period, ebit)
                if ebitda is not None:
                    ebitda = self.adjust_values_to_2025(period, ebitda)
            
            # Get multiples once
            ev_ebitda_multiple, ev_ebit_multiple = self.get_multiples(industry)

            # Calculate Enterprise Value only if values are available
            enterprise_ebitda_value = ebitda * ev_ebitda_multiple if ebitda is not None else None
            enterprise_ebit_value = ebit * ev_ebit_multiple if ebit is not None else None

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