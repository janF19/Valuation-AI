

import numpy as np
from typing import Dict, Any

class DCFCalculator:
    @staticmethod
    def calculate(financial_data: Dict[str, Any], 
                 growth_rate: float = 0.05,
                 discount_rate: float = 0.1) -> Dict[str, float]:
        
        cf = financial_data['operating_cash_flow'][-1]
        projections = [cf * (1 + growth_rate)**i for i in range(1, 6)]
        terminal_value = projections[-1] * (1 + growth_rate) / (discount_rate - growth_rate)
        
        discounted_cf = [
            cf / (1 + discount_rate)**i 
            for i, cf in enumerate(projections, 1)
        ]
        discounted_terminal = terminal_value / (1 + discount_rate)**5
        
        total_value = sum(discounted_cf) + discounted_terminal
        
        return {
            "dcf_valuation": total_value,
            "residual_value": financial_data['cash_equivalents'] - financial_data['total_debt'],
            "intrinsic_value": max(total_value, residual_value)
        }
        
        
        
        
#it would have been nice to return also dataframe showing the valuation - very useful for report