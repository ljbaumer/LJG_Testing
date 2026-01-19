from dataclasses import dataclass
from typing import List


@dataclass
class IncomeInputs:
    starting_total_comp: float
    annual_growth_rate: float
    effective_tax_rate: float


@dataclass 
class IncomeProjection:
    year: int
    gross_income: float
    taxes: float
    net_income: float


class IncomeModel:
    def __init__(self, inputs: IncomeInputs):
        self.inputs = inputs
    
    def project(self, years: int) -> List[IncomeProjection]:
        projections = []
        
        for year in range(1, years + 1):
            gross_income = self.inputs.starting_total_comp * (1 + self.inputs.annual_growth_rate) ** (year - 1)
            taxes = gross_income * self.inputs.effective_tax_rate
            net_income = gross_income - taxes
            
            projections.append(IncomeProjection(
                year=year,
                gross_income=gross_income,
                taxes=taxes,
                net_income=net_income
            ))
        
        return projections