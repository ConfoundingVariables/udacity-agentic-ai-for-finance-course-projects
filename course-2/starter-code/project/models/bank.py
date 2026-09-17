"""
Bank models and data structures
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from faker import Faker
import random


class Bank(BaseModel):
    """Bank model with BIC and details"""
    
    bic_code: str = Field(pattern=r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$')
    bank_name: str
    country_code: str = Field(pattern=r'^[A-Z]{2}$')
    city: str
    address: str
    swift_network_member: bool = True
    
    # Risk profiling
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    transaction_volume: int = Field(default=0)
    
    @classmethod
    def generate_fake_banks(cls, count: int) -> List['Bank']:
        """Generate fake banks for testing"""
        fake = Faker()
        banks = []
        
        # Common country codes for international banks
        countries = ['US', 'GB', 'DE', 'FR', 'JP', 'CH', 'SG', 'HK', 'AU', 'CA']
        
        for i in range(count):
            country = random.choice(countries)
            
            # Generate realistic BIC code
            bank_code = fake.lexify(text='????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            location_code = f"{country}{fake.lexify(text='??', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')}"
            branch_code = fake.lexify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            bic = f"{bank_code}{location_code}{branch_code}"
            
            bank = cls(
                bic_code=bic,
                bank_name=f"{fake.company()} Bank",
                country_code=country,
                city=fake.city(),
                address=fake.address().replace('\n', ', '),
                risk_score=random.uniform(0.1, 0.9),
                transaction_volume=random.randint(1000, 50000)
            )
            banks.append(bank)
        
        return banks
    
    def is_high_risk(self) -> bool:
        """Check if bank is considered high risk"""
        return self.risk_score > 0.7
    
    def get_risk_level(self) -> str:
        """Get risk level description"""
        if self.risk_score < 0.3:
            return "LOW"
        elif self.risk_score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"


class BankRegistry:
    """Registry for managing banks"""
    
    def __init__(self):
        self.banks: List[Bank] = []
        self._bic_to_bank = {}
    
    def add_bank(self, bank: Bank):
        """Add bank to registry"""
        self.banks.append(bank)
        self._bic_to_bank[bank.bic_code] = bank
    
    def get_bank_by_bic(self, bic: str) -> Optional[Bank]:
        """Get bank by BIC code"""
        return self._bic_to_bank.get(bic)
    
    def get_random_bank(self) -> Bank:
        """Get random bank from registry"""
        return random.choice(self.banks)
    
    def get_banks_by_country(self, country_code: str) -> List[Bank]:
        """Get all banks from specific country"""
        return [bank for bank in self.banks if bank.country_code == country_code]
    
    def initialize_with_fake_data(self, count: int = 30):
        """Initialize registry with fake bank data"""
        fake_banks = Bank.generate_fake_banks(count)
        for bank in fake_banks:
            self.add_bank(bank)
    
    def to_csv_data(self) -> List[dict]:
        """Convert banks to CSV-compatible format"""
        return [
            {
                'bic_code': bank.bic_code,
                'bank_name': bank.bank_name,
                'country_code': bank.country_code,
                'city': bank.city,
                'address': bank.address,
                'risk_score': bank.risk_score,
                'transaction_volume': bank.transaction_volume
            }
            for bank in self.banks
        ]
