"""
SWIFT message generation service
"""

import logging
from typing import List
from faker import Faker
import random
from datetime import datetime, timedelta

from models.swift_message import SWIFTMessage
from models.bank import BankRegistry


class SWIFTGenerator:
    """Service for generating realistic SWIFT messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fake = Faker()
        self.bank_registry = BankRegistry()
        
        # Initialize with fake banks
        self.bank_registry.initialize_with_fake_data(30)
        
        self.logger.info("SWIFT Generator initialized with bank registry")
    
    def generate_messages(self, count: int = 1000, bank_count: int = 30) -> List[SWIFTMessage]:
        """
        Generate specified number of SWIFT messages
        """
        
        messages = []
        
        for i in range(count):
            message = self._generate_single_message()
            messages.append(message)
            
        return messages
    
    def _generate_single_message(self) -> SWIFTMessage:
        """
        Generate a single realistic SWIFT message
        """
        # Random message type
        message_type = random.choice(["MT103", "MT202"])
        
        # Random banks
        sender_bank = self.bank_registry.get_random_bank()
        receiver_bank = self.bank_registry.get_random_bank()
        
        # Ensure different banks
        while receiver_bank.bic_code == sender_bank.bic_code:
            receiver_bank = self.bank_registry.get_random_bank()
        
        # Generate realistic amounts with some pattern variations
        amount = self._generate_realistic_amount()
        
        # Generate reference
        reference = self._generate_reference()
        
        # Generate value date (today to +5 business days)
        value_date = self._generate_value_date()
        
        # Generate currency (mostly USD, some variety)
        currency = self._generate_currency()
        
        # Create message
        message = SWIFTMessage(
            message_type=message_type,
            reference=reference,
            amount=f"{amount:.2f}",
            currency=currency,
            sender_bic=sender_bank.bic_code,
            receiver_bic=receiver_bank.bic_code,
            value_date=value_date
        )
        
        # Add MT103-specific fields
        if message_type == "MT103":
            message.ordering_customer = self._generate_customer_name()
            message.beneficiary = self._generate_customer_name()
            message.remittance_info = self._generate_remittance_info()
        
        return message
    
    def _generate_realistic_amount(self) -> float:
        """
        Generate realistic transaction amounts with various patterns
        """
        # Create distribution that roughly follows real-world patterns
        rand = random.random()
        
        if rand < 0.4:  # 40% small amounts (1-10,000)
            return round(random.uniform(1, 10000), 2)
        elif rand < 0.7:  # 30% medium amounts (10,000-100,000)
            return round(random.uniform(10000, 100000), 2)
        elif rand < 0.9:  # 20% large amounts (100,000-1,000,000)
            return round(random.uniform(100000, 1000000), 2)
        else:  # 10% very large amounts (1,000,000+)
            return round(random.uniform(1000000, 10000000), 2)
    
    def _generate_reference(self) -> str:
        """
        Generate realistic SWIFT reference
        """
        # Various reference patterns
        patterns = [
            lambda: f"PAY{random.randint(100000, 999999)}",
            lambda: f"TXN{self.fake.date_object().strftime('%Y%m%d')}{random.randint(1000, 9999)}",
            lambda: f"REF{self.fake.lexify(text='???????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')}",
            lambda: f"INV{random.randint(10000, 99999)}",
            lambda: f"{self.fake.lexify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000, 999999)}"
        ]
        
        return random.choice(patterns)()[:16]  # Ensure max length
    
    def _generate_value_date(self) -> str:
        """
        Generate realistic value date (YYMMDD format)
        """
        # Value date is typically today to +5 business days
        base_date = datetime.now()
        days_forward = random.randint(0, 7)  # 0-7 days forward
        
        value_date = base_date + timedelta(days=days_forward)
        return value_date.strftime('%y%m%d')
    
    def _generate_currency(self) -> str:
        """
        Generate currency code with realistic distribution
        """
        # Currency distribution based on real SWIFT usage
        currencies = {
            'USD': 0.5,   # 50% USD
            'EUR': 0.2,   # 20% EUR
            'GBP': 0.1,   # 10% GBP
            'JPY': 0.05,  # 5% JPY
            'CHF': 0.05,  # 5% CHF
            'CAD': 0.03,  # 3% CAD
            'AUD': 0.03,  # 3% AUD
            'SGD': 0.02,  # 2% SGD
            'HKD': 0.02   # 2% HKD
        }
        
        rand = random.random()
        cumulative = 0
        
        for currency, probability in currencies.items():
            cumulative += probability
            if rand <= cumulative:
                return currency
        
        return 'USD'  # Default fallback
    
    def _generate_customer_name(self) -> str:
        """
        Generate realistic customer name for MT103
        """
        # Mix of individual and corporate names
        if random.random() < 0.3:  # 30% corporate
            return f"{self.fake.company()} {random.choice(['Ltd', 'Inc', 'Corp', 'LLC', 'AG'])}"
        else:  # 70% individual
            return self.fake.name()
    
    def _generate_remittance_info(self) -> str:
        """
        Generate realistic remittance information
        """
        purposes = [
            "Payment for services",
            "Invoice payment",
            "Salary transfer",
            "Investment transfer",
            "Trade settlement",
            "Property purchase",
            "Loan repayment",
            "Consulting fees",
            "Equipment purchase",
            "Software licensing"
        ]
        
        return random.choice(purposes)
    
    def generate_test_batch_for_benfords(self, count: int = 100, fraud_ratio: float = 0.1) -> List[SWIFTMessage]:
        """
        Generate a test batch with known fraud patterns for Benford's Law testing
        """
        self.logger.info(f"Generating test batch of {count} messages with {fraud_ratio*100}% fraud patterns")
        
        messages = []
        fraud_count = int(count * fraud_ratio)
        clean_count = count - fraud_count
        
        # Generate clean messages (following Benford's Law)
        for i in range(clean_count):
            message = self._generate_single_message()
            messages.append(message)
        
        # Generate fraudulent messages (violating Benford's Law)
        for i in range(fraud_count):
            message = self._generate_fraudulent_message()
            messages.append(message)
        
        # Shuffle to randomize order
        random.shuffle(messages)
        
        self.logger.info(f"Generated test batch: {clean_count} clean, {fraud_count} fraudulent")
        return messages
    
    def _generate_fraudulent_message(self) -> SWIFTMessage:
        """
        Generate message with patterns that violate Benford's Law
        """
        # Generate message with suspicious patterns
        message = self._generate_single_message()
        
        # Force amounts starting with higher digits (violates Benford's Law)
        suspicious_first_digits = [5, 6, 7, 8, 9]
        first_digit = random.choice(suspicious_first_digits)
        
        # Create amount starting with suspicious digit
        magnitude = random.choice([100, 1000, 10000, 100000])
        base_amount = first_digit * magnitude
        variation = random.uniform(0, magnitude * 0.9)
        
        fraudulent_amount = base_amount + variation
        message.amount = f"{fraudulent_amount:.2f}"
        
        return message
