"""
SWIFT message models and validation
"""

from pydantic import BaseModel, Field 
from typing import Optional, Literal
from datetime import datetime
import uuid


class SWIFTMessage(BaseModel):
    """SWIFT message model with validation"""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: Literal["MT103", "MT202"]
    reference: str 
    amount: str 
    currency: str 
    sender_bic: str 
    receiver_bic: str 
    value_date: str 
    
    # Additional MT103 fields
    ordering_customer: Optional[str] = None
    beneficiary: Optional[str] = None
    remittance_info: Optional[str] = None
    
    # Processing status fields
    validation_status: str = Field(default="PENDING")
    validation_errors: list = Field(default_factory=list)
    fraud_status: str = Field(default="PENDING")
    fraud_score: Optional[float] = None
    processing_status: str = Field(default="PENDING")
    fraud_statements : list = Field(default_factory=list)
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    fraud_evaluation : str = Field(default="PENDING")
    chain_analysis : Optional[str] = ""
    agent_perspectives: Optional[str] = ""

    note: Optional[str] = None
    
    def get_first_digit(self) -> int:
        """Get first digit of amount for Benford's law analysis"""
        amount_str = self.amount.replace('.', '').lstrip('0')
        return int(amount_str[0]) if amount_str else 0
    
    def mark_as_fraudulent(self, score: float, reason: str):
        """Mark message as fraudulent"""
        self.fraud_status = "FRAUDULENT"
        self.fraud_score = score
        self.validation_errors.append(f"FRAUD: {reason}")
    
    def mark_as_held(self, score: float, reason: str):
        """Mark message as held for review"""
        self.fraud_status = "HELD"
        self.fraud_score = score
        self.validation_errors.append(f"HOLD: {reason}")
    
    def mark_as_clean(self, score: float = 0.0):
        """Mark message as clean"""
        self.fraud_status = "CLEAN"
        self.fraud_score = score
    
    class Config:
        arbitrary_types_allowed = True
