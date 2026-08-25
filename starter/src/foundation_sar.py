"""Validated schemas, CSV loading, case assembly, and append-only audit logging."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4

import pandas as pd
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class DomainModel(BaseModel):
    """Strict base model shared by the project schemas."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerData(DomainModel):
    customer_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    date_of_birth: date
    ssn_last_4: str = Field(pattern=r"^\d{4}$")
    address: str = Field(min_length=1)
    phone: str | None = None
    customer_since: date
    risk_rating: Literal["Low", "Medium", "High"]
    occupation: str | None = None
    annual_income: float | None = Field(default=None, ge=0)


class AccountData(DomainModel):
    account_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    account_type: str = Field(min_length=1)
    opening_date: date
    current_balance: float = Field(ge=0)
    average_monthly_balance: float = Field(ge=0)
    status: str = Field(min_length=1)


class TransactionData(DomainModel):
    transaction_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    transaction_date: date
    transaction_type: str = Field(min_length=1)
    amount: float = Field(ge=0)
    description: str | None = None
    counterparty: str | None = None
    location: str | None = None
    method: str | None = None


class CaseData(DomainModel):
    case_id: str = Field(min_length=1)
    created_at: datetime = Field(
        validation_alias=AliasChoices("created_at", "case_created_at")
    )
    customer: CustomerData
    accounts: list[AccountData]
    transactions: list[TransactionData]
    data_sources: dict[str, str] = Field(default_factory=dict)

    @field_validator("created_at", mode="before")
    @classmethod
    def normalize_created_at(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return value
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @property
    def case_created_at(self) -> datetime:
        """Compatibility name used by the supplied notebook tests."""

        return self.created_at

    @model_validator(mode="after")
    def validate_case(self) -> CaseData:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        account_ids = {account.account_id for account in self.accounts}
        if any(
            account.customer_id != self.customer.customer_id
            for account in self.accounts
        ):
            raise ValueError("all accounts must belong to the case customer")
        if len(account_ids) != len(self.accounts):
            raise ValueError("duplicate account_id")

        transaction_ids = {
            transaction.transaction_id for transaction in self.transactions
        }
        if account_ids and any(
            transaction.account_id not in account_ids
            for transaction in self.transactions
        ):
            raise ValueError("all transactions must belong to a case account")
        if len(transaction_ids) != len(self.transactions):
            raise ValueError("duplicate transaction_id")
        return self


class DatasetSummary(DomainModel):
    customer_count: int = Field(ge=0)
    account_count: int = Field(ge=0)
    transaction_count: int = Field(ge=0)


class RiskAnalystOutput(DomainModel):
    case_id: str = Field(default="UNASSIGNED", min_length=1)
    suspicious_activity_type: Literal[
        "Structuring", "Sanctions", "Fraud", "Money_Laundering", "Other"
    ] = Field(
        validation_alias=AliasChoices("suspicious_activity_type", "classification")
    )
    confidence_score: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    reasoning: str = Field(min_length=1)
    suspicious_indicators: list[str] = Field(
        validation_alias=AliasChoices("suspicious_indicators", "key_indicators")
    )

    @property
    def classification(self) -> str:
        """Compatibility name used by supplied tests and notebooks."""

        return self.suspicious_activity_type

    @property
    def key_indicators(self) -> list[str]:
        """Compatibility name used by supplied tests and notebooks."""

        return self.suspicious_indicators


class ComplianceOfficerOutput(DomainModel):
    case_id: str = Field(min_length=1)
    sar_narrative: str = Field(
        min_length=1,
        validation_alias=AliasChoices("sar_narrative", "narrative"),
    )
    regulatory_citations: list[str]
    completeness_check: bool
    reasoning: str = Field(
        min_length=1,
        validation_alias=AliasChoices("reasoning", "narrative_reasoning"),
    )
    word_count: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def derive_word_count(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, Any], data)
        if "word_count" in raw:
            return raw
        narrative = str(raw.get("sar_narrative", raw.get("narrative", "")))
        return {**raw, "word_count": len(narrative.split())}

    @model_validator(mode="after")
    def validate_word_count(self) -> ComplianceOfficerOutput:
        actual_count = len(self.sar_narrative.split())
        if self.word_count != actual_count:
            raise ValueError("word_count must match sar_narrative")
        if actual_count > 120:
            raise ValueError("narrative exceeds 120 word limit")
        return self

    @property
    def narrative(self) -> str:
        """Compatibility name used by supplied tests and notebooks."""

        return self.sar_narrative

    @property
    def narrative_reasoning(self) -> str:
        """Compatibility name used by supplied tests and notebooks."""

        return self.reasoning


class AuditLogEntry(DomainModel):
    event_id: str = Field(default_factory=lambda: f"AUDIT-{uuid4()}")
    timestamp: datetime
    component: str = Field(min_length=1)
    action: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    processing_time: float = Field(default=0, ge=0)
    success: bool = True
    error_message: str | None = None
    user_decision: str | None = None
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    regulatory_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timestamp(self) -> AuditLogEntry:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return self


class ExplainabilityLogger:
    """Write typed, append-only audit events and derived summaries."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.entries: list[dict[str, Any]] = []

    def log(
        self,
        *,
        component: str,
        action: str,
        case_id: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        reasoning: str | None = None,
        processing_time: float = 0,
        success: bool = True,
        error_message: str | None = None,
        user_decision: str | None = None,
        confidence_scores: dict[str, float] | None = None,
        regulatory_flags: list[str] | None = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            component=component,
            action=action,
            case_id=case_id,
            input_data=input_data or {},
            output_data=output_data or {},
            reasoning=reasoning,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            user_decision=user_decision,
            confidence_scores=confidence_scores or {},
            regulatory_flags=regulatory_flags or [],
        )
        record = entry.model_dump(mode="json")
        self.entries.append(record)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return entry

    def log_agent_action(
        self,
        *,
        agent_type: str,
        action: str,
        case_id: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        reasoning: str | None = None,
        execution_time_ms: float = 0,
        success: bool = True,
        error_message: str | None = None,
        user_decision: str | None = None,
        confidence_scores: dict[str, float] | None = None,
        regulatory_flags: list[str] | None = None,
    ) -> AuditLogEntry:
        """Compatibility adapter for the supplied agent interfaces."""

        entry = self.log(
            component=agent_type,
            action=action,
            case_id=case_id,
            input_data=input_data,
            output_data=output_data,
            reasoning=reasoning,
            processing_time=execution_time_ms / 1000,
            success=success,
            error_message=error_message,
            user_decision=user_decision,
            confidence_scores=confidence_scores,
            regulatory_flags=regulatory_flags,
        )
        self.entries[-1]["agent_type"] = agent_type
        self.entries[-1]["execution_time_ms"] = execution_time_ms
        return entry

    def write_summary(self, summary_path: str | Path) -> dict[str, int]:
        entries: list[dict[str, Any]] = []
        if self.log_path.exists():
            entries = [
                json.loads(line)
                for line in self.log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        summary = {
            "entry_count": len(entries),
            "success_count": sum(
                1 for entry in entries if entry.get("success") is True
            ),
            "failure_count": sum(
                1 for entry in entries if entry.get("success") is False
            ),
        }
        output_path = Path(summary_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


ModelT = TypeVar("ModelT", bound=DomainModel)


def _validated_rows(data_dir: Path, filename: str, model: type[ModelT]) -> list[ModelT]:
    try:
        frame = pd.read_csv(data_dir / filename, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"could not read {filename}: {exc}") from exc

    validated: list[ModelT] = []
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        normalized = {
            key: (value.strip() if isinstance(value, str) and value.strip() else None)
            for key, value in row.items()
        }
        try:
            validated.append(model.model_validate(normalized))
        except ValidationError as exc:
            raise ValueError(f"{filename} row {row_number}: {exc}") from exc
    return validated


class DataLoader:
    """Load/index source files once and assemble validated case snapshots."""

    def __init__(
        self,
        customers: dict[str, CustomerData] | ExplainabilityLogger | None = None,
        accounts: dict[str, AccountData] | None = None,
        transactions: dict[str, TransactionData] | None = None,
        logger: ExplainabilityLogger | None = None,
    ):
        if isinstance(customers, ExplainabilityLogger):
            if any(value is not None for value in (accounts, transactions, logger)):
                raise TypeError("logger-only construction does not accept other arguments")
            logger = customers
            customers = None

        self._customers = customers or {}
        self._accounts = accounts or {}
        self._transactions = transactions or {}
        self._logger = logger
        self._created_at = datetime.now(timezone.utc)
        self._case_ids = {
            customer_id: f"CASE-{uuid4()}" for customer_id in self._customers
        }
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        account_lists: dict[str, list[AccountData]] = {}
        for account in self._accounts.values():
            account_lists.setdefault(account.customer_id, []).append(account)
        self._accounts_by_customer = {
            key: tuple(values) for key, values in account_lists.items()
        }

        transaction_lists: dict[str, list[TransactionData]] = {}
        for transaction in self._transactions.values():
            transaction_lists.setdefault(transaction.account_id, []).append(transaction)
        self._transactions_by_account = {
            key: tuple(values) for key, values in transaction_lists.items()
        }

    @classmethod
    def load(
        cls,
        data_dir: str | Path,
        audit_log_path: str | Path | None = None,
    ) -> DataLoader:
        root = Path(data_dir)
        filenames = ("customers.csv", "accounts.csv", "transactions.csv")
        missing = [
            filename for filename in filenames if not (root / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing dataset files: {', '.join(missing)}")

        logger = ExplainabilityLogger(audit_log_path) if audit_log_path else None
        customers: dict[str, CustomerData] = {}
        for customer in _validated_rows(root, "customers.csv", CustomerData):
            if customer.customer_id in customers:
                raise ValueError(f"duplicate customer_id: {customer.customer_id}")
            customers[customer.customer_id] = customer

        accounts: dict[str, AccountData] = {}
        for account in _validated_rows(root, "accounts.csv", AccountData):
            if account.customer_id not in customers:
                raise ValueError(
                    f"accounts.csv references unknown customer: {account.customer_id}"
                )
            if account.account_id in accounts:
                raise ValueError(f"duplicate account_id: {account.account_id}")
            accounts[account.account_id] = account

        transactions: dict[str, TransactionData] = {}
        for transaction in _validated_rows(root, "transactions.csv", TransactionData):
            if transaction.account_id not in accounts:
                raise ValueError(
                    "transactions.csv references unknown account: "
                    f"{transaction.account_id}"
                )
            if transaction.transaction_id in transactions:
                raise ValueError(
                    f"duplicate transaction_id: {transaction.transaction_id}"
                )
            transactions[transaction.transaction_id] = transaction

        loader = cls(customers, accounts, transactions, logger=logger)
        if logger:
            logger.log(
                component="DataLoader",
                action="dataset_loaded",
                case_id="SYSTEM",
                output_data=loader.summary().model_dump(),
            )
        return loader

    def create_case_from_data(
        self,
        customer_data: dict[str, Any],
        account_data: list[dict[str, Any]],
        transaction_data: list[dict[str, Any]],
    ) -> CaseData:
        """Validate in-memory records through the same canonical schemas."""

        customer = CustomerData.model_validate(customer_data)
        accounts = [AccountData.model_validate(item) for item in account_data]
        transactions = [
            TransactionData.model_validate(item) for item in transaction_data
        ]
        case = CaseData(
            case_id=f"CASE-{uuid4()}",
            created_at=datetime.now(timezone.utc),
            customer=customer,
            accounts=accounts,
            transactions=transactions,
            data_sources={
                "customer_source": "in_memory",
                "account_source": "in_memory",
                "transaction_source": "in_memory",
            },
        )
        if self._logger:
            self._logger.log(
                component="DataLoader",
                action="case_created",
                case_id=case.case_id,
                input_data={"customer_id": customer.customer_id},
                output_data={
                    "account_count": len(accounts),
                    "transaction_count": len(transactions),
                },
            )
        return case

    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            customer_count=len(self._customers),
            account_count=len(self._accounts),
            transaction_count=len(self._transactions),
        )

    def list_cases(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cases: list[dict[str, Any]] = []
        for customer in list(self._customers.values())[:limit]:
            accounts = self._accounts_by_customer.get(customer.customer_id, ())
            transaction_count = sum(
                len(self._transactions_by_account.get(account.account_id, ()))
                for account in accounts
            )
            cases.append(
                {
                    "case_id": self._case_ids[customer.customer_id],
                    "created_at": self._created_at.isoformat(),
                    "customer_id": customer.customer_id,
                    "name": customer.name,
                    "risk_rating": customer.risk_rating,
                    "account_count": len(accounts),
                    "transaction_count": transaction_count,
                }
            )
        return cases

    def get_case(self, customer_id: str) -> CaseData:
        customer = self._customers.get(customer_id)
        if customer is None:
            raise KeyError(customer_id)
        accounts = self._accounts_by_customer.get(customer_id, ())
        transactions = tuple(
            transaction
            for account in accounts
            for transaction in self._transactions_by_account.get(account.account_id, ())
        )
        case_id = self._case_ids[customer_id]
        if self._logger:
            self._logger.log(
                component="DataLoader",
                action="case_created",
                case_id=case_id,
                input_data={"customer_id": customer_id},
            )
        return CaseData(
            case_id=case_id,
            created_at=self._created_at,
            customer=customer,
            accounts=list(accounts),
            transactions=list(transactions),
            data_sources={
                "customer_source": "customers.csv",
                "account_source": "accounts.csv",
                "transaction_source": "transactions.csv",
            },
        )


def load_csv_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, ...]:
    """Load the three raw CSV files for notebook exploration."""

    root = Path(data_dir)
    try:
        return (
            pd.read_csv(root / "customers.csv", dtype={"ssn_last_4": str}),
            pd.read_csv(root / "accounts.csv"),
            pd.read_csv(root / "transactions.csv"),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Error loading CSV data: {exc}") from exc


__all__ = [
    "AccountData",
    "AuditLogEntry",
    "CaseData",
    "ComplianceOfficerOutput",
    "CustomerData",
    "DataLoader",
    "DatasetSummary",
    "ExplainabilityLogger",
    "RiskAnalystOutput",
    "TransactionData",
    "load_csv_data",
]
