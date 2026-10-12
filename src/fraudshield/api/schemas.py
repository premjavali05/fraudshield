
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TransactionType = Literal[
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, le=32767)
    type: TransactionType
    amount: float = Field(ge=0, le=1e30, allow_inf_nan=False)
    oldbalanceOrg: float = Field(
        ge=0, le=1e30, allow_inf_nan=False
    )
    oldbalanceDest: float = Field(
        ge=0, le=1e30, allow_inf_nan=False
    )


class PredictionResponse(BaseModel):
    audit_id: int
    model_version: str
    feature_version: str
    risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    flag_for_review: bool
    decision: Literal["FLAG_FOR_REVIEW", "NOT_FLAGGED"]


class AuditRecordResponse(BaseModel):
    id: int
    created_at: str
    transaction_type: TransactionType
    risk_score: float
    decision_threshold: float
    flag_for_review: bool
    model_version: str
