
from typing import Final

import numpy as np
import pandas as pd

from fraudshield.api.schemas import TransactionRequest


TRANSACTION_TYPES: Final[tuple[str, ...]] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

# The exact feature contract recorded in our trained model.
MODEL_FEATURES: Final[tuple[str, ...]] = (
    "amount",
    "amount_log",
    "hour",
    "day",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "origin_balance_log",
    "destination_balance_log",
    "origin_balance_zero",
    "destination_balance_zero",
)

FEATURE_DTYPES = {
    "amount": "float32",
    "amount_log": "float32",
    "hour": "int8",
    "day": "int16",
    "type_CASH_IN": "uint8",
    "type_CASH_OUT": "uint8",
    "type_DEBIT": "uint8",
    "type_PAYMENT": "uint8",
    "type_TRANSFER": "uint8",
    "origin_balance_log": "float32",
    "destination_balance_log": "float32",
    "origin_balance_zero": "uint8",
    "destination_balance_zero": "uint8",
}


def build_feature_frame(
    transaction: TransactionRequest,
) -> pd.DataFrame:
    """Build the exact 13-column feature frame used during training."""

    # Match the float32 representation used for the raw CSV columns.
    amount = np.float32(transaction.amount)
    origin_balance = np.float32(transaction.oldbalanceOrg)
    destination_balance = np.float32(transaction.oldbalanceDest)

    # Match the notebook's log1p transformations.
    features = {
        "amount": amount,
        "amount_log": np.float32(np.log1p(float(amount))),
        "hour": np.int8(transaction.step % 24),
        "day": np.int16(transaction.step // 24),
        "origin_balance_log": np.float32(
            np.log1p(float(origin_balance))
        ),
        "destination_balance_log": np.float32(
            np.log1p(float(destination_balance))
        ),
        "origin_balance_zero": np.uint8(origin_balance == 0),
        "destination_balance_zero": np.uint8(
            destination_balance == 0
        ),
    }

    # Always create all five one-hot columns, including zeros.
    # This avoids missing-column errors when a request contains
    # just one transaction type.
    for transaction_type in TRANSACTION_TYPES:
        features[f"type_{transaction_type}"] = np.uint8(
            transaction.type == transaction_type
        )

    frame = pd.DataFrame([features]).astype(FEATURE_DTYPES)

    # Fail rather than silently serving a mismatched model.
    if tuple(frame.columns) != MODEL_FEATURES:
        frame = frame.loc[:, list(MODEL_FEATURES)]

    if tuple(frame.columns) != MODEL_FEATURES:
        raise RuntimeError("Feature order does not match the model contract.")

    if frame.shape != (1, 13):
        raise RuntimeError(
            f"Expected one row and 13 features, got {frame.shape}."
        )

    return frame
