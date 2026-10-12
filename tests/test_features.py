
import numpy as np

from fraudshield.api.schemas import TransactionRequest
from fraudshield.features.engineering import (
    MODEL_FEATURES,
    build_feature_frame,
)


def test_feature_contract_and_order():
    transaction = TransactionRequest(
        step=49,
        type="TRANSFER",
        amount=1000,
        oldbalanceOrg=1000,
        oldbalanceDest=250,
    )

    frame = build_feature_frame(transaction)

    assert tuple(frame.columns) == MODEL_FEATURES
    assert frame.shape == (1, 13)
    assert frame.loc[0, "hour"] == 1
    assert frame.loc[0, "day"] == 2

    assert frame.loc[0, "type_TRANSFER"] == 1
    assert sum(
        int(frame.loc[0, f"type_{kind}"])
        for kind in (
            "CASH_IN", "CASH_OUT", "DEBIT",
            "PAYMENT", "TRANSFER"
        )
    ) == 1

    assert np.isclose(
        frame.loc[0, "amount_log"],
        np.log1p(float(np.float32(1000))),
    )


def test_zero_balance_indicators():
    transaction = TransactionRequest(
        step=10,
        type="PAYMENT",
        amount=20,
        oldbalanceOrg=0,
        oldbalanceDest=0,
    )

    frame = build_feature_frame(transaction)

    assert frame.loc[0, "origin_balance_zero"] == 1
    assert frame.loc[0, "destination_balance_zero"] == 1
