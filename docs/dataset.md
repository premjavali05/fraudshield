# FraudShield Dataset

## Dataset

PaySim Synthetic Financial Dataset for Fraud Detection.

## Target

`isFraud`

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

## Initial observations

- The dataset contains millions of transactions.
- Fraudulent transactions represent a very small minority.
- The target is therefore highly imbalanced.
- The dataset contains a temporal `step` feature.
- Multiple transaction types are present.
- Several balance-related features require careful leakage analysis.
- Sender and receiver identifiers may be useful for behavioral feature engineering but should not be used directly as raw categorical features.

## Important evaluation consideration

Accuracy alone is not an appropriate primary metric because a model can achieve very high accuracy by predicting the majority class while missing most fraudulent transactions.

Primary evaluation metrics will include:

- Precision
- Recall
- F1-score
- PR-AUC
- Confusion Matrix

## Dataset limitation

PaySim is a synthetic dataset. Results obtained on this dataset should not automatically be interpreted as equivalent to performance on real banking transaction data.