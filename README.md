# FraudShield — Financial Transaction Fraud Risk Scoring

> **A machine-learning portfolio project that scores synthetic financial transactions and flags cases for review.**
>
> **Live demo:** `https://fraudshield-ml-project.streamlit.app/`  
> **GitHub repository:** `https://github.com/premjavali05/fraudshield`

FraudShield is an end-to-end ML application built to explore the full lifecycle of a fraud-detection model: inspecting data, engineering features, handling class imbalance, evaluating model behaviour, packaging an inference artifact, serving predictions, recording an audit trail, and deploying an interactive application.

**Important:** FraudShield is an educational research prototype trained on the synthetic PaySim dataset. It is not a banking product, does not prove that a transaction is fraudulent or legitimate, and must not be used to make real financial decisions.

---

## Contents

- [Why this project exists](#why-this-project-exists)
- [What the application does](#what-the-application-does)
- [Live application architecture](#live-application-architecture)
- [Technology stack](#technology-stack)
- [Dataset](#dataset)
- [Model and feature engineering](#model-and-feature-engineering)
- [Evaluation results](#evaluation-results)
- [Important limitations and interpretation](#important-limitations-and-interpretation)
- [Run FraudShield locally](#run-fraudshield-locally)
- [Run the optional FastAPI service](#run-the-optional-fastapi-service)
- [Repository structure](#repository-structure)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Possible next improvements](#possible-next-improvements)
- [Dataset citation](#dataset-citation)

## Why this project exists

Financial transaction fraud detection is not a simple accuracy-maximisation problem. Fraud is rare compared with legitimate activity, and an alerting system must balance two costly errors:

- **False negative:** a fraudulent transaction is not flagged.
- **False positive:** a legitimate transaction is flagged and may require unnecessary investigation.

A model can achieve high accuracy by labelling nearly every transaction as legitimate when fraud is extremely rare. FraudShield was built to study that problem and to turn the modelling work into a usable application—not just a notebook containing an accuracy score.

The project focuses on these engineering and ML questions:

1. What does the dataset contain, and which patterns may be unreliable or simulator-specific?
2. How should training be handled when the target classes are severely imbalanced?
3. How do precision, recall, F1, ROC-AUC, and average precision describe different aspects of performance?
4. How can the training-time feature contract be kept exactly consistent during inference?
5. How can a saved model be connected to an interface and an auditable prediction workflow?
6. What limitations must be disclosed before anyone interprets a fraud-risk score?

## What the application does

The Streamlit dashboard allows a user to enter a PaySim-style transaction and view a model risk score and review recommendation.

For each submitted transaction, FraudShield:

1. Validates the input fields and transaction type.
2. Derives the same 13 input features used by the selected model.
3. Loads the saved model artifact and uses it to produce a score.
4. Compares that score with the saved decision threshold (`0.7800721293318664`).
5. Displays either `FLAG_FOR_REVIEW` or `NOT_FLAGGED`.
6. Records limited prediction metadata in SQLite for the recent-history table.

A score above the threshold means **the model recommends review**. It does not confirm fraud. A score below the threshold does not prove that a transaction is legitimate.

## Live application architecture

The currently deployed Streamlit application performs inference directly; it does **not** need a separately hosted FastAPI server to make predictions.

```text
User
  |
  v
Streamlit dashboard (streamlit_app.py)
  |
  v
Input validation (Pydantic schema)
  |
  v
Shared feature function (build_feature_frame)
  |
  v
Saved scikit-learn model (artifacts/fraud_model.joblib)
  |
  v
Risk score + saved threshold
  |
  +----> Result displayed in the dashboard
  |
  +----> Prediction metadata recorded in SQLite
```

The repository also contains an optional FastAPI implementation. It is useful for local API development and testing, but the current direct-inference Streamlit deployment does not call it. `streamlit_app_api_version.py` is retained as the earlier dashboard variant that called the API.

## Technology stack

| Component | Purpose |
|---|---|
| Python | Application and ML implementation |
| pandas / NumPy | Data handling and numerical feature transformations |
| scikit-learn | `HistGradientBoostingClassifier` model |
| joblib | Save and load the trained model bundle |
| Streamlit | Interactive dashboard and hosted UI |
| Pydantic | Validate transaction input |
| FastAPI / Uvicorn | Optional standalone inference API |
| SQLite | Local prediction audit history |
| pytest / FastAPI TestClient | Automated feature and endpoint tests |
| Git / GitHub | Source control and project history |
| Streamlit Community Cloud | Hosting for the current dashboard |

## Dataset

FraudShield uses **Synthetic Financial Datasets For Fraud Detection**, a PaySim-generated dataset distributed through Kaggle.

- **Source:** [Kaggle — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Local file expected by the notebooks:** `data/raw/PS_20174392719_1491204439457_log.csv`
- **Rows:** 6,362,620
- **Columns:** 11
- **Fraud transactions:** 8,213
- **Fraud prevalence:** approximately 0.129%
- **Time representation:** `step` is a simulation step; in this dataset, one step represents one hour.
- **Transaction types:** `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`.

PaySim is synthetic data designed to resemble patterns in mobile-money transactions. It is not a collection of real, independently verified bank transactions. The CSV is intentionally **not committed to this repository**; download it from the dataset source and place it in `data/raw/` if you want to rerun the exploration and training notebooks.

### Data split used for the reported model

| Partition | Steps | Rows | Fraud cases | Purpose |
|---|---:|---:|---:|---|
| Training partition | 1–520 | 4,865,605 | 4,625 | Fit the final candidate model |
| Internal validation | Within steps 1–520 | 1,216,402 | 1,156 | Select the operating threshold |
| Later-period validation | 521–631 | 191,147 | 1,180 | Check behaviour under temporal distribution shift |
| Final temporal test | 632–743 | 89,466 | 1,252 | Evaluate the selected full-data model on later transactions |
| **Total** | **1–743** | **6,362,620** | **8,213** | |

The training partition contains all transactions in its designated time range. To reduce the influence of the majority class without throwing away legitimate rows, training used sample weights: fraud examples had weight `1.0`, and legitimate examples had weight approximately `0.095145`. This makes the aggregate training weight of the legitimate class approximately 100 times the fraud class's weight, while keeping every row in the training partition.

## Model and feature engineering

### Selected model

- **Estimator:** `HistGradientBoostingClassifier`
- **Training:** full training partition, with sample weights
- **Model artifact:** `artifacts/fraud_model.joblib`
- **Metadata:** `artifacts/model_metadata.json`
- **Feature/model version:** `v2_full_training_weighted`
- **Decision threshold:** `0.7800721293318664`
- **Threshold selection:** maximum F1 on internal validation predictions

The saved joblib bundle contains the fitted estimator, exact feature names and order, threshold, and version metadata. Keep the model artifact and feature function aligned; a feature mismatch can make inference fail or, worse, produce misleading results.

### Exact 13-feature contract

The application accepts five raw fields and deterministically transforms them into these 13 features:

| Feature | How it is derived |
|---|---|
| `amount` | Transaction amount, represented as `float32` |
| `amount_log` | `log1p(amount)` |
| `hour` | `step % 24` |
| `day` | `step // 24` |
| `type_CASH_IN` | One-hot indicator for `CASH_IN` |
| `type_CASH_OUT` | One-hot indicator for `CASH_OUT` |
| `type_DEBIT` | One-hot indicator for `DEBIT` |
| `type_PAYMENT` | One-hot indicator for `PAYMENT` |
| `type_TRANSFER` | One-hot indicator for `TRANSFER` |
| `origin_balance_log` | `log1p(oldbalanceOrg)` |
| `destination_balance_log` | `log1p(oldbalanceDest)` |
| `origin_balance_zero` | `1` when `oldbalanceOrg` is zero, otherwise `0` |
| `destination_balance_zero` | `1` when `oldbalanceDest` is zero, otherwise `0` |

Feature creation lives in `src/fraudshield/features/engineering.py` and is reused by the direct Streamlit app and FastAPI service. Keep that as the single source of truth.

### Excluded fields and why

- `isFraud` is the target, not a model input.
- `isFlaggedFraud` is an existing rule-based flag and was excluded from the primary model.
- `nameOrig` and `nameDest` are raw customer identifiers and are not used directly.
- `newbalanceOrig` and `newbalanceDest` are post-transaction values and are not used by this pre-transaction scoring workflow.
- The experimental `origin_amount_ratio` and `destination_amount_ratio` features were excluded from the selected model after ablation experiments showed that the origin ratio in particular produced a suspiciously large jump in performance.

## Evaluation results

The following results belong to the **full-training-partition weighted model**, evaluated on the later temporal test partition described above.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9988 |
| Average precision (AP) | 0.9447 |
| Fraud precision | 0.9159 (91.59%) |
| Fraud recall | 0.8355 (83.55%) |
| Fraud F1-score | 0.8739 (87.39%) |
| Decision threshold | 0.780072 |

Confusion matrix, in `[[TN, FP], [FN, TP]]` order:

```text
[[88118,    96],
 [  206,  1046]]
```

At the selected threshold, the test partition contained 1,046 fraud cases correctly flagged, 206 fraud cases not flagged, and 96 legitimate cases incorrectly flagged. The threshold expresses an operating choice based on internal validation; a real deployment would need its own cost analysis and validation procedure.

**Evaluation caveat:** the final temporal test was used to compare more than one candidate model during development. The reported result is a useful held-out temporal evaluation of the chosen model, but it is not a pristine, never-inspected benchmark after that comparison. A stronger future study should use a new untouched test period or an independent dataset.

## Important limitations and interpretation

1. **Synthetic data:** PaySim is a simulator dataset. These metrics must not be presented as expected performance on real banking transactions.
2. **Balance-feature risk:** the selected model uses pre-transaction balance features. The PaySim dataset page warns that fraud transactions are cancelled and advises caution about using balance columns for fraud detection. Balance-related patterns may therefore be simulator-specific. Excluding the two ratio features reduced one obvious shortcut, but it does not prove the remaining balance features generalize.
3. **Uncalibrated score:** `risk_score` is the output of the fitted classifier, not a calibrated real-world fraud probability. No probability-calibration study has been completed.
4. **Threshold is not universal:** `0.780072` was selected to maximize F1 on internal validation. A real fraud operation would select an operating point based on its review capacity, false-positive costs, missed-fraud costs, and representative validation data.
5. **Distribution shift:** fraud prevalence changes significantly over the dataset's time periods. Validation across later steps is informative but does not cover every possible shift.
6. **Limited security controls:** the public prototype is not an authenticated financial service. Do not submit real account details or sensitive financial data. Do not use its output to approve, decline, or block real transactions.
7. **Audit persistence:** SQLite provides a lightweight audit trail for local use and demonstration. The hosted app's local filesystem may be ephemeral; prediction history should not be assumed to survive restarts or redeployments.
8. **Test-set reuse:** because candidate-model comparison used the temporal test results, future model changes require a new independent evaluation set for an unbiased final estimate.

## Run FraudShield locally

### Prerequisites

- Python compatible with the versions pinned in `requirements.txt`.
- Git.
- The repository and its committed model artifact.
- For rerunning data exploration or training notebooks, the PaySim CSV placed in `data/raw/`.

### 1. Clone and enter the project

Replace the URL placeholder with the actual GitHub repository URL:

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd fraudshield
```

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, consult your local execution-policy settings or activate the environment using the method appropriate for your machine.

### 3. Run the Streamlit dashboard

```powershell
python -m streamlit run streamlit_app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`. The current dashboard loads the saved model directly, so FastAPI does not need to run for the dashboard to make predictions.

### 4. Run automated tests

```powershell
python -m pytest
```

The project includes tests for feature engineering, API validation and responses, and the prediction audit workflow.

### 5. (Optional) Run the FastAPI service separately

The API is kept as a separate inference interface for development and future integrations. From the project root, run:

```powershell
python -m uvicorn fraudshield.api.main:app --app-dir src --reload
```

Useful local endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic service information |
| `GET` | `/health` | Check that the model loaded |
| `POST` | `/predict` | Score one transaction |
| `GET` | `/audit/recent?limit=20` | Read recent prediction audit records |
| `GET` | `/docs` | Interactive Swagger documentation |

Example `POST /predict` request body:

```json
{
  "step": 100,
  "type": "TRANSFER",
  "amount": 500,
  "oldbalanceOrg": 1000,
  "oldbalanceDest": 0
}
```

Example response structure (score shown here is illustrative):

```json
{
  "audit_id": 1,
  "model_version": "v2_full_training_weighted",
  "feature_version": "v2_full_training_weighted",
  "risk_score": 0.9988,
  "decision_threshold": 0.7800721293318664,
  "flag_for_review": true,
  "decision": "FLAG_FOR_REVIEW"
}
```

## Repository structure

```text
fraudshield/
├── artifacts/
│   ├── fraud_model.joblib
│   ├── fraud_model_undersampled_v1.joblib
│   ├── model_metadata.json
│   └── model_metadata_undersampled_v1.json
├── data/
│   ├── raw/                 # Local PaySim CSV; not committed
│   ├── processed/           # Generated processed data, if used
│   └── app/                 # Local SQLite audit database; not committed
├── docs/
│   └── dataset.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_improved_model.ipynb
├── src/fraudshield/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── features/
│   │   └── engineering.py
│   └── storage/
│       └── database.py
├── tests/
│   ├── test_api.py
│   ├── test_audit.py
│   └── test_features.py
├── streamlit_app.py
├── streamlit_app_api_version.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

### Notebook order

1. `01_data_exploration.ipynb` — audits the dataset, class imbalance, target prevalence, and potentially misleading balance patterns.
2. `02_baseline_model.ipynb` — builds the initial Logistic Regression baseline and demonstrates why accuracy alone is not sufficient.
3. `03_improved_model.ipynb` — compares feature sets, handles class imbalance with sample weights, selects the threshold, and evaluates the chosen candidate.

## Deployment

The current dashboard is deployed using Streamlit Community Cloud from the GitHub repository. To reproduce or update the deployment:

1. Push the required source files, `requirements.txt`, and `artifacts/fraud_model.joblib` to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository and choose `streamlit_app.py` as the entrypoint.
3. Choose a Python version compatible with the pinned dependencies and saved scikit-learn artifact.
4. Check the deployment logs and verify that the model loads and a sample transaction can be scored.
5. After each update, test one flagged example, one unflagged example, and the recent-history table.

Replace the demo/repository URL placeholders at the top of this README with the actual URLs before sharing the project.

## Troubleshooting

- **Model artifact not found:** verify that `artifacts/fraud_model.joblib` exists and is committed to GitHub.
- **Feature contract mismatch:** do not manually change feature names or order. Check `bundle["features"]` in the saved artifact and `MODEL_FEATURES` in `src/fraudshield/features/engineering.py`.
- **Model-load error:** install the versions pinned in `requirements.txt`. Scikit-learn does not guarantee cross-version compatibility for saved Python estimator artifacts; do not load an arbitrary model file from an untrusted source.
- **Missing imports:** activate the project virtual environment and run `python -m pip install -r requirements.txt`.
- **API returns 422:** submit the five transaction input fields (`step`, `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`)—not the prediction response.
- **Hosted history disappeared:** the current SQLite database is local/ephemeral storage, not a guaranteed persistent cloud database.
- **Unexpected score:** check the actual inputs, feature order, model version, and threshold. A model score is not ground truth.

## Possible next improvements

These are future enhancements, not features to assume are already implemented:

- Add persistent hosted audit storage.
- Add CI that runs the test suite for each pull request.
- Record model/data versions and document reproducible retraining commands.
- Add probability calibration and select thresholds based on explicit false-positive and false-negative costs.
- Validate on an independent dataset or a newly reserved test period.
- Investigate performance without balance features and assess how much the model depends on PaySim-specific patterns.
- Add appropriate authentication, rate limiting, and operational monitoring before any broader exposure.
- Add Docker packaging if it helps make local and hosted environments more reproducible.

## Dataset citation

Dataset: [Synthetic Financial Datasets For Fraud Detection — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).

Please consult the dataset page for its current licence and reuse conditions. The dataset is not bundled in this repository.

Original PaySim paper:

E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, “PaySim: A financial mobile money simulator for fraud detection,” *28th European Modeling and Simulation Symposium (EMSS)*, 2016. See the [dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1) for context and citation details.

---

**Project status:** end-to-end research prototype deployed; not validated for real-world financial decision-making.
