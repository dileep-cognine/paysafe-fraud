# PaySafe Fraud Scoring MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/data-DVC-945DD6.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

Production-grade, end-to-end Machine Learning Operations (MLOps) system for real-time transaction fraud scoring. Designed with strict train/serve consistency, zero-leakage schema enforcement, automated quality gates, MLflow experiment tracking and model registry promotion, and secure containerized deployment.

---

## 1. System Architecture & Flow

The system strictly adheres to the following production lifecycle:

```text
Raw Data (CSV)
   ↓
Schema & Data Quality Validation (Pandera + Missingness/Range/Vocab checks)
   ↓
Leakage Checks (Assert target is absent from serve schemas)
   ↓
Shared Feature Builder (Exact same transform pipeline for Train & Serve)
   ↓
DVC Pipeline (Reproducible stages: prepare → train → evaluate)
   ↓
Model Training (HistGradientBoosting / scikit-learn)
   ↓
Evaluation & Quality Gate (ROC-AUC, PR-AUC, Precision@Recall80)
   ↓
MLflow Tracking & Model Registry (Logged parameters, metrics, artifacts, signature)
   ↓
Model Promotion (@champion / @challenger alias assigned if quality gate passes)
   ↓
FastAPI Serving API (/score endpoint loads @champion model)
   ↓
Docker Container (Multi-stage non-root container image)
```

---

## 2. Train vs Serve Feature Contract

To prevent training-serving skew and data leakage, data schemas are strictly governed:

| Field | Type | Train | Serve | Description |
| :--- | :--- | :---: | :---: | :--- |
| `transaction_id` | string | Yes | Yes | Identity only — excluded from model training features |
| `amount` | float | Yes | Yes | Transaction amount in currency units (must be > 0) |
| `merchant_category`| string | Yes | Yes | Merchant sector (e.g. `electronics`, `grocery`, `travel`) |
| `hour_of_day` | int | Yes | Yes | Hour of transaction initiation (0–23) |
| `device_risk` | float | Yes | Yes | Device risk confidence score (0.0–1.0) |
| `is_fraud` | int/bool | **Yes** | **NO** | Target label. **Strictly forbidden at serve time (leakage)** |

The **shared feature builder** (`src/fraud_scoring/features.py`) applies the fixed model-feature order and dtypes for both training and inference. It deliberately has no learned encoding yet; later preprocessing must be fitted once during training and reused at inference.

---

## 3. Repository Layout

```text
paysafe-fraud-scoring/
├── .env.example              # Template for environment secrets and endpoints
├── configs/                  # Environment-specific configuration profiles
│   ├── dev.yaml              # Local development configuration
│   ├── ci.yaml               # Automated CI test configuration
│   └── prod.yaml             # Production settings & strict quality thresholds
├── data/
│   ├── raw/                  # Versioned raw transactional data (.gitignored / DVC tracked)
│   └── processed/            # Engineered datasets & feature metadata
├── docs/                     # Architecture & operational documentation
│   ├── branch-protection.md  # GitHub branch protection policies
│   ├── containers.md         # Container security, multi-stage build, and SBOM
│   ├── design.md             # System lifecycle, roles, and Definition-of-Done
│   └── lineage.md            # DVC data lineage and DAG specifications
├── src/
│   └── fraud_scoring/        # Core package
│       ├── __init__.py
│       ├── api.py            # FastAPI scoring service (/score, /health)
│       ├── config.py         # Type-safe configuration loader (Pydantic + YAML)
│       ├── data_validation.py# Pandera schema checks & leakage guards
│       ├── evaluate.py       # Model evaluation & metric calculation
│       ├── features.py       # Shared train/serve feature transformer
│       ├── model_registry.py # MLflow tracking & alias promotion logic
│       ├── predict.py        # Inference pipeline & model loader
│       └── train.py          # Model training pipeline
├── tests/                    # Unit and integration test suite
│   ├── test_api.py           # API endpoint tests
│   ├── test_config.py        # Configuration validation tests
│   ├── test_data_validation.py # Data quality & leakage tests
│   ├── test_evaluation.py    # Quality gate & metrics tests
│   ├── test_features.py      # Feature engineering consistency tests
│   └── test_train_serve_consistency.py # End-to-end parity validation
├── Dockerfile                # Multi-stage, non-root container build
├── dvc.yaml                  # Reproducible pipeline definition
├── pyproject.toml            # Python packaging & tool configuration
├── requirements.txt          # Production runtime dependencies
└── requirements-dev.txt      # Development & testing dependencies
```

---

## 4. Getting Started

### 4.1 Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Docker (optional, for container deployment)

### 4.2 Virtual Environment & Installation

Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:
```bash
# Runtime package plus development tooling
pip install -e ".[dev]"
```

### 4.3 Dependency Management Strategy
- `pyproject.toml` is the source of truth for direct runtime and development dependencies.
- `requirements.txt` and `requirements-dev.txt` are compatible direct-dependency lists for plain `pip` and container builds.
- `requirements.lock` records the complete resolved environment used by CI and release builds. Refresh it only in a clean virtual environment after intentionally changing dependencies:

  ```bash
  pip install -e ".[dev]"
  pip freeze --all | Out-File -Encoding ascii requirements.lock  # PowerShell
  # pip freeze --all > requirements.lock                          # Linux/macOS
  ```

---

## 5. Configuration & Environments

Configuration is managed via YAML files in `configs/` merged with environment variables:

| Environment | Config File | Model Alias Loaded | Min ROC-AUC Threshold |
| :--- | :--- | :--- | :--- |
| **Development** (`dev`) | `configs/dev.yaml` | `champion` | 0.70 |
| **Continuous Integration** (`ci`) | `configs/ci.yaml` | `challenger` | 0.70 |
| **Production** (`prod`) | `configs/prod.yaml` | `champion` | 0.80 |

To set the active environment:
```bash
# Windows PowerShell
$env:APP_ENV="dev"

# Linux / macOS
export APP_ENV="dev"
```

The loader rejects missing configuration sections and unknown keys. Deployment values may override
the YAML baseline through `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MODEL_REGISTRY_NAME`,
`MODEL_ALIAS`, `API_HOST`, `API_PORT`, `API_WORKERS`, and `LOG_LEVEL`. Keep credentials in the
environment or your deployment secret store; never commit a `.env` file.

---

## 6. Verification & Quality Gates

Run the test suite:
```bash
pytest
```

Run code formatting and lint checks:
```bash
ruff check .
```

Verify configuration loader:
```bash
python -c "from fraud_scoring.config import load_config; cfg = load_config(); print(f'Loaded {cfg.environment} environment configuration successfully')"
```

## 7. Local candidate training and evaluation

Stage 4 uses a logistic-regression baseline because its probability scores and fitted
preprocessing are easy to inspect. The shared feature builder first enforces the
four ordered input columns. A scikit-learn pipeline then fits a numeric scaler,
one-hot encodes the merchant category, and fits the classifier **on training rows
only**. The complete fitted pipeline is saved with `joblib` for later inference;
serving must reuse it rather than fitting another encoder.

The raw CSV is validated before a stratified 80/20 split. The split and model
random seeds are set in `configs/*.yaml`. Training records the dataset SHA-256
and holdout row indices in the candidate artifact. Evaluation rejects a changed
dataset or feature contract, then calculates precision, recall, F1, ROC-AUC,
PR-AUC, precision at 80% recall, and validation class counts.

```bash
python -m fraud_scoring.train --config configs/dev.yaml
python -m fraud_scoring.evaluate --config configs/dev.yaml
```

The candidate is written to `artifacts/model/candidate.joblib` and measured
results to `artifacts/evaluation/metrics.json`. Both directories are ignored by
Git. The evaluation command exits nonzero if any configured threshold is missed.
The dev/CI thresholds are demonstration gates chosen from the measured 5,000-row
synthetic sample; they are not business acceptance criteria. Production has
stricter configured thresholds and must be calibrated with real reviewed data.
Passing this local gate makes a candidate eligible for a later approval process;
it does not register, promote, or deploy a model.
