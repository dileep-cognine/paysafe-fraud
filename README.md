# PaySafe Fraud Scoring MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/data-DVC-945DD6.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

An incremental MLOps assessment project for transaction fraud scoring. The current implementation covers synthetic data validation, shared train/serve features, candidate training, evaluation gates, MLflow experiment tracking, and explicit gated registry promotion. Serving and deployment are planned later stages.

---

## 1. System Architecture & Flow

The planned lifecycle is shown below; this repository currently implements through explicit model promotion:

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

The **shared feature builder** (`src/fraud_scoring/features.py`) applies the fixed model-feature order and dtypes for both training and inference. Learned encoding is fitted inside the training pipeline and saved with the classifier for later inference.

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

| Environment | Config File | Configured Model Alias | Min ROC-AUC Threshold |
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
Passing this local gate makes a candidate eligible for a separate approval and
promotion command; training alone does not register, promote, or deploy a model.

## 8. MLflow experiment tracking

`python -m fraud_scoring.train --config configs/dev.yaml` now trains, evaluates,
applies the existing quality gate, and records the complete candidate experiment
in MLflow. The standalone evaluation command remains available to recheck a
saved candidate. The run is recorded even when the gate fails; in that case the
training command exits nonzero after logging. Training itself never registers or
promotes a model.

The tracking URI and experiment name come from the selected YAML profile or the
`MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` environment overrides. With
the current `.env.example`, local runs use `sqlite:///mlruns.db` and the
`fraud-scoring-dev` experiment. If no override is set, `configs/dev.yaml` uses
`paysafe-fraud-scoring-dev` instead. Start the local UI from the repository root:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`. If you override the tracking URI, pass that
same URI to `--backend-store-uri` so the UI reads the same store.

Each run records model type and hyperparameters, seed, split fraction, feature
names, and preprocessing; the measured evaluation metrics and class/row counts;
the actual dataset path, byte size, row count, and raw-file SHA-256; and the
saved `metrics.json`. MLflow also records the input dataset metadata. The model
artifact contains the fitted scaler, one-hot encoder, and classifier. Its
signature and input example are inferred from a real validated synthetic row
passed through the shared serving feature builder: `amount`,
`merchant_category`, `hour_of_day`, and `device_risk`. The model output is the
two class probabilities; the fraud risk score is the second probability.
`transaction_id` and `is_fraud` are absent from the model input.

The `quality_gate_status` run tag is `PASS` or `FAIL`; failed threshold details
are tagged when applicable. A `git_commit` tag is added only when Git returns an
actual commit hash. The SHA-256 tag fingerprints the raw bytes; it is not a
fabricated DVC version. MLflow may display the candidate under its logged-model
section; that alone does not create a registered production model or alias.

## 9. Explicit model promotion and rollback

Training creates a **candidate**: a fitted model under evaluation. An MLflow
**run** holds the actual metrics and gate result. A **registered model** is the
configured name that groups approved **versions**. The mutable **alias**
(`champion` in the dev profile) points to one approved version. Training never
moves that alias. The current implementation does not serve a model yet.

After reviewing a completed run, an authorized model owner can promote its
printed run ID and logged-model URI. Use the same configuration profile and
tracking store as the training command:

```bash
python -m fraud_scoring.model_registry promote --config configs/dev.yaml --run-id <RUN_ID> --model-uri models:/<LOGGED_MODEL_ID>
```

The command checks that the run finished in the configured experiment and
environment, has a `PASS` tag, and has all six recorded metrics meeting the
currently configured Stage 4 thresholds. It also checks that the logged model
belongs to that run and is ready. A failed check exits without registering or
moving the alias. An eligible model is registered under `MODEL_REGISTRY_NAME`,
then the configured `MODEL_ALIAS` is assigned to its version. Repeating the
command reuses the version; older versions remain in the registry. Version
tags include the training run ID, gate status, environment, model type, and a
Git commit only when one exists.

To roll back, find the earlier approved version's `training_run_id` tag and
logged-model ID in MLflow, review its metrics, then run the same `promote`
command with that run ID and its original `models:/m-...` URI. The gate is
checked again against the selected profile's current thresholds. The alias
moves back to the existing version without deleting the newer version. See
`docs/branch-protection.md` for proposed approval roles and ownership.
