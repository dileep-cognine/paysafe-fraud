# PaySafe Fraud Scoring MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/data-DVC-945DD6.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

An assessment-ready MLOps system for transaction fraud scoring. It validates
synthetic historical transactions, prevents leakage, trains and evaluates a
candidate, tracks it in MLflow, promotes only approved versions, serves the
configured champion through FastAPI, packages the service in Docker, and offers
an optional Streamlit HTTP client.

---

## 1. System Architecture & Flow

The implemented lifecycle is:

```text
Raw Data (CSV)
   â†“
Schema & Data Quality Validation (Pandera + Missingness/Range/Vocab checks)
   â†“
Leakage Checks (Assert target is absent from serve schemas)
   â†“
Shared Feature Builder (Exact same transform pipeline for Train & Serve)
   â†“
DVC validation stage (reproducible raw-data gate)
   â†“
Model Training (logistic regression / scikit-learn)
   â†“
Evaluation & Quality Gate (ROC-AUC, PR-AUC, Precision@Recall80)
   â†“
MLflow Tracking & Model Registry (Logged parameters, metrics, artifacts, signature)
   â†“
Model Promotion (@champion / @challenger alias assigned if quality gate passes)
   â†“
FastAPI Serving API (/score endpoint loads @champion model)
   â†“
Docker Container (Multi-stage non-root container image)
```

---

## 2. Train vs Serve Feature Contract

To prevent training-serving skew and data leakage, data schemas are strictly governed:

| Field | Type | Train | Serve | Description |
| :--- | :--- | :---: | :---: | :--- |
| `transaction_id` | string | Yes | Yes | Identity only â€” excluded from model training features |
| `amount` | float | Yes | Yes | Transaction amount in currency units (must be > 0) |
| `merchant_category`| string | Yes | Yes | Merchant sector (e.g. `electronics`, `grocery`, `travel`) |
| `hour_of_day` | int | Yes | Yes | Hour of transaction initiation (0â€“23) |
| `device_risk` | float | Yes | Yes | Device risk confidence score (0.0â€“1.0) |
| `is_fraud` | int/bool | **Yes** | **NO** | Target label. **Strictly forbidden at serve time (leakage)** |

The **shared feature builder** (`src/fraud_scoring/features.py`) applies the fixed model-feature order and dtypes for both training and inference. Learned encoding is fitted inside the training pipeline and saved with the classifier for later inference.

---

## 3. Repository Layout

```text
paysafe-fraud-scoring/
â”œâ”€â”€ .env.example              # Template for environment secrets and endpoints
â”œâ”€â”€ configs/                  # Environment-specific configuration profiles
â”‚   â”œâ”€â”€ dev.yaml              # Local development configuration
â”‚   â”œâ”€â”€ ci.yaml               # Automated CI test configuration
â”‚   â””â”€â”€ prod.yaml             # Production settings & strict quality thresholds
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                  # Versioned raw transactional data (.gitignored / DVC tracked)
â”‚   â””â”€â”€ processed/            # Engineered datasets & feature metadata
â”œâ”€â”€ docs/                     # Architecture & operational documentation
â”‚   â”œâ”€â”€ branch-protection.md  # GitHub branch protection policies
â”‚   â”œâ”€â”€ containers.md         # Container security, multi-stage build, and SBOM
â”‚   â”œâ”€â”€ design.md             # System lifecycle, roles, and Definition-of-Done
â”‚   â””â”€â”€ lineage.md            # DVC data lineage and DAG specifications
â”œâ”€â”€ src/
â”‚   â””â”€â”€ fraud_scoring/        # Core package
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ api.py            # FastAPI scoring service (/score, /health)
â”‚       â”œâ”€â”€ config.py         # Type-safe configuration loader (Pydantic + YAML)
â”‚       â”œâ”€â”€ data_validation.py# Pandera schema checks & leakage guards
â”‚       â”œâ”€â”€ evaluate.py       # Model evaluation & metric calculation
â”‚       â”œâ”€â”€ features.py       # Shared train/serve feature transformer
â”‚       â”œâ”€â”€ model_registry.py # MLflow tracking & alias promotion logic
â”‚       â”œâ”€â”€ predict.py        # Inference pipeline & model loader
â”‚       â””â”€â”€ train.py          # Model training pipeline
â”œâ”€â”€ tests/                    # Unit and integration test suite
â”‚   â”œâ”€â”€ test_api.py           # API endpoint tests
â”‚   â”œâ”€â”€ test_config.py        # Configuration validation tests
â”‚   â”œâ”€â”€ test_data_validation.py # Data quality & leakage tests
â”‚   â”œâ”€â”€ test_evaluation.py    # Quality gate & metrics tests
â”‚   â”œâ”€â”€ test_features.py      # Feature engineering consistency tests
â”‚   â””â”€â”€ test_train_serve_consistency.py # End-to-end parity validation
â”œâ”€â”€ Dockerfile                # Multi-stage, non-root container build
â”œâ”€â”€ dvc.yaml                  # Reproducible pipeline definition
â”œâ”€â”€ pyproject.toml            # Python packaging & tool configuration
â”œâ”€â”€ requirements.txt          # Production runtime dependencies
â””â”€â”€ requirements-dev.txt      # Development & testing dependencies
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

Install the optional Streamlit client separately when needed:

```bash
pip install -e ".[ui]"
```

### 4.3 Dependency Management Strategy
- `pyproject.toml` is the source of truth for direct runtime and development dependencies.
- `requirements.txt` and `requirements-dev.txt` are compatible direct-dependency lists for plain `pip` and container builds.
- `requirements.lock` is a pip constraints file containing the resolved environment used by CI and container builds. Refresh it only in a clean virtual environment after intentionally changing dependencies:

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

Run code formatting, lint, and type checks:
```bash
python -m ruff format --check src tests
python -m ruff check .
python -m mypy src
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

### DVC reproducible pipeline

DVC versions the raw dataset separately from Git and declares the complete
`prepare → train → evaluate` pipeline in `dvc.yaml`. The preparation stage
validates and deterministically normalizes the DVC-tracked raw CSV; training
creates the candidate artifact without starting an MLflow run; evaluation
creates the measured `metrics.json` and applies the configured quality gate.

Initialize DVC once in the repository:

```bash
dvc init
dvc remote add -d local ../paysafe-fraud-dvc-storage
```

```bash
dvc repro
```

## 8. MLflow experiment tracking

`python -m fraud_scoring.train --config configs/dev.yaml` trains and saves a
local candidate for the reproducible DVC pipeline. To run that same real
training and evaluation lifecycle while recording an MLflow experiment, use:

```bash
python -m fraud_scoring.mlflow_tracking --config configs/dev.yaml
```

The tracking command records a run even when the gate fails, then exits nonzero.
It logs the feature contract, measured metrics, confusion matrix, dataset
reference, signature, and input example. Neither command registers or promotes
a model.

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
moves that alias. The API in Stage 7 serves the approved version; it never
loads an unpromoted training candidate.

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

## 10. Local FastAPI scoring service

Start the service from the repository root after promoting a passing candidate:

```bash
python -m fraud_scoring.api --config configs/dev.yaml
```

The command uses `API_HOST`, `API_PORT`, `API_WORKERS`, and `LOG_LEVEL` from the
selected YAML profile and environment overrides. The development port defaults
to 8000. If it is occupied, set `API_PORT` to an unused port before starting.
For local-only access, set `API_HOST=127.0.0.1`. Visit `/docs` on that host and
port for the generated Swagger UI.

At startup, the API resolves `models:/<MODEL_REGISTRY_NAME>@<MODEL_ALIAS>` in
MLflow, checks the version's passing gate tag, and loads the complete fitted
scikit-learn pipeline. It pins the resolved version in memory. A missing or
unloadable champion aborts startup; `/health` reports model readiness. Moving
the alias later does not change an already running process: restart the API to
load the newly approved version. Each response reports the version that process
actually loaded.

`POST /score` accepts `transaction_id` (identity only), `amount` (> 0), one of
the allowed `merchant_category` values, `hour_of_day` (integer 0â€“23), and
`device_risk` (0â€“1). Extra fields, including `is_fraud`, are rejected. The
request passes through the shared serving feature builder; only its four
non-identity features reach the fitted pipeline. `risk_score` is the model's
probability for fraud class 1. No training label is returned.

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"txn-1001","amount":125.50,"merchant_category":"electronics","hour_of_day":14,"device_risk":0.23}'
```

The response has numeric `risk_score` (0â€“1) and string `model_version` fields.
Invalid requests return 422; an unavailable model returns 503; unexpected
prediction failures return a generic 500 while details stay in server logs.
Logs record the model version and request duration without transaction values.

`GET /health` reports whether the champion loaded. `GET /model-info` returns the
non-secret active model name, alias, pinned version, and environment. `/docs`
provides the generated OpenAPI interface.

### Merchant category normalization

`merchant_category` is normalized in the shared feature contract before the
fitted encoder receives it. Known formatting and aliases use the centralized
deterministic mapping: for example, `Phone Shop` and `mobile-store` become
`electronics`, while `restaurant` becomes `dining`. Training applies those
same deterministic mappings before data-quality validation, and `/score` uses
the same component before inference. This prevents train/serve skew without
changing the estimator input columns or the `/score` response schema.

The approved model vocabulary remains `grocery`, `electronics`, `fashion`,
`travel`, `gaming`, `dining`, `crypto`, and `utilities`. No new category is
accepted at runtime because the fitted one-hot encoder was trained on that
contract. Book-related labels currently take the safe unknown path: there is
no approved books/retail category in the assessment dataset. The API returns
422 before prediction for such a value until a business owner approves a
mapping and the model is retrained if the vocabulary changes.

Optional LLM classification is disabled by default. Set
`LLM_CATEGORY_MAPPING_ENABLED=true`, configure
`LLM_CATEGORY_MAPPING_MODEL`, `LLM_CATEGORY_MAPPING_BASE_URL`, and
`LLM_CATEGORY_MAPPING_TIMEOUT_SECONDS`. The local Ollama provider receives
only the normalized category and allowed list;
it is instructed to return structured JSON with one allowed category or
`unknown`. Its response is validated again in the application. Invalid JSON,
timeouts, missing credentials, provider errors, and categories outside the
approved list all take the safe unknown path and never reach the model. Install
Ollama locally and pull the selected model before enabling it.

In one local verification on 2026-09-25, the approved synthetic-data model
returned `{"risk_score":0.31914670174006454,"model_version":"2"}` for the
request above. Ten sequential local HTTP requests had a 16.95 ms median and
185.87 ms maximum observed latency (also the nearest-rank p95 for 10 samples).
These are local measurements, not a production latency guarantee. The value
and version will change when the approved alias points to a different model.

## 11. Production-style Docker image

Build the Stage 8 image with `docker build --tag paysafe-fraud-scoring:latest .`.
It uses a pinned Python 3.13.13 slim Bookworm image, multi-stage wheel build,
non-root `appuser`, a `/health` Docker health check, and a `.dockerignore` that
excludes local data, MLflow state, tests, source-control files, virtual
environments, and `.env` files. It never embeds a model, database, or secret.

The image starts the same FastAPI factory without reload and reads its runtime
configuration from environment variables. Supply a reachable MLflow registry
URI and the configured model name and alias when running it. The image resolves
the approved alias at startup; it cannot use the host's local SQLite tracking
store as a production registry. The local image was built at 470.1 MB and
verified to run as non-root `appuser`. See [container instructions](docs/containers.md)
for the build, run, health, non-root, image-size, and scan commands.

## 12. Development workflow and CI

Use short-lived branches and pull requests for every change:

```text
feature branch → pull request → CI → review → main
```

1. Create a descriptive branch such as `feature/api` or `fix/model-loading`.
2. Implement the change and run the local checks below.
3. Push the branch and open a pull request targeting `main`.
4. GitHub Actions runs formatting, linting, type checks, tests, data validation,
   Docker build validation, and secret hygiene checks.
5. Obtain review and merge only after the required checks pass.

```bash
python -m ruff format --check src tests
python -m ruff check .
python -m mypy src
python -m pytest
python -m fraud_scoring.data_validation --data-path data/raw/transactions.csv --mode train
docker build --tag paysafe-fraud-scoring:local .
```

The CI workflow runs for pull requests targeting `main` and pushes to `main`.
It uses temporary local MLflow stores in tests and has no personal or production
credentials. It builds an image for validation only; it does not push or deploy
the image. Model promotion remains separate: `training → evaluation → quality
gate → explicit authorized promotion`. See [branch-protection guidance](docs/branch-protection.md)
for the GitHub settings a repository maintainer should configure manually.

## 13. Optional Streamlit client

The UI is a client of FastAPI only: it does not load MLflow models, perform
feature engineering, train, or promote models. Start a healthy API first, then:

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

It calls `GET /health`, `GET /model-info`, and `POST /score`. Set
`FRAUD_SCORING_API_URL` when the API is not at `http://127.0.0.1:8000`. The UI
shows returned score and version exactly as supplied by the API. Its low/medium/
high display labels are visual guidance only, not fraud-policy thresholds.

## 14. Assessment walkthrough

1. Show `data/README.md`, `data_validation.py`, and its validation command.
2. Show `features.py` and the train/serve consistency test.
3. Run training, inspect the generated evaluation metrics, and explain the gate.
4. Open MLflow to inspect the actual run, artifacts, and model signature.
5. Promote only an eligible model with the explicit registry command.
6. Start FastAPI and call `/health`, `/model-info`, and `/score`.
7. Show the optional Streamlit client calling the API.
8. Build the Docker image and explain its non-root runtime and external MLflow requirement.
9. Show CI, branch-protection guidance, `docs/lineage.md`, and the assessment checklist.

