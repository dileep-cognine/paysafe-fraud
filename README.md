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

### 4.1 Clone the repository

```bash
git clone https://github.com/dileep-cognine/paysafe-fraud.git
cd paysafe-fraud
```

### 4.2 Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Docker (optional, for container deployment)

### 4.3 Virtual Environment & Installation

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
python -m pip install --constraint requirements.lock -e ".[dev]"
```

Install the optional Streamlit client separately when needed:

```bash
python -m pip install --constraint requirements.lock -e ".[ui]"
```

### 4.4 Local configuration

Copy the non-secret local template before overriding any settings:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

`.env` is ignored by Git. Keep credentials in your environment or secret store;
the committed template contains only local defaults and placeholders.

### 4.5 Dependency Management Strategy
- `pyproject.toml` is the source of truth for direct runtime and development dependencies.
- `requirements.txt` and `requirements-dev.txt` are compatible direct-dependency lists for plain `pip` and container builds.
- `requirements.lock` is a pip constraints file containing the resolved environment used by CI and container builds. Refresh it only in a clean virtual environment after intentionally changing dependencies:

  ```bash
  python -m pip install --constraint requirements.lock -e ".[dev]"
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

## 6. End-to-end local runbook

Run the following from the repository root after completing the setup in
Section 4. The commands use the current HTTP MLflow configuration and do not
put the local database, artifacts, or secrets into Git.

### 6.1 Start the MLflow server

Start this in its own terminal and leave it running while you track, promote,
or serve a model. The server owns the local SQLite backend; application clients
connect to it over HTTP.

```powershell
.\venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlruns.db --serve-artifacts --artifacts-destination ./mlruns --host 0.0.0.0 --port 5000 --workers 1 --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
```

Open `http://localhost:5000` to inspect experiments, runs, registered models,
and aliases. If port 5000 is unavailable, choose another free port and set the
same value for `MLFLOW_TRACKING_URI` in `.env`, the local shell, and Docker.

### 6.2 Restore or generate the synthetic data

The repository is already initialized for DVC. If the configured local remote
is available, restore the tracked raw data:

```powershell
dvc pull
```

If no DVC remote is available for a fresh assessment checkout, generate the
documented synthetic demonstration data instead:

```powershell
python -m fraud_scoring.generate_data --output data/raw/transactions.csv --records 5000 --fraud-ratio 0.04 --seed 42
```

Generating data changes the raw-data dependency, so use it only when the
tracked dataset is unavailable or when intentionally refreshing the demo data.

### 6.3 Validate data and reproduce the DVC pipeline

Run the data-quality and leakage gate directly:

```powershell
python -m fraud_scoring.data_validation --data-path data/raw/transactions.csv --mode train
```

Then run the reproducible pipeline. It creates the validated dataset, a local
candidate artifact, and evaluation metrics. It does not create an MLflow run or
promote a model.

```powershell
dvc status
dvc repro
dvc status
dvc dag
```

The equivalent explicit stage commands are useful when debugging one stage:

```powershell
python -m fraud_scoring.prepare --input-path data/raw/transactions.csv --output-path data/processed/validated.csv
python -m fraud_scoring.train --config configs/dev.yaml --data-path data/processed/validated.csv --output-path artifacts/model/candidate.joblib
python -m fraud_scoring.evaluate --config configs/dev.yaml --data-path data/processed/validated.csv --model-path artifacts/model/candidate.joblib --output-path artifacts/evaluation/metrics.json
```

### 6.4 Track a real experiment and promote only a passing run

Run tracked training after the MLflow server is available. Copy the two values
printed by the command; they are real values created by that execution.

```powershell
$env:APP_ENV = "dev"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m fraud_scoring.mlflow_tracking --config configs/dev.yaml
```

If the quality gate passes, promote that exact run and logged-model URI. Do not
replace the placeholders with an invented ID or URI.

```powershell
python -m fraud_scoring.model_registry promote --config configs/dev.yaml --run-id <RUN_ID_PRINTED_BY_TRACKING> --model-uri <MODEL_URI_PRINTED_BY_TRACKING>
```

The promotion command exits without moving the alias if the run fails its
quality gate. Inspect the real run in MLflow before retrying or promoting.

### 6.5 Start and verify the local API

The API requires an approved champion alias in the same MLflow registry. It
loads the resolved version once at startup.

```powershell
$env:APP_ENV = "dev"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m fraud_scoring.api --config configs/dev.yaml
```

In another terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
$body = @{transaction_id='demo-001'; amount=125.5; merchant_category='grocery'; hour_of_day=14; device_risk=0.2} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/score -Method Post -ContentType application/json -Body $body
```

The request must not include `is_fraud`. Unknown fields and invalid feature
values are rejected with HTTP 422.

### 6.6 Build and run the Docker API

Keep the MLflow server from Section 6.1 running. Docker Desktop exposes the
host service through `host.docker.internal`; the image receives that URI at
runtime and never bundles the registry or model artifact.

```powershell
docker build -t paysafe-fraud-scoring:v1 .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 paysafe-fraud-scoring:v1
```

The same API deployment can be started with Compose after the MLflow server is
running:

```powershell
docker compose up --build
```

Compose defaults to `http://host.docker.internal:5000` for the external MLflow
server. See [container instructions](docs/containers.md#docker-compose) for
environment overrides and shutdown commands.

Use the same verification requests from Section 6.5. To run the local API and
Docker API together, use `-p 18000:8000` for Docker and substitute port 18000
in the verification URLs. On native Linux Docker Engine, append
`--add-host=host.docker.internal:host-gateway` to `docker run`.

### 6.7 Move the runnable local setup to another laptop

Git and DVC do not contain the local MLflow registry or artifacts. To move the
same approved champion, stop MLflow and all API clients on the source machine,
then transfer `mlruns.db` and the complete `mlruns/` directory to the root of
the destination clone. Start the MLflow server in Section 6.1 on the destination
machine, then use the local or Docker commands above. Keep this state out of
Git. Alternatively, run the real tracking and gated-promotion workflow there to
create a new approved version.

## 7. Verification & Quality Gates

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

## 8. Local candidate training and evaluation

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

The repository is already initialized for DVC. Its committed configuration uses
the local remote `../paysafe-fraud-dvc-storage`; do not run `dvc init` again in
a clone. Restore data when that remote is available, then inspect and reproduce
the pipeline:

```bash
dvc pull
dvc status
dvc repro
```

To move from the assessment's local remote to object storage later, configure
an approved remote outside this repository, for example
`dvc remote add -d production s3://<bucket>/<prefix>` or
`dvc remote add -d production gs://<bucket>/<prefix>`, authenticate through the
deployment environment, then run `dvc push`. S3/GCS is not currently configured.

## 9. MLflow experiment tracking

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
the current `.env.example`, local runs use `http://localhost:5000` and the
`fraud-scoring-dev` experiment. Without an experiment-name override,
`configs/dev.yaml` uses `paysafe-fraud-scoring-dev`. Start the tracking server
from the repository root before running tracking, promotion, or inference:

```powershell
.\venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlruns.db --serve-artifacts --artifacts-destination ./mlruns --host 0.0.0.0 --port 5000 --workers 1 --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
```

Open `http://localhost:5000` for the UI. SQLite is the server's backend;
application clients use HTTP. Do not pass the HTTP tracking URL as the server's
`--backend-store-uri`. The server proxies `mlflow-artifacts:/` references to
files under `./mlruns`.

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

## 10. Explicit model promotion and rollback

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

## 11. Local FastAPI scoring service

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

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
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

Run the commands above against your own approved alias and record the returned
model version and score as release evidence. Scores, latency, and model versions
are not fixed documentation values; they depend on the approved model and runtime.

## 12. Production-style Docker image

Build the Stage 8 image with `docker build --tag paysafe-fraud-scoring:latest .`.
It uses a pinned Python 3.13.13 slim Bookworm image, multi-stage wheel build,
non-root `appuser`, a `/health` Docker health check, and a `.dockerignore` that
excludes local data, MLflow state, tests, source-control files, virtual
environments, and `.env` files. It never embeds a model, database, or secret.

The image starts the same FastAPI factory without reload and reads its runtime
configuration from environment variables. Supply a reachable MLflow registry
URI and the configured model name and alias when running it. The image resolves
the approved alias at startup; it cannot use the host's local SQLite tracking
store as a production registry. See [container instructions](docs/containers.md)
for the build, run, health, non-root, image-size, and scan commands. Record
actual image size and runtime evidence in the release review rather than
treating documentation examples as evidence.

CI generates an SPDX JSON SBOM from its exact final image and uploads it as a
workflow artifact. Local SBOM commands and Syft installation guidance are in
[container instructions](docs/containers.md#sbom-generation).

CI also scans that same final image with Trivy. HIGH and CRITICAL findings fail
the Docker job, and its JSON result is retained as a workflow artifact. See the
[container vulnerability-scanning commands](docs/containers.md#image-vulnerability-scanning).

Use the complete commands in the [end-to-end local runbook](#6-end-to-end-local-runbook).
The Docker image connects to the host MLflow server at runtime; it never embeds
the database or model artifacts. See [container instructions](docs/containers.md)
for image security and release checks.

## 13. Development workflow and CI

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

## 14. Optional Streamlit client

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

## 15. Assessment walkthrough

1. Show `data/README.md`, `data_validation.py`, and its validation command.
2. Show `features.py` and the train/serve consistency test.
3. Run training, inspect the generated evaluation metrics, and explain the gate.
4. Open MLflow to inspect the actual run, artifacts, and model signature.
5. Promote only an eligible model with the explicit registry command.
6. Start FastAPI and call `/health`, `/model-info`, and `/score`.
7. Show the optional Streamlit client calling the API.
8. Build the Docker image and explain its non-root runtime and external MLflow requirement.
9. Show CI, branch-protection guidance, `docs/lineage.md`, and the assessment checklist.

