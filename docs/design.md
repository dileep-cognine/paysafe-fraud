# PaySafe Fraud Scoring design

## Lifecycle and repository layout

The authorization-time lifecycle is raw transactions → schema/data-quality and
leakage gates → shared features → DVC `prepare` → `train` → `evaluate` → MLflow
experiment → quality-gated registration → approved `@champion` alias → FastAPI
`/score`. The project keeps source in `src/fraud_scoring`, profiles in `configs`,
tests in `tests`, DVC data metadata in `data`/`.dvc`, and operational guidance in
`docs`.

The shared contract uses `amount`, `merchant_category`, `hour_of_day`, and
`device_risk`. `transaction_id` is identity-only; `is_fraud` is training-only.
Pandera and explicit checks reject invalid, unexpected, target-derived, and
post-authorization fields before training or serving.

## Ownership and controls

The Data/ML pipeline owner maintains the dataset contract and DVC stages. The
training/evaluation owner reviews measured metrics and experiment evidence. The
model reviewer/promoter approves an eligible run and moves the configured alias.
The API/service owner operates the service using that alias and its pinned
version. These roles describe responsibilities, not named people or automated
permissions.

Git stores code, configurations, tests, DVC metadata, and documentation. DVC
stores the raw-data reference and reproduces validated data, candidates, and
metrics. MLflow stores experiment and registry metadata. YAML profiles provide
development, CI, and production baselines; environment variables provide
deployment-specific values and secrets. `.env` is ignored and never copied into
the container.

## Training, serving, and operations

The DVC training path writes a local candidate and applies the configured gate.
The explicit MLflow command logs the same real lifecycle, including metrics,
dataset fingerprint, feature contract, confusion matrix, signature, and input
example. Only a passing run can be registered and assigned `@champion`.

FastAPI resolves the alias once at startup and returns its pinned version with
each real risk score. `/health` and `/model-info` expose readiness and
non-sensitive model metadata. Structured logs record model version and request
duration without transaction values. This is light operational monitoring; it
does not claim drift detection, automated retraining, or production alerting.

CI installs locked dependencies, formats, lints, type-checks, tests, validates
data, reproduces DVC stages, uploads reports/metrics, builds the Docker image,
and scans committed history for secrets. The multi-stage Docker image uses a
pinned base and non-root user; it requires a reachable MLflow registry at
runtime and does not contain a model or credentials.

## Definition of Done

A change is ready for review when the contract and leakage controls remain
intact; formatting, linting, typing, tests, data validation, and DVC reproduction
pass; CI builds the container; and documentation describes actual behavior.
Promotion additionally requires a real finished MLflow run whose recorded
metrics pass the currently selected thresholds.

## Trade-off

The small logistic-regression baseline keeps preprocessing, probability output,
and train/serve parity explainable. A production rollout would compare stronger
models and calibrate thresholds with reviewed business costs and representative
data.
