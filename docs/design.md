# PaySafe Fraud Scoring design

## Architecture and data flow

The system scores authorization-time transactions. A raw CSV first passes
Pandera schema checks, explicit quality checks, and leakage guards. The shared
feature builder then produces the fixed four-column model matrix: `amount`,
`merchant_category`, `hour_of_day`, and `device_risk`. `transaction_id` is
identity-only and `is_fraud` is a training target that never reaches serving.

The DVC validation stage makes the ingestion gate reproducible. Training fits a
scikit-learn pipeline containing numeric scaling, merchant-category one-hot
encoding, and logistic regression. Evaluation measures precision, recall, F1,
ROC-AUC, PR-AUC, and precision at 80% recall against configuration thresholds.

## Training and promotion

The DVC `train` stage writes a local candidate and the `evaluate` stage applies
the configured quality gate. The explicit MLflow tracking command runs the same
real lifecycle and records its candidate, metrics, data reference, feature
contract, confusion matrix, model signature, and input example. A passing
evaluation makes a candidate eligible; it does not deploy it. An authorized
owner explicitly promotes the logged MLflow model only after the promotion code
rechecks the gate and current thresholds. The configured `champion` alias then
identifies the approved registered version.

## Serving

FastAPI resolves the configured champion alias once at startup and pins that
version in memory. `/score` validates authorization-time input, uses the same
feature builder, and returns the actual fraud probability and pinned model
version. `/health` and `/model-info` report service state without exposing
secrets. The optional Streamlit page is only an HTTP client of these endpoints.

## Environment and security

YAML profiles define development, CI, and production baselines. Environment
variables override deployment-specific values; `.env` is ignored and is not
copied into the Docker image. The image uses a pinned Python base, multi-stage
build, `.dockerignore`, and a non-root runtime user. CI has no deployment or
registry credentials and does not promote models.

## Repository responsibilities and completion criteria

Git stores source code, configuration profiles, tests, DVC metadata, and
documentation. DVC stores the raw dataset reference and reproduces derived
data, local candidates, and metrics. MLflow stores experiment and registry
metadata; it is not a substitute for DVC data versioning. A change is ready for
review when formatting, linting, typing, tests, the raw-data gate, the DVC
pipeline, and Docker build checks pass in CI. Promotion is a separate reviewed
operation and requires a real passing MLflow run.

## Trade-off

The project uses a small, interpretable logistic-regression baseline instead of
a more complex fraud model. It makes preprocessing, probability output, and
training-serving parity easy to explain in an assessment. Real production work
would compare stronger models and calibrate thresholds with reviewed business
costs and representative data.
