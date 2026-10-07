# Assessment checklist

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Project configuration and dependency management | `pyproject.toml`, YAML profiles, direct requirements and lock file | `python -m pytest tests/test_config.py` |
| Synthetic, documented data | `generate_data.py`, `data/README.md` | `python -m fraud_scoring.generate_data --records 5000` |
| Schema and data-quality checks | `data_validation.py` with Pandera and explicit checks | `python -m fraud_scoring.data_validation --mode train` |
| Leakage prevention | fixed feature lists, denylist, and tests | `pytest tests/test_data_validation.py` |
| Shared train/serve feature builder | `features.py` used by training and prediction | `pytest tests/test_train_serve_consistency.py` |
| DVC reproducible pipeline | `dvc.yaml` declares `prepare → train → evaluate`; raw data is DVC-tracked | `dvc repro`, `dvc dag`, `dvc status`, `dvc push` |
| Training and evaluation gates | `train.py`, `evaluate.py`, configured thresholds | `python -m fraud_scoring.train --config configs/dev.yaml` |
| MLflow tracking | run parameters, metrics, data input, artifact, signature, example | `pytest tests/test_mlflow_tracking.py` |
| Registry and explicit alias promotion | `model_registry.py`; no automatic promotion | `pytest tests/test_model_registry.py` |
| FastAPI scoring | `/health`, `/model-info`, `/score`, `/docs` | `pytest tests/test_api.py` |
| Docker build and non-root runtime | multi-stage `Dockerfile`, `.dockerignore`, `appuser` | `docker build --tag paysafe-fraud-scoring:local .` |
| Scanning and SBOM awareness | documented in `docs/containers.md` | **PARTIAL:** run scanner/SBOM command in release environment |
| CI | `.github/workflows/ci.yml` checks format, lint, type, test, data, Docker, secrets | GitHub Actions after push/PR |
| Branch protection | manual GitHub settings in `docs/branch-protection.md` | **TODO:** maintainer must enable repository settings |
| Final walkthrough | README, design, lineage, and this checklist | Follow README walkthrough sequence |

No deployment, cloud registry push, automatic retraining, automatic promotion,
or production monitoring is claimed by this assessment repository.
