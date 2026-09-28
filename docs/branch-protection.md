# Proposed model-promotion ownership

These are project roles for the assessment, not claims about actual organization permissions.

- An ML Engineer / Model Owner reviews the completed training run, evaluation metrics, and dataset reference before requesting promotion.
- A Reviewer / Maintainer approves promotion and the movement of the configured alias. Registry write access should be limited to this role or a specifically authorized CI service identity.
- Ordinary training jobs only log candidates. They do not receive authority to move `champion`; a passing quality gate makes a run eligible, not automatically approved.

The separate `python -m fraud_scoring.model_registry promote` command enforces the recorded gate and current configured thresholds before registration or alias movement. This code gate does not replace access control in a real shared MLflow server; the server's credentials and permissions must be configured by its operator.

For rollback, identify the earlier approved model version and its `training_run_id` in MLflow, locate that run's logged-model URI, and invoke the same promotion command with those original identifiers. It rechecks the gate and points the alias back to the retained version. Do not delete the newer version; preserve it for investigation and audit.
