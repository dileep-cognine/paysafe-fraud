# Git workflow and branch protection

This document describes recommended settings for the `main` branch. It does
not claim that those GitHub repository settings are enabled.

## Required by assessment

Use a trunk-based workflow:

```text
main
  └── short-lived feature or fix branch
          ↓
          Pull Request
          ↓
          CI
          ↓
          review
          ↓
          main
```

Create descriptive, short-lived branches such as `feature/data-validation`,
`feature/model-training`, `feature/api`, or `fix/model-loading`. Do not use a
long-lived development branch. Open a pull request to `main`; the `CI` workflow
runs formatting, linting, type checks, the complete test suite, raw-data
validation, a Docker build, and secret hygiene checks.

Configure GitHub branch protection for `main` manually in **Settings → Branches**:

1. Require a pull request before merging.
2. Require the `Format, lint, type check, test, and validate data`, `Build Docker image`,
   and `Verify secret hygiene` checks to pass.
3. Require at least one approving code review.
4. Dismiss stale approvals when new commits are pushed.
5. Require resolved review conversations.
6. Restrict direct pushes to `main` to repository maintainers, or disable them entirely.

## Recommended project practice

Require the branch to be up to date before merge and use GitHub's merge queue
when the repository has multiple active contributors. Configure the required
status checks only after the workflow has run at least once so their exact names
are available in the GitHub UI.

`.env` and other `*.env` files are ignored by Git; `.env.example` is the only
committed environment template. CI uses no MLflow, provider, registry, cloud,
or deployment credentials. The `gitleaks` CI check scans committed history, and
Docker image scanning remains a release/deployment review step documented in
`docs/containers.md` (`trivy image <tag>`).

## Model promotion is separate

CI verifies code only. It never trains a production model, writes to a registry,
or moves an alias. Training creates a candidate; evaluation applies the quality
gate; then an authorized reviewer runs the explicit promotion command. An
arbitrary feature branch and a merge to `main` cannot automatically promote a
model to `champion`.

An ML Engineer / Model Owner reviews the completed training run, its evaluation
metrics, and dataset reference before requesting promotion. A Reviewer /
Maintainer approves promotion and alias movement. Registry write access should
be limited to this role or a separately authorized CI/CD identity. The promotion
command rechecks the recorded gate and current thresholds; rollback uses the
same command with the earlier approved version's original identifiers.
