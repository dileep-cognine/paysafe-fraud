# Container runtime

The Stage 8 image packages only the FastAPI service, its runtime dependencies,
and the committed configuration profiles. It uses the pinned
`python:3.13.13-slim-bookworm` official image digest because that Python version
matches the validated project environment while the slim Debian image keeps the
runtime base small. The Dockerfile has a builder stage that produces package
wheels and a separate runtime stage that installs only those wheels.

Build the image from the repository root:

```bash
docker build --tag paysafe-fraud-scoring:latest .
```

Run it with runtime configuration supplied by the deployment environment. Do
not pass a committed `.env` file or copy credentials into the image.

```bash
docker run --rm --publish 8000:8000 \
  --env APP_ENV=prod \
  --env MLFLOW_TRACKING_URI=https://mlflow.example.internal \
  --env MLFLOW_EXPERIMENT_NAME=paysafe-fraud-scoring-prod \
  --env MODEL_REGISTRY_NAME=paysafe-fraud-detector \
  --env MODEL_ALIAS=champion \
  --env API_HOST=0.0.0.0 \
  --env API_PORT=8000 \
  --env API_WORKERS=1 \
  --env LOG_LEVEL=INFO \
  paysafe-fraud-scoring:latest
```

The application resolves `models:/<MODEL_REGISTRY_NAME>@<MODEL_ALIAS>` on
startup. It does not train, bundle, or hardcode a model version. A production
container therefore needs network access and credentials, when required, for a
reachable MLflow tracking and registry service. A direct SQLite URI inside Docker does not access the host registry: its database and local artifacts
live on the host filesystem and are not a shared production service.

## Docker Compose

`compose.yaml` runs the same API image and keeps MLflow external, matching the
application's approved-alias design. Start the MLflow server described in the
README first, then start the API from the repository root:

```powershell
docker compose up --build
```

Compose uses `COMPOSE_MLFLOW_TRACKING_URI`, which defaults to
`http://host.docker.internal:5000`. This is intentionally separate from
`MLFLOW_TRACKING_URI` in `.env`, because `localhost` inside the API container
would refer to the container itself. Set a different reachable registry before
starting Compose when necessary:

```powershell
$env:COMPOSE_MLFLOW_TRACKING_URI = "https://mlflow.example.internal"
docker compose up --build
```

The API is published on `http://localhost:8000` by default. Set
`COMPOSE_API_PORT` to use another host port, inspect status with
`docker compose ps`, and stop it with `docker compose down`. The compose file
does not mount local model artifacts, a SQLite database, or `.env` into the
container.

Older direct-SQLite runs may record model artifacts using host-specific
`file:` paths. A Linux container cannot resolve Windows host paths, even if the
database file is mounted. Configure a network-reachable MLflow server with
remotely accessible or proxied artifacts before running `/score` in a container.
This preserves the approved-alias architecture instead of copying a local
candidate artifact into the image.

Check the running image:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
docker exec <container-id> whoami
docker image ls paysafe-fraud-scoring:latest
```

`whoami` should print `appuser`, not `root`. Docker also runs the configured
`HEALTHCHECK`, which performs `GET /health` without sending transaction data.

After a successful build, record the actual image size and run
`docker run --rm --entrypoint whoami paysafe-fraud-scoring:latest`; it should
return `appuser`. Health, Swagger, and scoring must be exercised against the
deployment MLflow server described above. Do not record a container score for a
local SQLite store unless that exact setup was actually tested.

## Image vulnerability scanning

Trivy scans the **final Docker image**, including its operating-system and
installed Python packages. It does not scan the source directory or only the
lock file. Build the documented local image tag first, then use the
non-blocking command to inspect findings:

```bash
trivy image --severity HIGH,CRITICAL paysafe-fraud-scoring:latest
```

Generate a machine-readable local result and apply the same blocking policy as
CI when needed:

```bash
mkdir -p artifacts/security
trivy image --format json --output artifacts/security/trivy-image.json --severity HIGH,CRITICAL paysafe-fraud-scoring:latest
trivy image --severity HIGH,CRITICAL --exit-code 1 paysafe-fraud-scoring:latest
```

CI builds `paysafe-fraud-scoring:${{ github.sha }}` and scans that exact final
image. HIGH and CRITICAL findings make the Docker job fail. The JSON result is
uploaded as the `trivy-image-scan-<sha>` GitHub Actions artifact, including
when the scan step fails. The local `artifacts/security/` result is ignored by
Git; retain real scan output with deployment or release evidence rather than
claiming a clean scan without running it.

Image scanning and SBOM generation have different responsibilities: Trivy
reports known vulnerabilities, while Syft inventories image components.

## SBOM generation

An SBOM is an inventory of the operating-system and Python components contained
in a final container image. This repository uses [Syft](https://github.com/anchore/syft)
to generate SPDX JSON from the image, not from `requirements.lock`, source code,
or the Docker build context. Generate an SBOM only after building the exact tag
you intend to release.

Install Syft as a developer or CI tool; it is deliberately not added to the
runtime image. On Windows, install it using a package manager supported by
Anchore's installation documentation, then confirm the installed version:

```powershell
winget install Anchore.Syft
syft version
```

Build the image, create the ignored artifact directory, and generate SPDX JSON:

```powershell
docker build --tag paysafe-fraud-scoring:v1 .
New-Item -ItemType Directory -Force sbom
syft paysafe-fraud-scoring:v1 -o spdx-json=sbom/paysafe-fraud-scoring.spdx.json
```

If an older Windows Syft release reports an `unable to place layer cache` error,
scan a temporary archive of the same final image with Syft's Linux tool image.
This is a local fallback for the Windows filesystem issue; it does not change
the production image or scan source code instead of the image.

```powershell
$archive = Join-Path $env:TEMP "paysafe-fraud-scoring-v1.tar"
$output = (Resolve-Path sbom).Path
docker save paysafe-fraud-scoring:v1 --output $archive
docker run --rm --mount "type=bind,source=$archive,target=/image.tar,readonly" --mount "type=bind,source=$output,target=/output" anchore/syft:latest docker-archive:/image.tar -o spdx-json=/output/paysafe-fraud-scoring.spdx.json
Remove-Item $archive
```

Verify that the output is valid SPDX JSON and contains actual image components:

```powershell
$sbom = Get-Content sbom/paysafe-fraud-scoring.spdx.json -Raw | ConvertFrom-Json
if (-not $sbom.packages -or $sbom.packages.Count -eq 0) { throw "SBOM contains no packages" }
"SPDX packages: $($sbom.packages.Count)"
```

The generated file is ignored by Git because it is a build artifact, not source
code. Record the image tag, `syft version`, output path, and verification result
with the release evidence. Generate CycloneDX only when a downstream consumer
requires it; SPDX JSON is the repository's standard artifact format.

CI builds `paysafe-fraud-scoring:${{ github.sha }}`, downloads Syft for the job,
generates `sbom/paysafe-fraud-scoring.spdx.json` from that exact image, verifies
that it has packages, and uploads it as the `container-sbom-spdx-<sha>` workflow
artifact. It does not commit the SBOM or add a separate branch-protection check.

## Existing local registry with Docker Desktop

Both local and Docker APIs use the same MLflow HTTP server. Only the server
opens `mlruns.db` and reads the artifact tree in `mlruns/`. Local clients use
`http://localhost:5000`; Docker uses `http://host.docker.internal:5000`.
The complete startup, scoring, and laptop-transfer commands are in
[the README](../README.md#6-end-to-end-local-runbook).
Do not pass the development `.env` wholesale to Docker: its localhost refers
to the container itself. Supply the Docker tracking URI explicitly.

The existing registry was backed up before migrating artifact metadata to
`mlflow-artifacts:/` references. These paths are resolved relative to each
client's tracking endpoint, so neither a Windows path nor a laptop IP is needed.
The existing champion remains version 4; its run, metrics, model bytes, source
identity, and alias are preserved. Migration evidence and the SQLite backup
are under ignored `artifacts/`; `mlflow_portable_migration.json` records the
changed fields and original values. This migration changes local registry
state, not files shipped in Git. No model is copied into the image.

The migration covers the champion's logged-model location, its registered
version's cached storage location, its source run artifact URI, and the local
experiment roots for future logging. Other historical model/run references
may still be host-specific. New experiments created through the server use
proxied artifact locations by default. Direct SQLite tracking is no longer the
application default: `mlflow-artifacts:/` downloads require HTTP tracking.
CI tests continue to use isolated SQLite stores and need no running server.

On another laptop, transfer the stopped server's database and complete artifact
tree together, or run the real training and gated promotion workflow there.
Never commit these files. For rollback, stop clients and restore the metadata
fields recorded in the migration manifest; do not overwrite newer runs with
an old full-database backup. Restoring old locations also restores their
host/IP dependency.

The server binds to all interfaces for Docker access with an explicit host
allowlist. This unauthenticated HTTP setup is intended for a trusted local
assessment network. Use authenticated HTTPS for shared deployments. Native
Linux Docker Engine needs `--add-host=host.docker.internal:host-gateway`;
Docker Desktop supplies the hostname automatically. Port 5000 was chosen to
avoid the existing loopback MLflow service on port 5000.
