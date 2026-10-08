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
docker run --rm --publish 8000:8000 --env APP_ENV=prod --env MLFLOW_TRACKING_URI=https://mlflow.example.internal --env MLFLOW_EXPERIMENT_NAME=paysafe-fraud-scoring-prod --env MODEL_REGISTRY_NAME=paysafe-fraud-detector --env MODEL_ALIAS=champion --env API_HOST=0.0.0.0 --env API_PORT=8000 --env API_WORKERS=1 --env LOG_LEVEL=info paysafe-fraud-scoring:latest
```

The application resolves `models:/<MODEL_REGISTRY_NAME>@<MODEL_ALIAS>` on
startup. It does not train, bundle, or hardcode a model version. A production
container therefore needs network access and credentials, when required, for a
reachable MLflow tracking and registry service. A direct SQLite URI inside Docker does not access the host registry: its database and local artifacts
live on the host filesystem and are not a shared production service.

## Docker Compose

`compose.yaml` runs a persistent MLflow tracking server and the same API image.
MLflow metadata and proxied artifacts live in the named `mlflow-data` volume.
Start MLflow first from the repository root:

```powershell
docker compose up -d mlflow
```

Open `http://localhost:5000` to inspect the registry. Local training and
promotion commands use `MLFLOW_TRACKING_URI=http://localhost:5000`. The API
uses the internal service address `http://mlflow:5000`; this is the default
value of `COMPOSE_MLFLOW_TRACKING_URI`.

On a new `mlflow-data` volume, create and promote a real passing candidate
before starting the API:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m fraud_scoring.mlflow_tracking --config configs/dev.yaml
python -m fraud_scoring.model_registry promote --config configs/dev.yaml --run-id <REAL_RUN_ID> --model-uri <REAL_MODEL_URI>
docker compose up -d --build api
```

The API is published on `http://localhost:8000` by default. Set
`COMPOSE_API_PORT` to use another host port, and `COMPOSE_MLFLOW_PORT` to move
the MLflow UI and local-client port together. Inspect status with
`docker compose ps`, stop services with `docker compose down`, and retain the
registry with the named volume. `docker compose down -v` deliberately deletes
the MLflow database and artifacts.

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
trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 paysafe-fraud-scoring:latest
```

CI builds `paysafe-fraud-scoring:${{ github.sha }}` and scans that exact final
image. Its JSON artifact retains every HIGH and CRITICAL finding as
`trivy-image-scan-<sha>`. The enforcement step blocks HIGH and CRITICAL
vulnerabilities with an upstream fixed version. Findings without a published
fix remain visible in the report for review and must be reassessed whenever the
pinned base-image digest changes. The local `artifacts/security/` result is
ignored by Git; retain real scan output with deployment or release evidence
rather than claiming a clean scan without running it.

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

Build the image, create the ignored artifact directory, and generate SPDX JSON
through Docker Desktop's daemon, matching the local image scan:

```powershell
docker build --tag paysafe-fraud-scoring:v1 .
New-Item -ItemType Directory -Force sbom
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "${PWD}/sbom:/sbom" anchore/syft:latest paysafe-fraud-scoring:v1 -o spdx-json=/sbom/paysafe-fraud-scoring.spdx.json
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

## Compose-managed local registry

Both local CLIs and the Docker API use the same Compose-managed MLflow service.
Local clients use `http://localhost:5000`; the API uses `http://mlflow:5000` on
the internal Compose network. No host MLflow process, host SQLite file, or
artifact directory is mounted into the API container.

The named `mlflow-data` volume stores the server's SQLite backend and proxied
artifact files. It survives `docker compose down`; `docker compose down -v`
deletes it. A pre-existing host-based MLflow registry is not imported
automatically. Create and promote a real model through the running Compose
server, or explicitly export and import the stopped volume when moving laptops.

This unauthenticated HTTP setup is intended for a trusted local assessment
environment. Use authenticated HTTPS and managed database/object storage for a
shared deployment.
