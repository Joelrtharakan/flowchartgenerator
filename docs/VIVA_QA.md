# Viva Preparation: Questions and Answers

Short answers tied to this project. Know the *why*, not just the tool names.

## DevOps basics and collaboration

**What is DevOps and how does it show up in this project?**
Developers and operations share one automated flow: plan, code, build, test, release, deploy, operate, monitor. Here a push triggers GitHub Actions, which tests, scans and packages the app, and Terraform/Ansible/Docker run it on Azure with Prometheus watching it.

**Which branching strategy did you use and why?**
Short-lived `feature/*`, `fix/*` and `infra/*` branches off `master`, merged by pull request after CI passes. `master` stays deployable. See `CONTRIBUTING.md`.

**Did you hit a merge conflict? How did you resolve it?**
Yes. Merging `deploy-frontend-to-render` conflicted with my change in `frontend/server.py` (both made `PORT`/`API_URL` configurable). Merging `data-gen` conflicted in `tests/dataset_script.py`, which had been refactored on `master`. I kept the refactored validation, added data-gen's `enforce_inline_format`, ran the tests, then committed the merge.

## CI/CD

**Walk me through your pipeline.**
`.github/workflows/ci-cd.yml`: (1) ruff lint, bandit SAST, pytest with coverage; gitleaks secret scan; (2) validation of Terraform, Kubernetes manifests, Ansible, Prometheus rules and Dockerfiles; (3) build each Docker image, smoke-test it, scan with Trivy, push to GHCR on `master`; (4) deploy to staging with Compose and verify Prometheus scrapes the API; (5) deploy to production through Ansible behind a manual approval.

**What is the artifact?**
The Docker images tagged with the commit SHA in GHCR, plus the test/coverage and Trivy reports.

**Why scan and test before building the image?**
To fail fast and cheaply, and never publish an image from code that fails quality gates.

**CI vs CD vs continuous deployment?**
CI merges and verifies every change; continuous delivery keeps every build releasable; continuous deployment releases automatically. Production here needs an approval, so it is continuous delivery.

**How do you handle secrets?**
Never in git. GitHub Secrets for the pipeline (`VM_SSH_KEY`, `GRAFANA_ADMIN_PASSWORD`), a `.env` with mode 600 on the VM, Kubernetes Secrets for the cluster, and gitleaks to catch accidents. `.gitignore` excludes Terraform state and tfvars.

## Containers and Kubernetes

**Why multi-stage builds?**
`Dockerfile.api` builds wheels in a builder stage and copies only them into a slim runtime image: smaller, no compilers or caches, smaller attack surface. It also runs as a non-root user and has a health check.

**Why isn't the model baked into the image?**
Weights are several GB and change on retraining. Adapters are mounted from a volume (a PVC on Kubernetes), so the image stays small and the model can be updated without a rebuild.

**Explain your Kubernetes objects.**
Deployments run the pods (2 replicas), Services give stable access, a HorizontalPodAutoscaler scales 2 to 6 replicas on CPU, a PVC holds adapters, a ConfigMap holds settings, and a Secret holds tokens. Probes (`/health`) drive readiness and restarts.

**How does a rolling update avoid downtime?**
`maxUnavailable: 0` and `maxSurge: 1`: a new pod must become ready before an old one is removed. `kubectl rollout undo` reverts.

## IaC and cloud

**What does Terraform do here? Why not click in the portal?**
It declares the resource group, network, security group, public IP and VM in code: repeatable, reviewable, destroyable. `terraform plan` shows changes before `apply`.

**Terraform vs Ansible?**
Terraform provisions infrastructure (the VM); Ansible configures what runs on it (Docker, firewall, the stack). Terraform is declarative state-based; Ansible is task-based and idempotent.

**What went wrong when you first applied it?**
A new Azure subscription failed on parallel resource-provider registration, so I registered Compute and Network manually and set `skip_provider_registration`. Azure also rejected my ed25519 key (RSA only).

## Monitoring and security

**What do you monitor?**
The API exposes Prometheus metrics (`/metrics`): request rate, latency histograms, status codes, memory. Prometheus scrapes them every 15 s, Grafana shows a dashboard, and alert rules fire on API down, a 5xx rate above 5%, and p95 latency above 60 s.

**Why is Grafana not public?**
Only ports 22 (restricted to my IP), 8000 and 8082 are open. Grafana and Prometheus bind to localhost and are reached through an SSH tunnel.

**What security scanning do you do?**
bandit (code), Trivy (images), gitleaks (secrets), Dependabot (dependencies), hadolint (Dockerfiles).

## Be honest about limits

- The fine-tuned LoRA adapter was lost with the previous VM, so the deployment currently serves the base Phi-3 model with a guided prompt and output cleanup.
- CPU inference takes about a minute per request.
- Free-tier credits limit how long the cloud VM can stay up; Terraform makes it quick to recreate.
