# Presentation Runbook

Target: 8 to 10 minutes plus questions. Rubric focus: pipeline, containers, IaC and cloud, monitoring, collaboration.

## Before you start (5 minutes earlier)

```bash
# 1. VM stack up and healthy
ssh -i ~/.ssh/flowchart_azure azureuser@20.219.147.18 'cd ~/flowchart && docker compose up -d && curl -s localhost:8000/health'
# 2. Tunnel for Grafana and Prometheus (keep this terminal open)
ssh -i ~/.ssh/flowchart_azure -L 3000:localhost:3000 -L 9090:localhost:9090 azureuser@20.219.147.18
# 3. Warm the model with one request (first call is slowest)
curl -s -X POST http://20.219.147.18:8082/generate -H 'Content-Type: application/json' -d '{"prompt":"User login"}'
```

Open these tabs: the GitHub repo (Actions, Pull requests), http://20.219.147.18:8082, http://localhost:3000 (Grafana), http://localhost:9090/targets.
If the VM is stopped: `az vm start -g flowchart-rg -n flowchart-vm`, then step 1.

## Storyline and timing

| Min | Slide / action | Point to make |
|---|---|---|
| 0:00 | Problem | LLM flowchart generator; why it needs DevOps (model, API, UI, retraining) |
| 1:00 | Architecture diagram | Dev -> GitHub -> CI -> GHCR -> staging -> production; Terraform provisions the VM |
| 2:00 | Git workflow | Show PR list: feature branches, issues closed by PRs, a resolved merge conflict, Dependabot triage |
| 3:30 | Pipeline run (GitHub Actions) | Walk the stages: lint, SAST, tests, gitleaks, IaC validation, build, Trivy, smoke test, publish, staging, approval gate |
| 5:00 | Containers | Multi-stage Dockerfile, non-root, health check; images in GHCR |
| 5:45 | Kubernetes | `kubectl get pods`, scale 2 -> 4, rolling update, `rollout undo` (see commands below) |
| 7:00 | Terraform and Ansible | `terraform plan`, Azure portal resources; what Ansible does |
| 7:45 | Live demo | Type a prompt in the UI; show the flowchart |
| 8:30 | Monitoring | Prometheus targets, Grafana dashboard, stop the API container to fire `ApiDown` |
| 9:30 | Limitations and wrap-up | Be upfront (see below) |

## Kubernetes demo commands (on the VM)

```bash
kubectl config use-context kind-flowchart
kubectl -n flowchart get pods,svc,hpa
kubectl -n flowchart scale deploy/api --replicas=4 && kubectl -n flowchart rollout status deploy/api
kubectl -n flowchart set env deploy/api DEMO_VERSION=v2 && kubectl -n flowchart rollout status deploy/api
kubectl -n flowchart rollout history deploy/api
kubectl -n flowchart rollout undo deploy/api
```

## Alert demo

```bash
ssh -i ~/.ssh/flowchart_azure azureuser@20.219.147.18 'cd ~/flowchart && docker compose stop api'
# wait ~1 min: Prometheus -> Alerts -> ApiDown goes Pending, then Firing
ssh -i ~/.ssh/flowchart_azure azureuser@20.219.147.18 'cd ~/flowchart && docker compose start api'
```
The API needs about 1 to 2 minutes to reload the model after a start.

## Be upfront about

- The fine-tuned LoRA adapter was lost with the earlier VM; the deployment serves base Phi-3 with a guided prompt and output cleanup.
- CPU inference takes about a minute per request.
- The demo autoscaler shows `<unknown>` because the kind cluster has no metrics-server; the scaling rule (70% CPU, 2 to 6 replicas) is in `k8s/base/api-hpa.yaml`.
- Production deploys run Ansible from a laptop because the VM firewall only allows the owner's IP for SSH.

## Screenshot checklist (for the report and slides)

- [ ] GitHub Actions run, all jobs green (open the run summary)
- [ ] Pull requests list with merged and closed PRs; one PR showing the merge conflict commit message
- [ ] Issues closed by PRs
- [ ] GHCR packages (`flowchart-api`, `flowchart-frontend`)
- [ ] Trivy scan step output
- [ ] Terraform: `terraform plan` / `apply` output and the Azure portal resource group
- [ ] `docker compose ps` on the VM
- [ ] `kubectl get pods`, scale, rollout and rollback output
- [ ] Prometheus Targets page (API up) and Alerts page
- [ ] Grafana "Flowchart API" dashboard with traffic
- [ ] Frontend with a generated flowchart
- [ ] Coursera certificate

## If something fails live

| Problem | Action |
|---|---|
| UI hangs on Generate | The first request after a restart is slow; wait up to 2 minutes or show a recorded result |
| VM unreachable | Show the Terraform output and screenshots; run the stack locally with `docker compose up` (`SKIP_MODEL_LOAD=1` for a quick demo) |
| Grafana blank | Send a few requests first; dashboards need traffic |
