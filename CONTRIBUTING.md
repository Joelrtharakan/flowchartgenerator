# Contributing / Team Workflow

## DevOps lifecycle and Agile

Work is planned in short sprints (1 week). Each task is a GitHub Issue on the project board
(`Backlog -> In progress -> In review -> Done`). Every change flows through the same loop:

`Plan (issue) -> Code (branch) -> Build/Test (CI) -> Release (GHCR image) -> Deploy (Ansible/K8s) -> Operate/Monitor (Prometheus + Grafana) -> Feedback (alerts, issues)`

## Branching strategy

| Branch | Purpose | Rules |
|---|---|---|
| `master` | Always deployable. Every push builds, scans, publishes and deploys. | Protected: PR + green CI required, no force-push |
| `feature/<topic>` | New functionality | Branch from `master`, short-lived |
| `fix/<topic>` | Bug fixes | Branch from `master` |
| `infra/<topic>` | Terraform / K8s / pipeline changes | Branch from `master` |

Recommended branch protection (Settings -> Branches -> `master`):
require a pull request, require status checks `Lint, SAST and unit tests`, `Secret scan (gitleaks)`,
`Validate Terraform, Kubernetes, Ansible, Dockerfiles`, and `Build, scan and publish`, and block force pushes.

## Pull-request workflow

1. Create an issue, then `git switch -c feature/<topic>`.
2. Commit in small steps using imperative messages (`Add /health endpoint`).
3. `git push -u origin feature/<topic>` and open a PR using the template.
4. CI runs automatically; a teammate reviews; address comments with new commits.
5. Squash-merge into `master`; the branch is deleted. The pipeline deploys.

## Handling merge conflicts

```bash
git fetch origin
git switch feature/<topic>
git rebase origin/master      # or: git merge origin/master
# fix the conflict markers in each file, then
git add <file> && git rebase --continue
git push --force-with-lease
```

Keep PRs small to make conflicts rare, and re-run the tests locally after resolving.

## Local checks before pushing

```bash
pip install -r requirements-dev.txt
ruff check . && bandit -q -r training/app.py frontend/server.py
pytest --cov=app --cov=server
```
