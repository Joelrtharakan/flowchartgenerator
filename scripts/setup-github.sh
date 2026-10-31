#!/usr/bin/env bash
# One-time GitHub setup for the CI/CD pipeline. Run it yourself from the repo root:
#   bash scripts/setup-github.sh
# Needs: gh (logged in as the repo admin), ssh access to the VM with the Terraform-created key.
set -euo pipefail

REPO="${REPO:-Joelrtharakan/flowchartgenerator}"
VM_HOST="${VM_HOST:-20.219.147.18}"
VM_USER="${VM_USER:-azureuser}"
ADMIN_KEY="${ADMIN_KEY:-$HOME/.ssh/flowchart_azure}"   # key Terraform put on the VM
DEPLOY_KEY="$HOME/.ssh/gh_deploy"                      # new key used only by GitHub Actions

echo "== 1/5 Secrets"
gh secret set VM_HOST --repo "$REPO" --body "$VM_HOST"
gh secret set VM_USER --repo "$REPO" --body "$VM_USER"

ssh -i "$ADMIN_KEY" "$VM_USER@$VM_HOST" "grep ^GRAFANA_ADMIN_PASSWORD ~/flowchart/.env | cut -d= -f2" \
  | tr -d '\n' | gh secret set GRAFANA_ADMIN_PASSWORD --repo "$REPO"

echo "== 2/5 Deploy-only SSH key for GitHub Actions"
if [ ! -f "$DEPLOY_KEY" ]; then
  ssh-keygen -t ed25519 -N "" -C github-actions-deploy -f "$DEPLOY_KEY" -q
fi
ssh -i "$ADMIN_KEY" "$VM_USER@$VM_HOST" \
  "grep -qF \"\$(cat)\" ~/.ssh/authorized_keys || cat >> ~/.ssh/authorized_keys" < "${DEPLOY_KEY}.pub"
gh secret set VM_SSH_KEY --repo "$REPO" < "$DEPLOY_KEY"

echo "== 3/5 Environments (production needs your approval)"
ME_ID="$(gh api user -q .id)"
gh api -X PUT "repos/$REPO/environments/staging" >/dev/null
echo "{\"reviewers\":[{\"type\":\"User\",\"id\":$ME_ID}]}" \
  | gh api -X PUT "repos/$REPO/environments/production" --input - >/dev/null

echo "== 4/5 Branch protection on main (PR + passing checks required)"
gh api -X PUT "repos/$REPO/branches/main/protection" --input - >/dev/null <<JSON
{
  "required_status_checks": {
    "strict": false,
    "contexts": [
      "Lint, SAST and unit tests",
      "Secret scan (gitleaks)",
      "Validate Terraform, Kubernetes, Ansible, Dockerfiles"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

echo "== 5/5 Production deploy switch"
echo "The Azure firewall only allows SSH from your own IP, so GitHub's runners cannot reach the VM."
echo "Pick ONE:"
echo "  a) Deploy from your Mac:   bash scripts/deploy-azure.sh        (recommended, no firewall change)"
echo "  b) Let the pipeline deploy: open port 22 (key-only) in terraform/variables.tf allowed_ssh_cidr,"
echo "     run 'terraform apply', then: gh variable set DEPLOY_ENABLED --repo $REPO --body true"
echo
echo "Done."
