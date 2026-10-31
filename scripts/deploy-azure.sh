#!/usr/bin/env bash
# Deploy the published images to the Azure VM with Ansible, from your own machine.
#   bash scripts/deploy-azure.sh [image_tag]
# Prereqs: images published by CI (merge to main first), ansible installed (brew install ansible),
# and the GHCR packages set to Public (GitHub -> your profile -> Packages -> package settings),
# or export GHCR_TOKEN with a token that has read:packages.
set -euo pipefail

VM_HOST="${VM_HOST:-20.219.147.18}"
VM_USER="${VM_USER:-azureuser}"
KEY="${KEY:-$HOME/.ssh/flowchart_azure}"
OWNER="${OWNER:-joelrtharakan}"
TAG="${1:-latest}"

for tool in ansible-playbook ansible-galaxy ssh; do
  command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found (brew install ansible). Nothing on the VM was touched." >&2; exit 1; }
done

cd "$(dirname "$0")/../ansible"
printf '[flowchart]\n%s ansible_user=%s ansible_ssh_private_key_file=%s\n' "$VM_HOST" "$VM_USER" "$KEY" > inventory.ini

# Pre-flight: make sure both images can be pulled BEFORE touching the running stack
for img in flowchart-api flowchart-frontend; do
  if ! ssh -i "$KEY" "$VM_USER@$VM_HOST" "${GHCR_TOKEN:+echo '$GHCR_TOKEN' | docker login ghcr.io -u '$OWNER' --password-stdin >/dev/null 2>&1;} docker pull ghcr.io/$OWNER/$img:$TAG >/dev/null 2>&1"; then
    echo "ERROR: cannot pull ghcr.io/$OWNER/$img:$TAG from the VM." >&2
    echo "Set the package to Public (GitHub -> Packages -> $img -> Package settings) or export GHCR_TOKEN (read:packages)." >&2
    echo "The running stack was NOT touched." >&2
    exit 1
  fi
done

# The earlier hand-deployed stack uses the same ports; stop it first (volumes, including the model cache, are kept)
ssh -i "$KEY" "$VM_USER@$VM_HOST" 'cd ~/flowchart 2>/dev/null && docker compose down || true'

: "${GRAFANA_ADMIN_PASSWORD:=$(ssh -i "$KEY" "$VM_USER@$VM_HOST" "grep ^GRAFANA_ADMIN_PASSWORD ~/flowchart/.env | cut -d= -f2")}"
export GRAFANA_ADMIN_PASSWORD

ansible-galaxy collection install community.general >/dev/null
ansible-playbook playbook.yml \
  -e image_owner="$OWNER" \
  -e image_tag="$TAG" \
  ${GHCR_TOKEN:+-e ghcr_user="$OWNER" -e ghcr_token="$GHCR_TOKEN"}

echo "Frontend: http://$VM_HOST:8082   API docs: http://$VM_HOST:8000/docs"
