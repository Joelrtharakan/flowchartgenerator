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

cd "$(dirname "$0")/../ansible"
printf '[flowchart]\n%s ansible_user=%s ansible_ssh_private_key_file=%s\n' "$VM_HOST" "$VM_USER" "$KEY" > inventory.ini

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
