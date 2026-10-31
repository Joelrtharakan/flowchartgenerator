# Local infrastructure as code: Terraform creates a multi-node Kubernetes cluster
# inside Docker (kind). Use this when no cloud account is available; the Azure
# equivalent lives in ../terraform.
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    kind = {
      source  = "tehcyx/kind"
      version = "~> 0.6"
    }
  }
}

variable "cluster_name" {
  type    = string
  default = "flowchart"
}

variable "worker_nodes" {
  type    = number
  default = 1
}

resource "kind_cluster" "this" {
  name           = var.cluster_name
  wait_for_ready = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"

      extra_port_mappings {
        container_port = 30082
        host_port      = 8082
      }
    }

    dynamic "node" {
      for_each = range(var.worker_nodes)
      content {
        role = "worker"
      }
    }
  }
}

output "kubeconfig_context" {
  value = "kind-${var.cluster_name}"
}

output "frontend_url" {
  value = "http://localhost:8082"
}
