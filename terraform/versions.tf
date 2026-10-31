terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
}

provider "azurerm" {
  features {}
  # New subscriptions fail when Terraform auto-registers every provider in parallel.
  # Register only what we use first: az provider register -n Microsoft.Compute -n Microsoft.Network --wait
  skip_provider_registration = true
  # Credentials come from `az login` or ARM_* environment variables - never hard-coded.
}
