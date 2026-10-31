variable "project" {
  description = "Name prefix for all resources"
  type        = string
  default     = "flowchart"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "centralindia"
}

variable "vm_size" {
  description = "VM size. Phi-3 mini on CPU needs >= 16 GB RAM."
  type        = string
  default     = "Standard_D4s_v5"
}

variable "admin_username" {
  type    = string
  default = "azureuser"
}

variable "ssh_public_key" {
  description = "Contents of your SSH public key (e.g. file(\"~/.ssh/id_rsa.pub\"))"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH to the VM. Restrict this to your own IP."
  type        = string
  default     = "0.0.0.0/0"
}

variable "tags" {
  type = map(string)
  default = {
    project = "flowchart-generator"
    managed = "terraform"
  }
}
