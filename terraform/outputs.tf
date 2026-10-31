output "public_ip" {
  value = azurerm_public_ip.pip.ip_address
}

output "ssh_command" {
  value = "ssh ${var.admin_username}@${azurerm_public_ip.pip.ip_address}"
}

output "api_url" {
  value = "http://${azurerm_public_ip.pip.ip_address}:8000"
}

output "frontend_url" {
  value = "http://${azurerm_public_ip.pip.ip_address}:8082"
}
