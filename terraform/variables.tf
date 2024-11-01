# variables.tf
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "Region for the database instance"
  type        = string
  default     = "europe-west1"
}

variable "instance_name" {
  description = "Nome dell'istanza Cloud SQL"
  type        = string
  default     = "omnia-db-instance"
}

variable "database_name" {
  description = "Nome del database"
  type        = string
  default     = "omniadb"
}

variable "db_user" {
  description = "omnia_user"
  type        = string
  default     = "omnia_user"
}

variable "authorized_ip" {
  description = "151.40.117.156/32"
  type        = string
}

variable "omnia_password" {
  description = "G9XbKxq5iMJ@bmL6&M2EcL"
  type        = string
  sensitive   = true
}