# main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud SQL instance
resource "google_sql_database_instance" "omnia_db" {
  name             = var.instance_name
  database_version = "POSTGRES_14"
  region           = var.region
  
  settings {
    tier = "db-f1-micro"  # Per sviluppo
    
    backup_configuration {
      enabled    = true
      start_time = "02:00"
    }
    
    ip_configuration {
      ipv4_enabled = true
      
      authorized_networks {
        name  = "allow-development"
        value = var.authorized_ip
      }
    }
  }

  deletion_protection = false
}

# Database
resource "google_sql_database" "omniadb" {
  name     = "omniadb"
  instance = google_sql_database_instance.omnia_db.name  # Corretto qui
}

# Database user
resource "google_sql_user" "omnia_user" {
  name     = "omnia_user"
  instance = google_sql_database_instance.omnia_db.name  # Corretto qui
  password = var.omnia_password
}