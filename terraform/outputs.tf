# outputs.tf
output "omnia_user_name" {
  value = google_sql_user.omnia_user.name
}

output "omnia_database_name" {
  value = google_sql_database.omniadb.name
}

output "instance_connection_name" {
  value = google_sql_database_instance.omnia_db.connection_name  # Corretto qui
}

output "database_instance_ip" {
  value = google_sql_database_instance.omnia_db.public_ip_address
}