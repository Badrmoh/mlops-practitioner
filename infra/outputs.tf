output "created_bucket_id" {
  description = "The ID/Name of the newly created bucket"
  value       = aws_s3_bucket.this.id
}