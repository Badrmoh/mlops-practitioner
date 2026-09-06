variable "aws_region" {
  description = "Default Region of RustFS"
  type        = string
  default     = "us-east-1"
}

variable "rustfs_endpoint" {
  description = "The endpoint for RustFS"
  type        = string
  default     = "http://localhost:9000"
}

variable "bucket_name" {
  description = "The name of the bucket to create in RustFS"
  type        = string
  default     = "mlflow"
}