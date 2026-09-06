provider "aws" {
  region = var.aws_region

  # Route S3 API calls to your RustFS endpoint
  endpoints {
    s3 = var.rustfs_endpoint
  }
  s3_use_path_style           = true
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
}

# Create the bucket in RustFS
resource "aws_s3_bucket" "this" {
  bucket = var.bucket_name
}
