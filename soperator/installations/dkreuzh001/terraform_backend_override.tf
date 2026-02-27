terraform {
  backend "s3" {
    bucket = "tfstate-slurm-k8s-64bacace4bc02cd49515090b9f8ea2b2"
    key    = "slurm-k8s.tfstate"

    endpoints = {
      s3 = "https://storage.eu-north1.nebius.cloud:443"
    }
    region = "eu-north1"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}
