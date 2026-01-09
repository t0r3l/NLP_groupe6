# =============================================================================
# GCP Provider Configuration for RAG Historian
# =============================================================================

variable "gcp_project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "gcp_region" {
  type        = string
  default     = "europe-west3"
  description = "GCP Region (Frankfurt)"
}

variable "gcp_zone" {
  type        = string
  default     = "europe-west3-a"
  description = "GCP Zone"
}

variable "gcp_credentials_file" {
  type        = string
  default     = "~/.gcp/terraform-credentials.json"
  description = "Path to GCP service account credentials JSON"
}

# Google Beta provider (required for some GKE features)
provider "google-beta" {
  project     = var.gcp_project_id
  region      = var.gcp_region
  credentials = file(var.gcp_credentials_file)
}

# Standard Google provider
provider "google" {
  project     = var.gcp_project_id
  region      = var.gcp_region
  credentials = file(var.gcp_credentials_file)
}

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}


