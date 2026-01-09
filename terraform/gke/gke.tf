# =============================================================================
# GKE Cluster Configuration for RAG Historian
# =============================================================================

# -----------------------------------------------------------------------------
# GKE Cluster (Control Plane)
# -----------------------------------------------------------------------------
resource "google_container_cluster" "primary" {
  provider = google-beta
  name     = var.cluster_name
  location = var.gcp_zone

  # Disable deletion protection for development (enable for production)
  deletion_protection = false

  # Create smallest default node pool and delete it immediately
  # We use a separately managed node pool for more control
  remove_default_node_pool = true
  initial_node_count       = 1

  # Master authentication
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Addons configuration
  addons_config {
    # HTTP Load Balancing (for Ingress)
    http_load_balancing {
      disabled = false
    }

    # Horizontal Pod Autoscaling
    horizontal_pod_autoscaling {
      disabled = false
    }

    # GCE Persistent Disk CSI Driver (for PVCs)
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  # Network configuration
  network    = "default"
  subnetwork = "default"

  # Workload Identity (recommended for GCP service access)
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }
}

# -----------------------------------------------------------------------------
# GKE Node Pool (Worker Nodes)
# -----------------------------------------------------------------------------
resource "google_container_node_pool" "primary_nodes" {
  provider   = google-beta
  name       = "${var.cluster_name}-node-pool"
  location   = var.gcp_zone
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count

  # Auto-scaling configuration
  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Node configuration
  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = var.node_disk_size_gb
    disk_type    = "pd-standard"

    # Preemptible nodes (cheaper, but can be terminated - good for dev)
    # Set to false for production
    preemptible = true

    metadata = {
      disable-legacy-endpoints = "true"
    }

    # OAuth scopes for node service account
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
      "https://www.googleapis.com/auth/compute",
      "https://www.googleapis.com/auth/devstorage.read_write",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]

    # Labels for node selection
    labels = {
      environment = "development"
      app         = "rag-historian"
    }

    # Taints (optional - for dedicated workloads)
    # taint {
    #   key    = "dedicated"
    #   value  = "rag-historian"
    #   effect = "NO_SCHEDULE"
    # }
  }
}

# -----------------------------------------------------------------------------
# Google Container Registry (GCR) - for Docker images
# -----------------------------------------------------------------------------
resource "google_artifact_registry_repository" "rag_historian" {
  provider      = google
  location      = var.gcp_region
  repository_id = "rag-historian"
  description   = "Docker images for RAG Historian application"
  format        = "DOCKER"
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
output "cluster_name" {
  value       = google_container_cluster.primary.name
  description = "GKE cluster name"
}

output "cluster_endpoint" {
  value       = google_container_cluster.primary.endpoint
  description = "GKE cluster endpoint"
  sensitive   = true
}

output "cluster_ca_certificate" {
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  description = "GKE cluster CA certificate"
  sensitive   = true
}

output "registry_url" {
  value       = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/rag-historian"
  description = "Artifact Registry URL for Docker images"
}

output "kubectl_config_command" {
  value       = "gcloud container clusters get-credentials ${var.cluster_name} --zone ${var.gcp_zone} --project ${var.gcp_project_id}"
  description = "Command to configure kubectl"
}


