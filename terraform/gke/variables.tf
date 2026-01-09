# =============================================================================
# GKE Cluster Variables for RAG Historian
# =============================================================================

variable "cluster_name" {
  type        = string
  default     = "rag-historian-cluster"
  description = "Name of the GKE cluster"
}

variable "node_count" {
  type        = number
  default     = 2
  description = "Number of nodes in the node pool"
}

variable "node_machine_type" {
  type        = string
  default     = "e2-standard-2"
  description = "Machine type for nodes (e2-standard-2 = 2 vCPU, 8GB RAM)"
}

variable "node_disk_size_gb" {
  type        = number
  default     = 50
  description = "Disk size in GB for each node"
}

variable "kubernetes_version" {
  type        = string
  default     = "latest"
  description = "Kubernetes version for the cluster"
}


