# Terraform Configuration for RAG Historian

## GKE (Google Kubernetes Engine) Deployment

### Prerequisites

1. **GCP Account** with billing enabled
2. **GCP Project** created
3. **Service Account** with the following roles:
   - `Kubernetes Engine Admin`
   - `Compute Admin`
   - `Artifact Registry Admin`
   - `Service Account User`

4. **Tools installed:**
   ```bash
   # Google Cloud SDK
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init

   # Terraform
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/

   # kubectl
   gcloud components install kubectl
   ```

### Setup

1. **Authenticate with GCP:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Create a service account key:**
   ```bash
   # Create service account
   gcloud iam service-accounts create terraform-sa \
     --display-name "Terraform Service Account"

   # Grant roles
   PROJECT_ID=$(gcloud config get-value project)
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:terraform-sa@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/container.admin"
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:terraform-sa@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/compute.admin"
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:terraform-sa@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/artifactregistry.admin"

   # Create key file
   mkdir -p ~/.gcp
   gcloud iam service-accounts keys create ~/.gcp/terraform-credentials.json \
     --iam-account=terraform-sa@$PROJECT_ID.iam.gserviceaccount.com
   ```

3. **Configure Terraform variables:**
   ```bash
   cd terraform/gke
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your values
   ```

### Deploy

```bash
cd terraform/gke

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply (creates GKE cluster)
terraform apply

# Get kubectl credentials
$(terraform output -raw kubectl_config_command)
```

### Destroy

```bash
cd terraform/gke
terraform destroy
```

## Cost Estimate (GKE)

| Resource | Specification | Monthly Cost (approx) |
|----------|---------------|----------------------|
| GKE Control Plane | 1 zonal cluster | Free (1 free per account) |
| Nodes (preemptible) | 2x e2-standard-2 | ~$30 |
| Persistent Disks | 15 Gi standard | ~$1 |
| Load Balancer | 1 regional | ~$20 |
| Artifact Registry | ~2 GB storage | ~$0.20 |
| **Total** | | **~$50-60/month** |

> Note: Using preemptible nodes. For production, use regular nodes (~$120/month).


