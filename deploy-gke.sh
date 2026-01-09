#!/bin/bash
# =============================================================================
# RAG Historian - GKE Deployment Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RAG Historian - GKE Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# =============================================================================
# Configuration - MODIFY THESE VALUES
# =============================================================================
GCP_PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
GCP_REGION="${GCP_REGION:-europe-west3}"
GCP_ZONE="${GCP_ZONE:-europe-west3-a}"
CLUSTER_NAME="${CLUSTER_NAME:-rag-historian-cluster}"
IMAGE_NAME="rag-historian-app"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Derived values
REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/rag-historian"
FULL_IMAGE="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

# =============================================================================
# Pre-flight checks
# =============================================================================
echo -e "\n${YELLOW}Pre-flight checks...${NC}"

# Check required tools
for cmd in gcloud docker kubectl terraform; do
  if ! command -v $cmd &> /dev/null; then
    echo -e "${RED}Error: $cmd is not installed${NC}"
    exit 1
  fi
done
echo -e "${GREEN}✓ All required tools installed${NC}"

# Check GCP authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
  echo -e "${RED}Error: Not authenticated with GCP. Run: gcloud auth login${NC}"
  exit 1
fi
echo -e "${GREEN}✓ GCP authentication OK${NC}"

# =============================================================================
# Step 1: Create GKE Cluster with Terraform
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1: Creating GKE Cluster with Terraform${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd terraform/gke

# Create terraform.tfvars if it doesn't exist
if [ ! -f terraform.tfvars ]; then
  echo -e "${YELLOW}Creating terraform.tfvars...${NC}"
  cat > terraform.tfvars <<EOF
gcp_project_id = "${GCP_PROJECT_ID}"
gcp_region     = "${GCP_REGION}"
gcp_zone       = "${GCP_ZONE}"
cluster_name   = "${CLUSTER_NAME}"
EOF
fi

# Initialize and apply Terraform
terraform init
terraform plan -out=tfplan
echo -e "${YELLOW}Review the plan above. Apply? (yes/no)${NC}"
read -r APPLY_CONFIRM
if [ "$APPLY_CONFIRM" = "yes" ]; then
  terraform apply tfplan
else
  echo -e "${RED}Aborted by user${NC}"
  exit 1
fi

cd ../..

# =============================================================================
# Step 2: Configure kubectl
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2: Configuring kubectl${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone ${GCP_ZONE} \
  --project ${GCP_PROJECT_ID}

echo -e "${GREEN}✓ kubectl configured for cluster ${CLUSTER_NAME}${NC}"

# =============================================================================
# Step 3: Build and Push Docker Image
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 3: Building and Pushing Docker Image${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Configure Docker for GCR/Artifact Registry
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet

# Build image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag for registry
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE}

# Push to registry
echo -e "${YELLOW}Pushing to Artifact Registry...${NC}"
docker push ${FULL_IMAGE}

echo -e "${GREEN}✓ Image pushed: ${FULL_IMAGE}${NC}"

# =============================================================================
# Step 4: Update Kubernetes Manifests
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 4: Updating Kubernetes Manifests${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Update image in streamlit-app.yaml
sed -i "s|image: .*rag-historian-app.*|image: ${FULL_IMAGE}|g" k8s/streamlit-app.yaml
sed -i "s|image: nlp_groupe6-streamlit-app:.*|image: ${FULL_IMAGE}|g" k8s/streamlit-app.yaml

echo -e "${GREEN}✓ Updated k8s/streamlit-app.yaml with image: ${FULL_IMAGE}${NC}"

# =============================================================================
# Step 5: Configure Secrets
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 5: Configuring Secrets${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -z "$GROQ_API_KEY" ]; then
  echo -e "${YELLOW}Enter your Groq API key:${NC}"
  read -r GROQ_API_KEY
fi

# Encode and update secret
ENCODED_KEY=$(echo -n "$GROQ_API_KEY" | base64)
sed -i "s|groq-api-key: .*|groq-api-key: ${ENCODED_KEY}|g" k8s/secrets.yaml

echo -e "${GREEN}✓ Secret configured${NC}"

# =============================================================================
# Step 6: Deploy to GKE
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 6: Deploying to GKE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Apply all manifests
kubectl apply -k k8s/

echo -e "${GREEN}✓ Manifests applied${NC}"

# =============================================================================
# Step 7: Wait for Deployments
# =============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 7: Waiting for Deployments${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

NAMESPACE="rag-historian"

echo -e "${YELLOW}Waiting for ChromaDB...${NC}"
kubectl -n ${NAMESPACE} rollout status statefulset/chromadb --timeout=180s

echo -e "${YELLOW}Waiting for Streamlit App...${NC}"
kubectl -n ${NAMESPACE} rollout status deployment/streamlit-app --timeout=300s

# =============================================================================
# Step 8: Show Status
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Pods:${NC}"
kubectl -n ${NAMESPACE} get pods

echo -e "\n${YELLOW}Services:${NC}"
kubectl -n ${NAMESPACE} get svc

echo -e "\n${YELLOW}Ingress:${NC}"
kubectl -n ${NAMESPACE} get ingress

# Get external IP
echo -e "\n${YELLOW}Getting external IP (may take a few minutes)...${NC}"
EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
  EXTERNAL_IP=$(kubectl -n ${NAMESPACE} get ingress rag-historian-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  if [ -z "$EXTERNAL_IP" ]; then
    echo -e "  Waiting for external IP..."
    sleep 10
  fi
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Access Your Application${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  ${BLUE}Streamlit App:${NC} http://${EXTERNAL_IP}/"
echo -e "${GREEN}========================================${NC}"

