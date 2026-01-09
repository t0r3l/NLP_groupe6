#!/bin/bash
# =============================================================================
# RAG Historian - Kubernetes Deployment Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RAG Historian - K8s Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
NAMESPACE="rag-historian"
IMAGE_NAME="nlp_groupe6-streamlit-app"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"  # Set your registry, e.g., "docker.io/username" or "ghcr.io/username"

# Step 1: Build Docker image
echo -e "\n${YELLOW}Step 1: Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Step 2: Tag and push to registry (if registry is set)
if [ -n "$REGISTRY" ]; then
    echo -e "\n${YELLOW}Step 2: Pushing to registry ${REGISTRY}...${NC}"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    
    # Update the image in the deployment
    echo -e "${YELLOW}Updating image reference in deployment...${NC}"
    sed -i "s|image: rag-historian-app:latest|image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" k8s/streamlit-app.yaml
else
    echo -e "\n${YELLOW}Step 2: Skipping push (no REGISTRY set)${NC}"
    echo -e "${YELLOW}For local testing with minikube, run: minikube image load ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
fi

# Step 3: Create secret for Groq API key
echo -e "\n${YELLOW}Step 3: Setting up secrets...${NC}"
if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}Warning: GROQ_API_KEY environment variable not set!${NC}"
    echo -e "${YELLOW}Set it with: export GROQ_API_KEY='your-key-here'${NC}"
else
    # Encode the API key
    ENCODED_KEY=$(echo -n "$GROQ_API_KEY" | base64)
    sed -i "s|groq-api-key: .*|groq-api-key: ${ENCODED_KEY}|g" k8s/secrets.yaml
    echo -e "${GREEN}Groq API key configured in secrets.yaml${NC}"
fi

# Step 4: Apply Kubernetes manifests
echo -e "\n${YELLOW}Step 4: Applying Kubernetes manifests...${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found. Please install kubectl.${NC}"
    exit 1
fi

# Apply with kustomize
kubectl apply -k k8s/

# Step 5: Wait for deployments
echo -e "\n${YELLOW}Step 5: Waiting for deployments to be ready...${NC}"
kubectl -n ${NAMESPACE} rollout status statefulset/chromadb --timeout=120s
kubectl -n ${NAMESPACE} rollout status deployment/streamlit-app --timeout=180s

# Step 6: Show status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Pods:${NC}"
kubectl -n ${NAMESPACE} get pods

echo -e "\n${YELLOW}Services:${NC}"
kubectl -n ${NAMESPACE} get svc

echo -e "\n${YELLOW}To access the application:${NC}"
echo -e "  - Port-forward: kubectl -n ${NAMESPACE} port-forward svc/streamlit-app 8501:8501"
echo -e "  - Or configure Ingress with your domain in k8s/ingress.yaml"

