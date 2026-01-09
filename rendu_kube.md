# RAG Historian - Déploiement Kubernetes sur GKE

## 📋 Table des matières

1. [Architecture du déploiement](#-architecture-du-déploiement)
2. [Composants Kubernetes](#-composants-kubernetes)
3. [Étapes de déploiement](#-étapes-de-déploiement)
   - [Étape 1 : Création du Dockerfile](#étape-1--création-du-dockerfile)
   - [Étape 2 : Configuration Docker Compose (test local)](#étape-2--configuration-docker-compose-test-local)
   - [Étape 3 : Build des images Docker](#étape-3--build-des-images-docker)
   - [Étape 4 : Test local avec Docker Compose](#étape-4--test-local-avec-docker-compose)
   - [Étape 5 : Configuration GCP et Terraform](#étape-5--configuration-gcp-et-terraform)
   - [Étape 6 : Création du cluster GKE avec Terraform](#étape-6--création-du-cluster-gke-avec-terraform)
   - [Étape 7 : Push de l'image vers Artifact Registry](#étape-7--push-de-limage-vers-artifact-registry)
   - [Étape 8 : Configuration des manifests Kubernetes](#étape-8--configuration-des-manifests-kubernetes)
   - [Étape 9 : Déploiement sur GKE](#étape-9--déploiement-sur-gke)
   - [Étape 10 : Vérification et accès](#étape-10--vérification-et-accès)

---

## 🏗️ Architecture du déploiement

```
                         ┌──────────────────────────────────────┐
                         │         API EXTERNE - GROQ           │
                         │      https://api.groq.com/openai/v1  │
                         │  ┌────────────────────────────────┐  │
                         │  │  LLM: LLaMA 3 / Mixtral        │  │
                         │  │  Génération des réponses       │  │
                         │  └────────────────────────────────┘  │
                         └──────────────────┬───────────────────┘
                                            │
                                            │ HTTPS (API REST)
                                            │ Authorization: Bearer <GROQ_API_KEY>
                                            │
┌───────────────────────────────────────────┼───────────────────────────────────────┐
│                              GOOGLE CLOUD │PLATFORM                               │
│                                           │                                       │
│  ┌────────────────────────────────────────┼───────────────────────────────────┐  │
│  │                        GKE CLUSTER (rag│-historian-cluster)                │  │
│  │                          Zone: europe-west3-a                              │  │
│  │                                        │                                   │  │
│  │  ┌─────────────────────────────────────┼────────────────────────────────┐ │  │
│  │  │                    NAMESPACE: rag-historian                          │ │  │
│  │  │                                     │                                │ │  │
│  │  │   ┌─────────────────┐      ┌────────┼─────────────────────────────┐  │ │  │
│  │  │   │    INGRESS      │      │        │     CONFIGMAP               │  │ │  │
│  │  │   │   (GCE LB)      │      │  CHROMADB_HOST: chromadb             │  │ │  │
│  │  │   │   Port 80/443   │      │  CHROMADB_PORT: 8000                 │  │ │  │
│  │  │   └────────┬────────┘      └──────────────────────────────────────┘  │ │  │
│  │  │            │                                                         │ │  │
│  │  │            ▼                  ┌────────────────────────────────────┐ │ │  │
│  │  │   ┌─────────────────┐         │           SECRET                   │ │ │  │
│  │  │   │    SERVICE      │         │  groq-api-key: <base64>  ─────────────┘ │  │
│  │  │   │  streamlit-app  │         │  (Clé pour API Groq)               │   │  │
│  │  │   │  ClusterIP:8501 │         └────────────────────────────────────┘   │  │
│  │  │   └────────┬────────┘                                                  │  │
│  │  │            │                                                           │  │
│  │  │            ▼                                                           │  │
│  │  │   ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │   │                    DEPLOYMENT: streamlit-app                    │ │  │
│  │  │   │  ┌───────────────────────────────────────────────────────────┐  │ │  │
│  │  │   │  │  POD                                                      │  │ │  │
│  │  │   │  │  ┌──────────────┐  ┌────────────────────────────────────┐ │  │ │  │
│  │  │   │  │  │initContainer │  │    Container: streamlit-app       │ │  │ │  │
│  │  │   │  │  │wait-chromadb │─▶│    nlp_groupe6-streamlit-app      │─┼──┼─┼──┼──── HTTPS ──▶ API Groq
│  │  │   │  │  │  (busybox)   │  │    Port: 8501                     │ │  │ │  │     (Génération LLM)
│  │  │   │  │  └──────────────┘  │    ┌────────────────────────────┐ │ │  │ │  │
│  │  │   │  │                    │    │  Flux RAG:                 │ │ │  │ │  │
│  │  │   │  │                    │    │  1. Question utilisateur   │ │ │  │ │  │
│  │  │   │  │                    │    │  2. Embedding (MiniLM)     │ │ │  │ │  │
│  │  │   │  │                    │    │  3. Recherche ChromaDB ────┼─┼─┼──┼─┼──┼──▶ ChromaDB
│  │  │   │  │                    │    │  4. Contexte récupéré      │ │ │  │ │  │
│  │  │   │  │                    │    │  5. Appel API Groq ────────┼─┼─┼──┼─┼──┼──▶ LLM Groq
│  │  │   │  │                    │    │  6. Réponse générée        │ │ │  │ │  │
│  │  │   │  │                    │    └────────────────────────────┘ │ │  │ │  │
│  │  │   │  │                    └────────────────────────────────────┘ │  │ │  │
│  │  │   │  └───────────────────────────────────────────────────────────┘  │ │  │
│  │  │   └─────────────────────────────────────────────────────────────────┘ │  │
│  │  │                                    │                                   │  │
│  │  │                                    │ HTTP (port 8000)                  │  │
│  │  │                                    ▼                                   │  │
│  │  │   ┌─────────────────┐                                                  │  │
│  │  │   │    SERVICE      │                                                  │  │
│  │  │   │    chromadb     │                                                  │  │
│  │  │   │  ClusterIP:8000 │                                                  │  │
│  │  │   └────────┬────────┘                                                  │  │
│  │  │            │                                                           │  │
│  │  │            ▼                                                           │  │
│  │  │   ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │   │                   STATEFULSET: chromadb                         │ │  │
│  │  │   │  ┌───────────────────────────────────────────────────────────┐  │ │  │
│  │  │   │  │  POD chromadb-0                                           │  │ │  │
│  │  │   │  │  ┌─────────────────────────────────────────────────────┐  │  │ │  │
│  │  │   │  │  │  Container: chromadb/chroma:latest                  │  │  │ │  │
│  │  │   │  │  │  Port: 8000 | Stockage vectoriel des embeddings     │  │  │ │  │
│  │  │   │  │  │  CPU: 250m-1000m | RAM: 512Mi-2Gi                   │  │  │ │  │
│  │  │   │  │  └─────────────────────────────────────────────────────┘  │  │ │  │
│  │  │   │  └───────────────────────────┬───────────────────────────────┘  │ │  │
│  │  │   └──────────────────────────────┼──────────────────────────────────┘ │  │
│  │  │                                  │                                    │  │
│  │  │   ┌──────────────────────────────▼──────────────────────────────────┐ │  │
│  │  │   │        PersistentVolumeClaim: chromadb-pvc (10Gi)               │ │  │
│  │  │   │              storageClass: standard-rw (GCE PD)                 │ │  │
│  │  │   │              Stockage persistant des embeddings Wikipedia       │ │  │
│  │  │   └─────────────────────────────────────────────────────────────────┘ │  │
│  │  └───────────────────────────────────────────────────────────────────────┘  │
│  │                                                                             │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  │               NODE POOL (2 nodes, auto-scale 1-5)                     │ │
│  │  │               Machine: e2-standard-2 (2 vCPU, 8GB)                    │ │
│  │  │               Disk: 50GB pd-standard | Preemptible: true              │ │
│  │  └───────────────────────────────────────────────────────────────────────┘ │
│  └─────────────────────────────────────────────────────────────────────────────┘
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │            ARTIFACT REGISTRY: rag-historian                               │ │
│  │     europe-west3-docker.pkg.dev/{PROJECT}/rag-historian                   │ │
│  │                   └── rag-historian-app:latest                            │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    │
                                    │ HTTP/HTTPS
                                    ▼
                         ┌──────────────────────┐
                         │       INTERNET       │
                         │      (Utilisateurs)  │
                         └──────────────────────┘
```

### Flux de données RAG

```
┌─────────────┐     ┌─────────────────────────────────────────────────────────────────┐
│ Utilisateur │     │                    POD: streamlit-app                           │
│             │     │  ┌─────────────────────────────────────────────────────────────┐│
│  "Qui a     │────▶│  │ 1. Réception question                                       ││
│   fondé     │     │  │ 2. Embedding local (MiniLM) ──────────────────────────────┐ ││
│   l'Empire  │     │  │ 3. Recherche similarité ──────────────────────────────────┼─┼┼──▶ ChromaDB
│   du Mali?" │     │  │ 4. Récupération contexte (chunks Wikipedia) ◀─────────────┘ ││     (VectorDB)
│             │     │  │ 5. Construction prompt (question + contexte) ───────────────┼┼──▶ API Groq
│             │◀────│  │ 6. Réponse LLM ◀────────────────────────────────────────────┼┼────(LLM)
│ "Soundiata  │     │  │ 7. Affichage dans Streamlit                                 ││
│  Keïta..."  │     │  └─────────────────────────────────────────────────────────────┘│
└─────────────┘     └─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Composants Kubernetes

| Ressource | Nom | Description |
|-----------|-----|-------------|
| **Namespace** | `rag-historian` | Isolation des ressources de l'application |
| **Deployment** | `streamlit-app` | Application Streamlit RAG (1 replica) |
| **StatefulSet** | `chromadb` | Base de données vectorielle ChromaDB (1 replica) |
| **Service** | `streamlit-app` | ClusterIP:8501 - Expose le frontend en interne |
| **Service** | `chromadb` | ClusterIP:8000 - Communication interne avec ChromaDB |
| **Ingress** | `rag-historian-ingress` | Load Balancer GCP pour accès externe HTTP/HTTPS |
| **PVC** | `chromadb-pvc` | Stockage persistant 10Gi pour les embeddings |
| **ConfigMap** | `rag-historian-config` | Variables d'environnement (CHROMADB_HOST, CHROMADB_PORT) |
| **Secret** | `rag-historian-secrets` | Clé API Groq (encodée en base64) |

### 🔌 Services Externes

| Service | URL | Rôle |
|---------|-----|------|
| **API Groq** | `https://api.groq.com/openai/v1` | LLM pour génération des réponses (LLaMA 3, Mixtral) |

### 🔄 Flux RAG (Retrieval-Augmented Generation)

1. **Question** → L'utilisateur pose une question via l'interface Streamlit
2. **Embedding** → La question est convertie en vecteur par MiniLM (local, dans le pod)
3. **Retrieval** → ChromaDB recherche les chunks Wikipedia les plus similaires
4. **Contexte** → Les chunks pertinents sont récupérés (métadonnées: entité, région, période)
5. **Génération** → L'API Groq reçoit (question + contexte) et génère une réponse naturelle
6. **Affichage** → La réponse est affichée avec les sources utilisées

---

## 🚀 Étapes de déploiement

### Étape 1 : Création du Dockerfile

Créer un Dockerfile multi-stage optimisé pour l'application Streamlit :

```dockerfile
# =============================================================================
# Stage 1: Builder - Installation des dépendances
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Installation des outils de build et uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/* /root/.cache

# Copie des fichiers de dépendances
COPY pyproject.toml Readme.md ./
COPY src_rag/ ./src_rag/

# Installation des packages Python (CPU-only pour PyTorch)
RUN uv pip install --system --no-cache --compile-bytecode \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -e .

# =============================================================================
# Stage 2: Runtime - Image de production minimale
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="NLP Groupe 6" \
      version="1.0" \
      description="RAG Historian - Streamlit App"

WORKDIR /app

# Création utilisateur non-root et installation curl
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copie des packages Python depuis le builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code applicatif
COPY --chown=appuser:appuser src_rag/ ./src_rag/
COPY --chown=appuser:appuser app.py config.yml ./
COPY --chown=appuser:appuser data/ ./data/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

### Étape 2 : Configuration Docker Compose (test local)

Créer le fichier `docker-compose.yml` :

```yaml
services:
  streamlit-app:
    image: nlp_groupe6-streamlit-app:latest
    container_name: rag-historian-app
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
    depends_on:
      chromadb:
        condition: service_healthy
    networks:
      - frontend
      - backend
    volumes:
      - ./data:/app/data:ro

  chromadb:
    image: chromadb/chroma:latest
    container_name: rag-historian-chromadb
    ports:
      - "8000:8000"
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=FALSE
    volumes:
      - chroma_data:/chroma/chroma
    networks:
      - frontend
      - backend
    healthcheck:
      test: ["CMD-SHELL", "timeout 2 bash -c '</dev/tcp/localhost/8000' || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

volumes:
  chroma_data:
    driver: local
```

---

### Étape 3 : Build des images Docker

```bash
# Se placer dans le répertoire du projet
cd /home/torel/Projects/NLP/NLP_groupe6

# Build de l'image Streamlit
docker build -t nlp_groupe6-streamlit-app:latest .

# Vérifier que l'image a été créée
docker images | grep nlp_groupe6
```

**Résultat attendu :**
```
nlp_groupe6-streamlit-app    latest    dc88a1ac7339   XX hours ago   13.7GB
```

---

### Étape 4 : Test local avec Docker Compose

```bash
# Créer le fichier .env avec la clé API
echo "GROQ_API_KEY=votre_clé_api_groq" > .env

# Démarrer les services
docker-compose up -d

# Vérifier que les containers sont en cours d'exécution
docker ps

# Tester les endpoints
curl http://localhost:8501/_stcore/health      # Streamlit
curl http://localhost:8000/api/v2/heartbeat    # ChromaDB

# Voir les logs en cas de problème
docker-compose logs -f

# Arrêter les services
docker-compose down
```

**Résultat attendu :**
```
NAMES                    STATUS                    PORTS
rag-historian-app        Up X seconds (healthy)    0.0.0.0:8501->8501/tcp
rag-historian-chromadb   Up X seconds (healthy)    0.0.0.0:8000->8000/tcp
```

---

### Étape 5 : Configuration GCP et Terraform

#### 5.1 Installation des outils requis

```bash
# Installer gcloud CLI (si pas déjà fait)
# https://cloud.google.com/sdk/docs/install

# Installer kubectl
# https://kubernetes.io/docs/tasks/tools/

# Installer Terraform
brew install terraform   # macOS
# ou
sudo apt-get install terraform   # Ubuntu/Debian

# Installer le plugin d'authentification GKE
gcloud components install gke-gcloud-auth-plugin
```

#### 5.2 Authentification GCP

```bash
# Connexion à GCP
gcloud auth login

# Définir le projet
export GCP_PROJECT_ID="votre-project-id"
gcloud config set project $GCP_PROJECT_ID

# Activer les APIs nécessaires
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable compute.googleapis.com
```

#### 5.3 Configuration Terraform

```bash
# Se placer dans le répertoire Terraform
cd terraform/gke

# Créer le fichier de variables
cat > terraform.tfvars <<EOF
gcp_project_id = "votre-project-id"
gcp_region     = "europe-west3"
gcp_zone       = "europe-west3-a"
cluster_name   = "rag-historian-cluster"
EOF
```

---

### Étape 6 : Création du cluster GKE avec Terraform

```bash
# Initialiser Terraform
terraform init

# Prévisualiser les changements
terraform plan

# Créer l'infrastructure (cluster GKE + Artifact Registry)
terraform apply

# Confirmer avec "yes" quand demandé
```

**Ressources créées :**
- Cluster GKE `rag-historian-cluster`
- Node Pool avec 2 nodes e2-standard-2
- Artifact Registry `rag-historian`

```bash
# Configurer kubectl pour utiliser le cluster
gcloud container clusters get-credentials rag-historian-cluster \
    --zone europe-west3-a \
    --project $GCP_PROJECT_ID

# Vérifier la connexion
kubectl get nodes
```

---

### Étape 7 : Push de l'image vers Artifact Registry

```bash
# Variables
export GCP_PROJECT_ID="votre-project-id"
export GCP_REGION="europe-west3"
export REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/rag-historian"

# Configurer Docker pour Artifact Registry
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet

# Taguer l'image locale pour le registry
docker tag nlp_groupe6-streamlit-app:latest ${REGISTRY_URL}/rag-historian-app:latest

# Pousser l'image vers Artifact Registry
docker push ${REGISTRY_URL}/rag-historian-app:latest

# Vérifier que l'image est bien uploadée
gcloud artifacts docker images list ${REGISTRY_URL}
```

---

### Étape 8 : Configuration des manifests Kubernetes

#### 8.1 Mettre à jour l'image dans le manifest

```bash
# Mettre à jour le chemin de l'image dans streamlit-app.yaml
sed -i "s|image: nlp_groupe6-streamlit-app:latest|image: ${REGISTRY_URL}/rag-historian-app:latest|g" k8s/streamlit-app.yaml
```

#### 8.2 Configurer le secret avec la clé API Groq

```bash
# Encoder la clé API en base64
export GROQ_API_KEY="votre_clé_api_groq"
ENCODED_KEY=$(echo -n "$GROQ_API_KEY" | base64)

# Mettre à jour le fichier secrets.yaml
sed -i "s|groq-api-key: .*|groq-api-key: ${ENCODED_KEY}|g" k8s/secrets.yaml
```

#### 8.3 Structure des manifests K8s

```
k8s/
├── namespace.yaml       # Namespace rag-historian
├── secrets.yaml         # Secret pour GROQ_API_KEY
├── configmap.yaml       # ConfigMap (CHROMADB_HOST, CHROMADB_PORT)
├── chromadb.yaml        # StatefulSet + Service + PVC ChromaDB
├── streamlit-app.yaml   # Deployment + Service Streamlit
├── ingress.yaml         # Ingress GCE Load Balancer
└── kustomization.yaml   # Configuration Kustomize
```

---

### Étape 9 : Déploiement sur GKE

```bash
# Appliquer tous les manifests avec Kustomize
kubectl apply -k k8s/

# Vérifier le déploiement
kubectl -n rag-historian get all

# Attendre que ChromaDB soit prêt
kubectl -n rag-historian rollout status statefulset/chromadb --timeout=180s

# Attendre que Streamlit soit prêt
kubectl -n rag-historian rollout status deployment/streamlit-app --timeout=300s

# Voir les logs des pods
kubectl -n rag-historian logs -f deployment/streamlit-app
kubectl -n rag-historian logs -f statefulset/chromadb
```

---

### Étape 10 : Vérification et accès

#### 10.1 Vérifier l'état des ressources

```bash
# Voir tous les pods
kubectl -n rag-historian get pods -o wide

# Voir les services
kubectl -n rag-historian get svc

# Voir l'ingress et obtenir l'IP externe
kubectl -n rag-historian get ingress

# Décrire l'ingress pour plus de détails
kubectl -n rag-historian describe ingress rag-historian-ingress
```

#### 10.2 Obtenir l'IP externe

```bash
# Attendre l'attribution de l'IP (peut prendre 2-5 minutes)
EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
  echo "Waiting for external IP..."
  EXTERNAL_IP=$(kubectl -n rag-historian get ingress rag-historian-ingress \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
  sleep 10
done
echo "Application accessible à : http://${EXTERNAL_IP}/"
```

#### 10.3 Tester l'application

```bash
# Tester le health check
curl http://${EXTERNAL_IP}/_stcore/health

# Ouvrir dans le navigateur
echo "Ouvrez : http://${EXTERNAL_IP}/"
```

---

## 🧹 Nettoyage (fin du lab)

```bash
# Supprimer les ressources Kubernetes
kubectl delete -k k8s/

# Détruire l'infrastructure Terraform
cd terraform/gke
terraform destroy

# Confirmer avec "yes"
```

---

## 📊 Résumé des commandes

| Étape | Commande principale |
|-------|---------------------|
| Build Docker | `docker build -t nlp_groupe6-streamlit-app:latest .` |
| Test local | `docker-compose up -d` |
| Auth GCP | `gcloud auth login` |
| Créer cluster | `terraform apply` |
| Config kubectl | `gcloud container clusters get-credentials ...` |
| Push image | `docker push ${REGISTRY_URL}/rag-historian-app:latest` |
| Déployer K8s | `kubectl apply -k k8s/` |
| Vérifier | `kubectl -n rag-historian get all` |
| Obtenir IP | `kubectl -n rag-historian get ingress` |
| Nettoyer | `terraform destroy` |

---

## 👥 Auteurs

**NLP Groupe 6** - ESGI 5IABD1 2026

