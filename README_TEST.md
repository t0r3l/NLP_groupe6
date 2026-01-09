# 🧪 Test de l'Application RAG Historian - Déploiement GKE

## 🌐 Accès à l'Application

**URL de production :** http://34.160.26.154/

---

## ✅ Tests de Validation

### 1. Vérification de la Disponibilité

```bash
# Health check Streamlit
curl http://34.160.26.154/_stcore/health
# Réponse attendue: ok

# Test de la page d'accueil
curl -s -o /dev/null -w "%{http_code}" http://34.160.26.154/
# Réponse attendue: 200
```

### 2. Vérification des Pods Kubernetes

```bash
# Vérifier que tous les pods sont Running
kubectl -n rag-historian get pods

# Résultat attendu:
# NAME                             READY   STATUS    RESTARTS   AGE
# chromadb-0                       1/1     Running   0          XXm
# streamlit-app-xxxxx-xxxxx        1/1     Running   0          XXm
```

### 3. Vérification des Services

```bash
# Liste des services
kubectl -n rag-historian get svc

# Résultat attendu:
# NAME            TYPE        CLUSTER-IP      PORT(S)
# chromadb        ClusterIP   10.x.x.x        8000/TCP
# streamlit-app   ClusterIP   10.x.x.x        8501/TCP
```

### 4. Vérification de l'Ingress

```bash
# Vérifier l'ingress et l'IP externe
kubectl -n rag-historian get ingress

# Résultat attendu:
# NAME                     CLASS   HOSTS   ADDRESS          PORTS
# rag-historian-ingress    gce     *       34.160.26.154    80
```

### 5. Vérification des Logs

```bash
# Logs Streamlit (dernières 20 lignes)
kubectl -n rag-historian logs deployment/streamlit-app --tail=20

# Logs ChromaDB
kubectl -n rag-historian logs statefulset/chromadb --tail=20
```

---

## 🎯 Tests Fonctionnels

### Test 1 : Page d'Accueil
1. Ouvrir http://34.160.26.154/ dans un navigateur
2. ✅ La page Streamlit "RAG Historian - Civilisations Africaines" doit s'afficher
3. ✅ Le titre "🌍 RAG Historian" doit être visible
4. ✅ La sidebar avec la configuration doit apparaître

### Test 2 : Connexion ChromaDB
1. Dans l'application, le modèle RAG doit se charger sans erreur
2. ✅ Message "Chargement du modèle RAG..." puis disparition du spinner
3. ✅ Aucune erreur de connexion dans les logs

```bash
# Vérifier la connexion ChromaDB depuis le pod
kubectl -n rag-historian exec deployment/streamlit-app -- curl -s chromadb:8000/api/v1/heartbeat
# Réponse attendue: {"nanosecond heartbeat":...}
```

### Test 3 : Requête RAG (Civilisations Africaines)
1. Entrer une question dans l'interface :
   - ✅ "Qui a fondé l'Empire du Mali ?"
   - ✅ "Quelle était la capitale de l'Empire du Ghana ?"
   - ✅ "Comment s'appelaient les guerrières du Dahomey ?"
   - ✅ "Quel roi a adopté le christianisme à Aksoum ?"

2. Vérifier la réponse :
   - ✅ Une réponse textuelle doit être générée par l'API Groq
   - ✅ Les sources (chunks Wikipedia) doivent être affichées
   - ✅ Les métadonnées (Entité, Région, Période) doivent apparaître
   - ✅ Le score "Reply Accuracy" doit s'afficher

### Test 4 : Performance
1. Soumettre plusieurs requêtes consécutives
2. ✅ Le temps de réponse doit être < 30 secondes
3. ✅ L'application ne doit pas crasher
4. ✅ Les logs ne doivent pas montrer d'erreurs API Groq

---

## 🔍 Debugging

### Problème : Page non accessible (502/504)

```bash
# Vérifier l'état du pod Streamlit
kubectl -n rag-historian describe pod -l app=streamlit-app

# Vérifier les events de l'ingress
kubectl -n rag-historian describe ingress rag-historian-ingress

# Vérifier le backend service de l'ingress
kubectl -n rag-historian get endpoints streamlit-app
```

### Problème : Erreur de connexion ChromaDB

```bash
# Vérifier que ChromaDB est running
kubectl -n rag-historian get pod chromadb-0

# Tester la connectivité interne
kubectl -n rag-historian exec deployment/streamlit-app -- nc -zv chromadb 8000

# Vérifier les logs ChromaDB
kubectl -n rag-historian logs chromadb-0 --tail=50
```

### Problème : Pas de réponse RAG / Erreur API Groq

```bash
# Vérifier les logs pour les erreurs API
kubectl -n rag-historian logs deployment/streamlit-app | grep -i error

# Vérifier que la clé GROQ est configurée
kubectl -n rag-historian get secret rag-historian-secrets -o jsonpath='{.data.groq-api-key}' | base64 -d

# Tester la connectivité vers Groq depuis le pod
kubectl -n rag-historian exec deployment/streamlit-app -- curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $(kubectl -n rag-historian get secret rag-historian-secrets -o jsonpath='{.data.groq-api-key}' | base64 -d)"
```

### Problème : Pod en CrashLoopBackOff

```bash
# Voir les événements du pod
kubectl -n rag-historian describe pod -l app=streamlit-app

# Voir les logs du conteneur précédent (avant crash)
kubectl -n rag-historian logs deployment/streamlit-app --previous
```

---

## 📊 Métriques et Monitoring

### Vérifier les ressources utilisées

```bash
# CPU/Mémoire des pods
kubectl -n rag-historian top pods

# Exemple de sortie:
# NAME                            CPU(cores)   MEMORY(bytes)
# chromadb-0                      50m          256Mi
# streamlit-app-xxx-xxx           100m         512Mi
```

### Vérifier le stockage PVC

```bash
# État des PVC
kubectl -n rag-historian get pvc

# Résultat attendu:
# NAME           STATUS   VOLUME       CAPACITY   ACCESS MODES
# chromadb-pvc   Bound    pvc-xxx      10Gi       RWO
```

### Vérifier les ressources du cluster

```bash
# Nodes du cluster
kubectl get nodes

# Utilisation des nodes
kubectl top nodes
```

---

## 🧹 Nettoyage (Fin du Lab)

```bash
# 1. Supprimer le namespace (tous les déploiements K8s)
kubectl delete namespace rag-historian

# 2. Supprimer l'IP statique réservée (si créée)
gcloud compute addresses delete rag-historian-ip --global

# 3. Détruire le cluster GKE avec Terraform
cd terraform/gke
terraform destroy -auto-approve

# 4. Supprimer les images Docker (optionnel)
gcloud artifacts docker images delete \
  europe-west3-docker.pkg.dev/rag-historian-gke-2026/rag-historian/rag-historian-app:latest
```

---

## 📝 Informations de Déploiement

| Ressource | Valeur |
|-----------|--------|
| **URL Production** | http://34.160.26.154/ |
| **Projet GCP** | rag-historian-gke-2026 |
| **Cluster GKE** | rag-historian-cluster |
| **Zone** | europe-west3-a |
| **Namespace** | rag-historian |
| **Image Docker** | `europe-west3-docker.pkg.dev/rag-historian-gke-2026/rag-historian/rag-historian-app:latest` |
| **API Externe** | api.groq.com (LLM) |
| **Embedding** | MiniLM (local, dans le pod) |
| **VectorDB** | ChromaDB (dans cluster) |

---

## 🔗 Liens Utiles

- **Application** : http://34.160.26.154/
- **GCP Console** : https://console.cloud.google.com/kubernetes/list?project=rag-historian-gke-2026
- **Artifact Registry** : https://console.cloud.google.com/artifacts?project=rag-historian-gke-2026

---

**Groupe 6 - ESGI 5IABD1 2026**
