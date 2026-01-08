# Rapport NLP - RAG sur les Civilisations Africaines

## Introduction

Ce rapport présente les résultats d'expérimentations sur un système RAG (Retrieval-Augmented Generation) appliqué à des données Wikipedia concernant les civilisations africaines précoloniales.

## Corpus

Notre corpus est constitué de **131 pages Wikipedia** sur les civilisations africaines précoloniales.

| Métrique | Valeur |
|----------|--------|
| Nombre de pages | 131 |
| Total tokens | 37,268 |
| Moyenne | 284.5 tokens |
| Médiane | 218.0 tokens |
| Min | 30 tokens |
| Max | 851 tokens |
| Écart-type | 204.9 |

Les pages sont relativement courtes (moyenne ~285 tokens), ce qui influence le choix des tailles de chunks optimales.

---

Pour évaluer les performances, nous utilisons:
- **MRR (Mean Reciprocal Rank)** : mesure la qualité du retrieval (plus c'est élevé, mieux c'est)
- **percent_correct** : pourcentage des réponses avec une similarité sémantique > 0.7 avec la réponse attendue
- **reply_similarity** : moyenne des similarités sémantiques entre les réponses générées et attendues

Nous privilégions le percent_correct car il représente le pourcentage de réponses considérées comme correctes, tandis que reply_similarity peut être faussé par des outliers étant donné qu'il est basé sur la moyenne.

---

## Test 1 : Comparaison des Embeddings avec différentes tailles de chunks

Pour commencer nos tests, nous avons comparé 5 modèles d'embeddings différents avec 3 tailles de chunks.

| embedding | chunk_size | nb_chunks | mrr     | percent_correct | reply_similarity |
|-----------|------------|-----------|---------|-----------------|------------------|
| miniLM    | 128        | 355       | ~0.0650 | ~0.60           | ~0.65            |
| miniLM    | 256        | 211       | ~0.1050 | ~0.70           | ~0.68            |
| miniLM    | 512        | 153       | ~0.2050 | ~0.80           | ~0.75            |
| mpnet     | 128        | 355       | ~0.0700 | ~0.50           | ~0.55            |
| mpnet     | 256        | 211       | ~0.1683 | ~0.70           | ~0.68            |
| mpnet     | 512        | 153       | ~0.2050 | ~0.70           | ~0.70            |
| bge       | 128        | 355       | ~0.0733 | ~0.60           | ~0.62            |
| bge       | 256        | 211       | ~0.1550 | ~0.60           | ~0.63            |
| bge       | 512        | 153       | ~0.1250 | ~0.80           | ~0.72            |
| e5        | 128        | 355       | ~0.0600 | ~0.70           | ~0.67            |

### Analyse

Nous observons que:
- Le modèle **miniLM** avec une chunk_size de 512 obtient les meilleurs résultats globaux (MRR=0.205, percent_correct=0.80)
- Les chunk sizes plus grandes (512) produisent généralement de meilleurs MRR


**Meilleur embedding sélectionné : miniLM** (score combiné le plus élevé)

---

## Test 2 : Tests avec chunk_size et overlap

La prochaine expérimentation vise à observer si l'on peut obtenir de meilleurs résultats avec un overlap de ~10%.

| chunk_size | overlap | nb_chunks | mrr     | percent_correct | reply_similarity |
|------------|---------|-----------|---------|-----------------|------------------|
| 128        | 12      | 369       | ~0.1400 | ~0.60           | ~0.7406          |
| 256        | 25      | 217       | ~0.2200 | ~0.70           | ~0.8019          |
| 512        | 51      | 153       | ~0.1500 | ~0.70           | ~0.7640          |


### Analyse


L'overlap augmente considérablement le MRR et la reply similarity des chunks 128 et 256 
mais leur percent correct stagne et les performances du meilleur chunk baissent.


---

## Test 3 : Tests avec chunk_size et small2big

Une autre approche pour améliorer le contexte des chunks est le small2big. L'algorithme implémenté sélectionne les 10 chunks les plus similaires puis les fusionne avec leurs chunks adjacents s'ils sont dans le top 10 pour enfin retourner au llm Les 5 groupes de chunks par ordre décroissant par rapport à leur score de similarité max.
.

| chunk_size | small2big | nb_chunks | mrr     | percent_correct | reply_similarity |
|------------|-----------|-----------|---------|-----------------|------------------|
| 128        | True      | 355       | ~0.2500 | ~0.60           | ~0.7019          |
| 256        | True      | 211       | ~0.5500 | ~0.70           | ~0.7985          |
| 512        | True      | 153       | ~0.4700 | ~0.70           | ~0.7584          |


### Analyse

Nous observons un meilleur MRR avec cette technique mais le percent correct stagne et la reply similarity baisse légèrement

## Test 4 : Tests avec chunk_size, overlap et metadata

Une autre idée d'amélioration est l'ajout de métadonnées (entité, région, période) aux chunks.

| chunk_size | overlap | add_metadata | nb_chunks | mrr     | percent_correct | reply_similarity |
|------------|---------|--------------|-----------|---------|-----------------|------------------|
| 128        | 12      | True         | 369       | ~0.0800 | ~0.80           | ~0.8603          |
| 256        | 25      | True         | 217       | ~0.1583 | ~0.80           | ~0.8261          |
| 512        | 51      | True         | 153       | ~0.1000 | ~0.70           | ~0.7177          |



### Analyse

En utilisant cette technique, le MRR se trouve considérablement diminué pour toutes les chunks size mais le percent correct et la reply similarity des chunk sizes 128 et 256 convergent vers le score obtenu lors du test 1 avec la chunk size à 512 sans overlap qui avait un percent correct de ~0.80 et une reply similarity de ~0.75 même si meilleur MRR

---


## Test 5 : Tests avec chunk_size, small2big et metadata

Voyons si utiliser le small2big plutôt que l'overlap peut nous aider à monter nos scores



| chunk_size | small2big | add_metadata | nb_chunk | mrr     | percent_correct | reply_similarity |
|------------|-----------|--------------|----------|---------|-----------------|------------------|
| 128        | True      | True         | 355      | ~0.3000 | ~0.80           | ~0.8575          |
| 256        | True      | True         | 211      | ~0.6500 | ~0.80           | ~0.8669          |
| 512        | True      | True         | 153      | ~0.5500 | ~0.70           | ~0.7744          |


### Analyse

Nous obtenons en effet le meilleur score global de tous nos tests avec la chunk size de 256.



## Tableau récapitulatif des meilleurs résultats


---

## Conclusion

Les résultats montrent clairement que:

1. **L'embedding miniLM** offre le meilleur compromis performance/rapidité pour ce corpus de données Wikipedia.

2. **La stratégie small2big plus performante que l'overlap**.

3. L'ajout des métadonnées aux chunks améliore considérablement la reply accuracy des petits chunks

4. La meilleure configuration trouvée est 

| chunk_size | small2big | add_metadata | nb_chunk | mrr     | percent_correct | reply_similarity |
|------------|-----------|--------------|----------|---------|-----------------|------------------|
| 256        | True      | True         | 211      | ~0.6500 | ~0.80           | ~0.8669          |