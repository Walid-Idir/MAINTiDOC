# NOTE TECHNIQUE DU PROJET MAINTiDOC
## Pipeline RAG Multimodal & Orchestration Agentique pour la Maintenance Industrielle

---

### 1. Présentation du Projet et Objectifs

Le projet **MAINTiDOC** vise à concevoir et déployer un système d'interrogation documentaire technique fondé sur une architecture **RAG (Retrieval-Augmented Generation) Multimodale et Agentique**, 100% locale et open source.

Le système permet aux ingénieurs et techniciens de maintenance de :
- Poser des questions en langage naturel sur des manuels machines et articles scientifiques.
- Obtenir des réponses factuelles et vérifiables, accompagnées de citations précises des sources (numéro de page, tableau ou figure).
- Interroger à la fois des fragments de **texte**, des **tableaux structurés de maintenance** et des **schémas visuels complexes**.
- Bénéficier d'une orchestration autonome via un **Agent LangChain** capable de sélectionner dynamiquement l'outil de recherche adéquat.

---

### 2. Architecture Globale du Système

```
                                  DOCUMENT PDF MULTIMODAL
                  (Texte technique, Tableaux de maintenance, Schémas techniques)
                                            │
                                            ▼
                          PyMuPDF (fitz) / LlamaIndex Extractor
                         ┌──────────────────┼──────────────────┐
                         │                  │                  │
                         ▼                  ▼                  ▼
                  Fragments Texte      Tableaux MD       Figures PNG
                         │                  │                  │
                         │                  │                  ▼
                         │                  │            Modèle CLIP (512D)
                         │                  │       (Zero-Shot Tags & Desc.)
                         │                  │                  │
                         ▼                  ▼                  ▼
                   BAAI/bge-small-en-v1.5 (384D)       Table PGVector CLIP
                         │                              (image_clip_embeddings)
                         ▼
                   Table Principale PGVector
                   (data_document_embeddings)
                         ▲
                         │
        ┌────────────────┴────────────────────────────────┐
        │                                                 │
        ▼                                                 ▼
[Mode 1 : RAG Direct LlamaIndex]            [Mode 2 : Agent LangChain (ReAct)]
  - Top-K Cosine Retrieval                    - Outil : search_text_and_tables
  - Prompt Template factuel                   - Outil : search_visual_clip
  - LLM Qwen 2.5 (Ollama)                     - Outil : inspect_documents_in_db
        │                                     - Boucle Pensée / Action / Obs.
        │                                     - LLM Qwen 2.5 (Ollama)
        └────────────────┬────────────────────────────────┘
                         ▼
             INTERFACE STREAMLIT PRO
             (Q&A, Citations, Vision CLIP, Monitoring)
```

---

### 3. Modèles Choisis et Justifications Techniques

| Rôle | Modèle Choisi | Dimension / Poids | Justification & Avantages |
| :--- | :--- | :--- | :--- |
| **Embeddings Textuels & Tableaux** | **`BAAI/bge-small-en-v1.5`** | 384 dimensions (~133 Mo) | Classé parmi les meilleurs modèles sur le benchmark MTEB (*Massive Text Embedding Benchmark*). Produit des vecteurs denses très compacts, idéaux pour une recherche sémantique rapide avec PGVector sans surcharger la mémoire vive. Inférence CPU quasi instantanée. |
| **Représentation Multimodale (Images)** | **`openai/clip-vit-base-patch32`** | 512 dimensions (~600 Mo) | Architecture ViT (*Vision Transformer*) projetant texte et images dans un espace vectoriel partagé. Permet à la fois la recherche *Text-to-Image* directe par distance cosinus et la classification zero-shot de schémas techniques sans entraînement préalable. |
| **Génération & Raisonnement (LLM)** | **`Qwen 2.5 (1.5B)`** via **Ollama** | 1.5 milliard de paramètres (~986 Mo quantifié) | Modèle open source ultra-compact offrant d'excellentes capacités de raisonnement technique et de factualité. Supporte nativement le *Tool Calling / Function Calling* et dispose d'une fenêtre de contexte étendue (jusqu'à 32k tokens), tout en tournant à haute vitesse en local. |

---

### 4. Outils et Composants Logiciels Utilisés

1. **Extraction de données (`extractor.py`)** :
   - `PyMuPDF` (`fitz`) : extraction rapide et fidèle du texte par page, extraction raster des images au format PNG, et détection des tables avec conversion directe en Markdown (`find_tables()`).
   - Analyseur heuristique de légendes contextuelles (capture automatique des labels `Figure X:` et `Table X:`).

2. **Base de données Vectorielle (`PostgreSQL 16` + `PGVector`)** :
   - Hébergée localement dans l'environnement WSL2 Ubuntu.
   - Extension `vector` native avec index HNSW (*Hierarchical Navigable Small World*) pour une recherche par similarité cosinus `<=>` en temps sub-milliseconde.
   - Deux tables dédiées :
     - `data_document_embeddings` (384 dimensions) pour les fragments textuels et les tableaux.
     - `image_clip_embeddings` (512 dimensions) pour les vecteurs visuels des schémas.

3. **Orchestration Agentique (`agentic_rag.py`)** :
   - `LangChain` & `langchain-ollama` : configuration de l'agent autonome avec la boucle **ReAct** (*Reasoning + Acting*).
   - Outils mis à disposition de l'agent :
     - `search_text_and_tables` : recherche dans les tableaux et textes.
     - `search_visual_clip` : recherche visuelle de diagrammes.
     - `inspect_documents_in_db` : diagnostic de la base.

4. **Interface Graphique Utilisateur (`app.py`)** :
   - `Streamlit` (version 1.44) : interface web interactive avec upload de documents, basculement dynamique de mode (Agent vs RAG direct), explorateur de figures et monitoring de la base de données.

---

### 5. Limites du Système et Pistes d'Amélioration

| Composant | Limites Actuelles | Pistes d'Amélioration Futures |
| :--- | :--- | :--- |
| **Résolution Visuelle CLIP** | CLIP standard découpe en patchs de 32x32 pixels sur une image redimensionnée en 224x224. Sur des schémas P&ID industriels très denses en tuyauteries ou annotations minuscules, certains détails textuels au sein de l'image peuvent être perdus. | Remplacer ou compléter CLIP par un modèle de type **ColPali** (*Vision-Language Retrieval*) ou **CLIP ViT-L/14@336px** qui préserve la haute résolution des documents. |
| **Taille du LLM (1.5B)** | Bien que rapide et factuel sur les questions directes, le modèle 1.5B peut être limité lorsqu'un problème nécessite de combiner 4 ou 5 étapes de calcul et de déduction logique consécutives. | Prévoir une bascule configurable vers **Qwen 2.5 7B** ou **14B** lorsque les ressources GPU locales le permettent. |
| **Tableaux imbriqués complexes** | L'extracteur PyMuPDF traite parfaitement les tableaux aux bordures définies, mais peut avoir des difficultés avec les cellules multi-lignes fusionnées verticalement sans lignes de démarcation. | Intégrer un parseur spécialisé comme **Table Transformer (TATR)** ou **Surya OCR** pour les documents scannés complexes. |
| **Multilinguisme des Schémas** | Les schémas annotés en allemand ou japonais nécessitent des prompts CLIP traduits pour garantir un score cosinus maximal. | Ajouter une étape de normalisation/traduction automatique de la requête utilisateur avant projection dans l'espace CLIP. |

---

### 6. Exemples Concrets de Questions / Réponses Validés

#### Exemple 1 : Question Technique sur Tableau Structuré (Couples de serrage)
- **Question** : *« Quels sont les couples de serrage pour la culasse et le flasque ? »*
- **Outil appelé par l'Agent** : `search_text_and_tables(query='couples de serrage culasse flasque')`
- **Source retrouvée** : `manuel_maintenance_compresseur_industriel.pdf`, Page 2, `Table 2: Spécifications de serrage dynamométrique et tolérances de montage`.
- **Réponse générée par le système** :
  > *« Les couples de serrage pour la culasse et le flasque sont les suivants :*
  > - *Culasse étage haute pression : M16 Classe 10.9, couple recommandé de **210 N·m** (Pâte anti-grippante Cuivre).*
  > - *Flasque d'aspiration vis : M12 Classe 8.8, couple recommandé de **85 N·m** (Huile moteur légère).*
  > 
  > *Source : Manuel de maintenance compresseur CPX-450, Page 2, Tableau 2. »*

---

#### Exemple 2 : Question Procédurale sur Spécifications Fluides (Vidange d'huile)
- **Question** : *« Quelle est la periodicite de vidange de l'huile du compresseur et quelle reference d'huile utiliser ? »*
- **Outil appelé** : `search_text_and_tables(query='periodicite vidange huile compresseur reference')`
- **Source retrouvée** : `manuel_maintenance_compresseur_industriel.pdf`, Page 1, `Table 1: Programme d'interventions préventives systématiques`.
- **Réponse générée par le système** :
  > *« Selon le programme d'interventions préventives :*
  > - *La périodicité de vidange d'huile est fixée à **4 000 heures** de fonctionnement.*
  > - *La référence recommandée est l'**Huile ISO VG 46 Synthétique 25 L (Réf: OIL-SYN-46)**.*
  > 
  > *Source : Manuel de maintenance compresseur CPX-450, Page 1, Tableau 1. »*

---

#### Exemple 3 : Question Visuelle d'Architecture avec CLIP (Schéma Transformer)
- **Question** : *« Quel est le diagramme d'architecture du modèle Transformer et où se trouve-t-il ? »*
- **Outil appelé par l'Agent** : `search_visual_clip(visual_query='Transformer architecture diagram')`
- **Source retrouvée** : `Attention is all you need.pdf`, Page 4, Image `page_4_img_1_129.png` (Similarité CLIP : `0.7712`).
- **Tags visuels détectés** : `transformer model architecture diagram with encoder and decoder blocks`.
- **Réponse générée par le système** :
  > *« Le diagramme d'architecture du modèle Transformer se trouve à la **page 4** de l'article "Attention is all you need" (Figure 1). Le schéma illustre le modèle complet Encodeur-Décodeur, composé d'empilements de blocs Multi-Head Attention et de connexions résiduelles avec Layer Normalization.*
  > 
  > *Source : Attention is all you need.pdf, Page 4, Figure 1. »*

---

#### Exemple 4 : Question Scientifique Mathématique (Formule Attention)
- **Question** : *« Quelle est la formule du Scaled Dot-Product Attention et comment est-elle calculée ? »*
- **Source retrouvée** : `Attention is all you need.pdf`, Page 4 (Texte documentaire).
- **Réponse générée par le système** :
  > *« La formule du Scaled Dot-Product Attention est :*
  > $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  > *Elle est calculée en effectuant le produit scalaire des requêtes (Queries $Q$) avec toutes les clés (Keys $K$), divisé par le facteur d'échelle $\sqrt{d_k}$, puis en appliquant une fonction softmax pour obtenir les poids associés aux valeurs (Values $V$).*
  > 
  > *Source : Attention is all you need.pdf, Page 4. »*

---

### 7. Conclusion et Vérification des Livrables

Tous les éléments constitutifs du cahier des charges sont intégralement réalisés et opérationnels :
1. ✅ **Code source documenté** et structuré en modules indépendants.
2. ✅ **Base PostgreSQL PGVector alimentée** avec texte, tableaux Markdown, figures PNG et métadonnées associées.
3. ✅ **Interface graphique Streamlit Enterprise** active sur `http://localhost:8501`.
4. ✅ **Note technique détaillée** formalisant les choix d'architecture, les modèles, les limites et les cas de tests réels.

