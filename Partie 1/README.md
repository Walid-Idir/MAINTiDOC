# PROJET MAINTiDOC - PARTIE 1 : PIPELINE RAG MULTIMODAL & INDEXATION TECHNIQUE

Système complet d'extraction multimodale, indexation vectorielle et Question-Réponse (RAG) avec des modèles 100% open source et locaux.

---

## 🏗️ Architecture du Projet

```
PROJET MAINTiDOC/Partie 1/
│
├── app.py                    # Interface graphique Web Streamlit (Chargement, Q&A, Recherche visuelle)
├── agentic_rag.py            # 🤖 Orchestrateur Agentique LangChain (Sélection d'outils, ReAct & PGVector)
├── config.py                 # Configuration centrale (DB PGVector, modèles BGE, CLIP, Ollama Qwen)
├── extractor.py              # Extraction multimodale (Texte, Tableaux Markdown, Images PNG & légendes)
├── image_processor.py        # Embeddings CLIP (512D), classification zero-shot & description textuelle
├── indexer.py                # Batch embeddings BGE (384D) & indexation dans PGVector
├── search_engine.py          # Moteur de recherche sémantique avec filtres par type
├── rag_pipeline.py           # Pipeline RAG complet avec LlamaIndex + Qwen 2.5 (Ollama)
├── cli_search.py             # Interface en ligne de commande unifiée (Recherche sémantique, CLIP, RAG)
├── main.py                   # Script d'exécution de bout en bout (Étapes 1 à 7)
├── create_maintenance_pdf.py # Générateur du document technique de maintenance industrielle
└── extracted_assets/         # Dossier des images extraites des PDFs
```

---

## 📋 Récapitulatif des 4 Exigences Réalisées

| Étape | Exigence | Statut | Solution Implémentée |
| :--- | :--- | :---: | :--- |
| **1** | **Sélection du document** | ✅ | Deux documents testés : `Attention is all you need.pdf` et `manuel_maintenance_compresseur_industriel.pdf` (contenant texte, tableaux de maintenance et schémas techniques). |
| **2** | **Extraction et indexation** | ✅ | Extraction avec **PyMuPDF / LlamaIndex**, production d'embeddings avec **`BAAI/bge-small-en-v1.5`** (384D) et stockage dans **PostgreSQL PGVector** (`docdb`). |
| **3** | **Traitement des images (CLIP)** | ✅ | Encodage des images avec **`openai/clip-vit-base-patch32`** (512D), classification visuelle zero-shot, génération de descriptions textuelles contextuelles et recherche Text-to-Image dans PGVector. |
| **4** | **Pipeline RAG (Qwen via Ollama)** | ✅ | Intégration de **Qwen 2.5 (1.5B)** via **Ollama**. Récupération des fragments pertinents (Top-K) et génération de réponses factuelles fondées sur le contexte avec citation stricte des sources. |

---

## 🚀 Guide d'Utilisation

### 1. Activer l'environnement virtuel
```powershell
& "c:\Users\user\PROJET MAINTiDOC\venv\Scripts\Activate.ps1"
cd "c:\Users\user\PROJET MAINTiDOC\Partie 1"
```

### 2. Lancer l'interface Web Streamlit
```powershell
python -m streamlit run app.py
```
Accédez ensuite à l'application dans votre navigateur : **`http://localhost:8501`**

### 3. Ré-indexer les documents en ligne de commande (optionnel)
```powershell
python main.py
```

### 4. Poser des questions avec l'Agent LangChain (Orchestration Agentique)
```powershell
# L'Agent sélectionne tout seul l'outil texte/tableaux :
python agentic_rag.py --query "Quels sont les couples de serrage pour la culasse et le flasque ?"

# L'Agent sélectionne tout seul l'outil visuel CLIP pour les diagrammes :
python agentic_rag.py --query "Quel est le diagramme d'architecture du modèle Transformer et où se trouve-t-il ?"

# L'Agent inspecte l'état complet de la base de données :
python agentic_rag.py --query "Quels documents et combien de tableaux sont disponibles dans la base ?"
```

### 5. Poser des questions avec le LLM en RAG direct (LlamaIndex)
```powershell
# Question sur le compresseur industriel (tableau de maintenance) :
python cli_search.py --ask "Quelle est la periodicite de vidange d'huile et quelle huile utiliser ?"

# Question sur les tolérances et couples de serrage :
python cli_search.py --ask "Quels sont les couples de serrage pour la culasse et le flasque ?"

# Question sur l'article Transformer :
python cli_search.py --ask "Quelle est la formule du Scaled Dot-Product Attention ?"
```

### 4. Recherche visuelle d'images via CLIP (Text-to-Image)
```powershell
# Retrouver les schémas techniques par description visuelle directe :
python cli_search.py --clip "circuit de lubrification et compresseur"
python cli_search.py --clip "neural network architecture with encoder and decoder blocks"
```

### 5. Recherche sémantique standard dans la base vectorielle
```powershell
python cli_search.py "intervalle de remplacement du filtre a air"
python cli_search.py "multi-head attention mechanism" --type text
```

