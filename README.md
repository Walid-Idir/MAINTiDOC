# MAINTiDOC - Intelligence Documentaire & Maintenance Industrielle

Projet d'extraction multimodale, indexation vectorielle et Question-Réponse (RAG) avec orchestration agentique LangChain, 100% open source et local.

---

## 🎯 Livrables Attendus - Partie 1 : Socle Documentaire et RAG Textuel

- **Code source documenté** : Extraction, indexation, recherche multimodale, génération et orchestration agentique.
- **Base PGVector alimentée** : Stockage PostgreSQL 16 des embeddings denses (BGE 384D) et visuels (CLIP 512D) avec métadonnées associées.
- **Interface Streamlit fonctionnelle** : Application web interactive (`app.py`) avec upload PDF, bascule de mode (Agent LangChain vs RAG Direct), vision CLIP et monitoring.
- **Note technique détaillée** : [`Partie 1/NOTE_TECHNIQUE.md`](Partie%201/NOTE_TECHNIQUE.md) décrivant les modèles choisis (BGE, CLIP, Qwen 2.5), les outils, les limites et 4 cas concrets de test validés.

---

## 🏗️ Architecture du Projet

```
MAINTiDOC/
│
├── requirements.txt            # Dépendances Python du projet
├── README.md                   # Présentation du dépôt
│
└── Partie 1/                   # 🚀 Partie 1 : Socle Documentaire & RAG Textuel
    ├── app.py                  # Interface Web Streamlit (Pro UI)
    ├── agentic_rag.py          # 🤖 Orchestrateur Agentique LangChain (ReAct & Outils)
    ├── rag_pipeline.py         # Pipeline RAG Direct LlamaIndex + Qwen 2.5 (Ollama)
    ├── image_processor.py      # Modèle CLIP (512D), Zero-Shot & Descriptions enrichies
    ├── extractor.py            # Extracteur multimodal PyMuPDF (Texte, Tables MD, Images PNG)
    ├── indexer.py              # Indexeur vectoriel BGE (384D) dans PostgreSQL PGVector
    ├── search_engine.py        # Moteur de recherche sémantique multi-type
    ├── cli_search.py           # Interface en ligne de commande unifiée
    ├── main.py                 # Pipeline d'exécution automatisé de bout en bout
    ├── config.py               # Paramètres DB, modèles et chemins
    ├── schema_pgvector.sql     # Schémas DDL des tables PostgreSQL
    ├── NOTE_TECHNIQUE.md       # 📄 Note technique complète (Modèles, Limites, Cas de test)
    ├── README.md               # Guide d'utilisation détaillé de la Partie 1
    │
    ├── Attention is all you need.pdf                  # Document 1 (Article IA Transformer)
    ├── manuel_maintenance_compresseur_industriel.pdf  # Document 2 (Manuel technique industriel)
    └── extracted_assets/                              # Images PNG et tableaux Markdown extraits
```

---

## 🚀 Démarrage Rapide

### 1. Installation
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Démarrage de l'interface Streamlit
```powershell
cd "Partie 1"
python -m streamlit run app.py
```
Accès navigateur : **`http://localhost:8501`**

### 3. Exécution de l'Agent LangChain en ligne de commande
```powershell
python agentic_rag.py --query "Quels sont les couples de serrage pour la culasse et le flasque ?"
```
