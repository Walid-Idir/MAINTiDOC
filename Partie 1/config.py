"""
Configuration pour le projet MAINTiDOC
Connexion PostgreSQL / PGVector et paramètres d'embedding.
"""

import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
ASSETS_DIR = BASE_DIR / "extracted_assets"
IMAGES_DIR = ASSETS_DIR / "images"
TABLES_DIR = ASSETS_DIR / "tables"
TEXT_DIR = ASSETS_DIR / "text"

# Création automatique des répertoires d'extraction
for d in [ASSETS_DIR, IMAGES_DIR, TABLES_DIR, TEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configuration PostgreSQL & PGVector
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "docuser")
PG_PASSWORD = os.getenv("PG_PASSWORD", "docpass")
PG_DB = os.getenv("PG_DB", "docdb")
PG_TABLE = os.getenv("PG_TABLE", "document_embeddings")

# Configuration du modèle d'embedding open source (Texte & Tableaux)
# BAAI/bge-small-en-v1.5 est un modèle open source performant produisant des vecteurs de dimension 384
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Configuration du modèle multimodal CLIP (Images)
# openai/clip-vit-base-patch32 projette les images et le texte dans un espace commun de dimension 512
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
CLIP_TABLE = "image_clip_embeddings"

# Configuration du modèle LLM Open Source via Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

# Document par défaut recommandé
DEFAULT_PDF = BASE_DIR / "Attention is all you need.pdf"
