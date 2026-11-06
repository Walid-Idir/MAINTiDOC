"""
Module de traitement des images avec CLIP et génération de descriptions textuelles
pour le projet MAINTiDOC.

Fonctionnalités :
  1. Représentation des images par vecteurs CLIP (512 dimensions)
  2. Génération automatique de descriptions textuelles enrichies (tags visuels CLIP + contexte du PDF)
  3. Stockage et indexation dans PostgreSQL avec PGVector (table dédiée image_clip_embeddings)
  4. Recherche multi-modale 'Text-to-Image' et 'Image-to-Image' par similarité cosinus directe dans PGVector
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import psycopg2
from psycopg2.extras import RealDictCursor
from llama_index.core.schema import TextNode

from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DB,
    CLIP_MODEL_NAME,
    CLIP_DIM,
    CLIP_TABLE,
)


class CLIPImageProcessor:
    """
    Gestionnaire du traitement visuel par CLIP et de l'indexation PGVector.
    """

    # Concepts candidats pour la classification zero-shot des schémas techniques
    ZERO_SHOT_CANDIDATES = [
        "transformer model architecture diagram with encoder and decoder blocks",
        "scaled dot-product attention and multi-head attention mechanism diagram",
        "neural network layer flow and positional encoding scheme",
        "industrial screw compressor mechanical flow and piping schematic",
        "mechanical assembly drawing with electric motor and rotors",
        "engineering block diagram with boxes and connecting arrows",
        "technical chart graph or performance data curve",
        "machine maintenance troubleshooting flowchart"
    ]

    def __init__(self):
        print(f"[*] Chargement du modèle multimodal CLIP : {CLIP_MODEL_NAME}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()

        # Initialisation de la table PGVector pour CLIP
        self._init_pgvector_table()

    def _init_pgvector_table(self):
        """Crée la table PostgreSQL avec la colonne vector(512) et l'index HNSW si nécessaire."""
        try:
            conn = psycopg2.connect(
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASSWORD,
                host=PG_HOST,
                port=PG_PORT
            )
            cur = conn.cursor()
            
            # Activer l'extension vector si ce n'est pas déjà fait
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Table dédiée pour les embeddings CLIP des images
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {CLIP_TABLE} (
                id SERIAL PRIMARY KEY,
                image_filename VARCHAR(255) UNIQUE,
                image_path TEXT NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                page_number INT NOT NULL,
                dimensions VARCHAR(50),
                caption TEXT,
                visual_tags TEXT,
                generated_description TEXT NOT NULL,
                clip_embedding vector({CLIP_DIM}) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_{CLIP_TABLE}_hnsw 
            ON {CLIP_TABLE} USING hnsw (clip_embedding vector_cosine_ops);
            """
            cur.execute(create_table_sql)
            conn.commit()
            cur.close()
            conn.close()
            print(f"[✓] Table PGVector '{CLIP_TABLE}' vérifiée et prête (dimension {CLIP_DIM}).")
        except Exception as e:
            print(f"[!] Erreur initialisation table CLIP PGVector : {e}")

    def compute_image_embedding(self, image_path: str | Path) -> List[float]:
        """Calcule l'embedding CLIP 512D pour une image depuis ses pixels."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Gestion compatibilité versions transformers (tensor ou output object)
            features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            # Normalisation L2 pour similarité cosinus directe
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.squeeze().cpu().tolist()

    def compute_text_query_embedding(self, query_text: str) -> List[float]:
        """Calcule l'embedding CLIP 512D pour une requête textuelle."""
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.squeeze().cpu().tolist()

    def classify_visual_tags(self, image_path: str | Path, top_k: int = 2) -> List[str]:
        """Utilise le zero-shot de CLIP pour identifier les meilleurs concepts visuels de l'image."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=self.ZERO_SHOT_CANDIDATES,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity
            probs = logits_per_image.softmax(dim=1).squeeze()

        top_indices = torch.topk(probs, k=min(top_k, len(self.ZERO_SHOT_CANDIDATES))).indices.tolist()
        return [self.ZERO_SHOT_CANDIDATES[i] for i in top_indices]

    def generate_image_description(
        self,
        image_path: str | Path,
        caption: str,
        page_number: int,
        source_file: str,
        dimensions: str = "",
        page_text: str = ""
    ) -> Dict[str, Any]:
        """
        Génère une description textuelle complète et structurée combinant :
        - Les étiquettes visuelles zero-shot inférées par CLIP
        - La légende extraite du PDF
        - Le contexte textuel technique de la page correspondante
        - Les métadonnées physiques (dimensions, format)
        """
        # 1. Tags visuels CLIP
        try:
            visual_tags = self.classify_visual_tags(image_path, top_k=2)
            tags_str = ", ".join(visual_tags)
        except Exception:
            tags_str = "technical diagram / illustration"

        # 2. Contexte extrait de la page (recherche de phrases clés)
        context_snippet = ""
        if page_text:
            # Chercher des phrases faisant référence à "Figure" ou "Architecture" ou "Schéma"
            sentences = re.split(r'(?<=[.!?])\s+', page_text)
            relevant = [s.strip() for s in sentences if any(w in s.lower() for w in ["figure", "architecture", "model", "schéma", "moteur", "attention"])]
            if relevant:
                context_snippet = " ".join(relevant[:3])[:350]

        # 3. Assemblage de la description textuelle finale
        description = (
            f"Description Visuelle & Technique : Cette image correspond à un schéma technique "
            f"identifié comme : [{tags_str}]. "
            f"Titre et Légende dans le document : {caption}. "
        )
        if context_snippet:
            description += f"Explication technique contextuelle : {context_snippet}. "
        if dimensions:
            description += f"Caractéristiques de l'image : résolution {dimensions} px, issue de la page {page_number} du fichier '{source_file}'."

        return {
            "visual_tags": tags_str,
            "generated_description": description
        }

    def process_and_index_images(
        self,
        extracted_images: List[Dict[str, Any]],
        doc_pages_text: Optional[Dict[int, str]] = None
    ) -> List[TextNode]:
        """
        Traite un ensemble d'images extraites :
        1. Génère les descriptions textuelles
        2. Calcule les vecteurs CLIP (512D)
        3. Stocke les images et vecteurs dans la table PGVector image_clip_embeddings
        4. Produit des TextNodes LlamaIndex pour indexation dans le catalogue principal
        """
        doc_pages_text = doc_pages_text or {}
        llama_nodes: List[TextNode] = []

        print(f"\n[*] Traitement de {len(extracted_images)} images avec CLIP & Génération de descriptions...")

        conn = psycopg2.connect(
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT
        )
        cur = conn.cursor()

        for idx, img_info in enumerate(extracted_images, start=1):
            img_path = Path(img_info["path"])
            filename = img_path.name
            page_num = img_info.get("page_number", 1)
            source_file = img_info.get("source_file", "document.pdf")
            caption = img_info.get("caption", f"Figure de la page {page_num}")
            dimensions = img_info.get("dimensions", "")
            page_text = doc_pages_text.get(page_num, "")

            print(f"  [{idx}/{len(extracted_images)}] Analyse image : {filename} (Page {page_num})...")

            # A. Calcul de l'embedding CLIP 512D
            clip_vector = self.compute_image_embedding(img_path)

            # B. Génération de la description textuelle enrichie
            desc_data = self.generate_image_description(
                image_path=img_path,
                caption=caption,
                page_number=page_num,
                source_file=source_file,
                dimensions=dimensions,
                page_text=page_text
            )
            gen_desc = desc_data["generated_description"]
            tags = desc_data["visual_tags"]

            # C. Insertion / Mise à jour dans la table PGVector image_clip_embeddings
            insert_sql = f"""
            INSERT INTO {CLIP_TABLE} (
                image_filename, image_path, source_file, page_number,
                dimensions, caption, visual_tags, generated_description, clip_embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_filename) DO UPDATE SET
                caption = EXCLUDED.caption,
                visual_tags = EXCLUDED.visual_tags,
                generated_description = EXCLUDED.generated_description,
                clip_embedding = EXCLUDED.clip_embedding;
            """
            cur.execute(
                insert_sql,
                (filename, str(img_path), source_file, page_num, dimensions, caption, tags, gen_desc, clip_vector)
            )

            # D. Création d'un TextNode LlamaIndex avec la description générée
            node_text = (
                f"Type: Image / Schéma Technique (Traité par CLIP)\n"
                f"Document: {source_file}\n"
                f"Page: {page_num}\n"
                f"Légende: {caption}\n"
                f"Tags Visuels: {tags}\n"
                f"Description Générée:\n{gen_desc}\n"
                f"Fichier: {filename}"
            )
            node = TextNode(
                text=node_text,
                metadata={
                    "source_file": source_file,
                    "type": "image",
                    "page_number": page_num,
                    "image_id": idx,
                    "image_filename": filename,
                    "caption": caption,
                    "visual_tags": tags,
                    "dimensions": dimensions,
                    "asset_path": str(img_path),
                    "generated_description": gen_desc
                }
            )
            llama_nodes.append(node)

        conn.commit()
        cur.close()
        conn.close()

        print(f"[✓] {len(extracted_images)} images enregistrées dans PGVector ('{CLIP_TABLE}') et préparées pour LlamaIndex !")
        return llama_nodes

    def search_images_clip(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Effectue une recherche 'Text-to-Image' zero-shot avec CLIP dans PostgreSQL PGVector.
        Utilise l'opérateur de distance cosinus (<=>) de PGVector.
        """
        # 1. Calcul du vecteur CLIP de la requête
        query_vector = self.compute_text_query_embedding(query)

        # 2. Requête SQL PGVector par similarité cosinus
        conn = psycopg2.connect(
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        search_sql = f"""
        SELECT 
            id,
            image_filename,
            image_path,
            source_file,
            page_number,
            dimensions,
            caption,
            visual_tags,
            generated_description,
            1 - (clip_embedding <=> %s::vector) AS similarity_score
        FROM {CLIP_TABLE}
        ORDER BY clip_embedding <=> %s::vector
        LIMIT %s;
        """
        cur.execute(search_sql, (query_vector, query_vector, top_k))
        results = cur.fetchall()
        cur.close()
        conn.close()

        return [dict(r) for r in results]
