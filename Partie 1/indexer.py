"""
Module d'indexation vectorielle dans PostgreSQL avec l'extension PGVector.
Calcule les embeddings via un modèle open source (HuggingFaceEmbedding)
et stocke les vecteurs ainsi que leurs métadonnées dans la base de données.
"""

import sys
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DB,
    PG_TABLE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)


class PGVectorIndexer:
    """
    Gère l'indexation de nœuds LlamaIndex dans PostgreSQL PGVector.
    """

    def __init__(self):
        print(f"[*] Initialisation du modèle d'embedding open source : {EMBEDDING_MODEL_NAME}...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            trust_remote_code=True
        )
        Settings.embed_model = self.embed_model
        Settings.llm = None  # Pas de dépendance API externe requise pour l'indexation

        print(f"[*] Connexion au Vector Store PGVector ({PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}, table={PG_TABLE})...")
        self.vector_store = PGVectorStore.from_params(
            database=PG_DB,
            host=PG_HOST,
            password=PG_PASSWORD,
            port=PG_PORT,
            user=PG_USER,
            table_name=PG_TABLE,
            embed_dim=EMBEDDING_DIM,
            perform_setup=True,  # Crée la table et l'extension vector si nécessaire
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def index_nodes(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """
        Calcule les embeddings en batch et insère les nœuds dans PostgreSQL / PGVector.
        """
        print(f"[*] Calcul accéléré des embeddings en batch pour {len(nodes)} éléments (texte, tableaux, images)...")
        texts_to_embed = [node.get_content() for node in nodes]
        embeddings = self.embed_model.get_text_embedding_batch(texts_to_embed, show_progress=True)
        
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb

        print("[*] Insertion des vecteurs et métadonnées dans la base PostgreSQL PGVector...")
        self.vector_store.add(nodes)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        print("[✓] Indexation vectorielle terminée avec succès dans PGVector !")
        return index

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Interroge directement PostgreSQL pour vérifier le contenu de la table PGVector.
        """
        stats = {
            "total_rows": 0,
            "by_type": {},
            "table_name": f"data_{PG_TABLE}"
        }
        try:
            conn = psycopg2.connect(
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASSWORD,
                host=PG_HOST,
                port=PG_PORT
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Nom de la table créée par PGVectorStore de LlamaIndex (préfixée par 'data_')
            real_table = f"data_{PG_TABLE}"
            
            cur.execute(f"SELECT COUNT(*) AS total FROM {real_table};")
            stats["total_rows"] = cur.fetchone()["total"]

            # Répartition par type de nœud (text, table, image)
            cur.execute(f"""
                SELECT metadata_->>'type' AS doc_type, COUNT(*) AS count
                FROM {real_table}
                GROUP BY metadata_->>'type'
                ORDER BY count DESC;
            """)
            rows = cur.fetchall()
            for r in rows:
                stats["by_type"][r["doc_type"]] = r["count"]

            cur.close()
            conn.close()
        except Exception as e:
            stats["error"] = str(e)

        return stats
