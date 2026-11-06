"""
Module de recherche sémantique multi-modale dans PGVector.
Permet d'interroger la base vectorielle PostgreSQL et de retrouver
le texte, les tableaux et les images les plus pertinents.
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
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


class MultiModalSearchEngine:
    """
    Moteur de recherche sémantique s'appuyant sur PGVector et LlamaIndex.
    """

    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            trust_remote_code=True
        )
        self.vector_store = PGVectorStore.from_params(
            database=PG_DB,
            host=PG_HOST,
            password=PG_PASSWORD,
            port=PG_PORT,
            user=PG_USER,
            table_name=PG_TABLE,
            embed_dim=EMBEDDING_DIM,
            perform_setup=False,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        node_type: Optional[str] = None
    ):
        """
        Recherche sémantique vectorielle avec filtre optionnel par type (text, table, image).
        """
        filters = None
        if node_type:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="type", value=node_type)]
            )

        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            filters=filters
        )
        nodes_with_scores = retriever.retrieve(query)
        return nodes_with_scores

    def display_results(self, query: str, results, header: str = "Résultats de recherche"):
        """
        Affiche les résultats de manière élégante et structurée.
        """
        print("\n" + "=" * 70)
        print(f"🔍 REQUÊTE : '{query}'")
        print(f"📋 {header} ({len(results)} éléments trouvés)")
        print("=" * 70)

        if not results:
            print("  Aucun résultat trouvé.")
            return

        for idx, item in enumerate(results, start=1):
            node = item.node
            score = item.score
            meta = node.metadata
            doc_type = meta.get("type", "inconnu").upper()
            page = meta.get("page_number", "?")

            print(f"\n[#{idx}] TYPE : {doc_type} | PAGE : {page} | SCORE SIMILARITÉ : {score:.4f}")
            print("-" * 50)
            
            if doc_type == "IMAGE":
                print(f"  🖼️  Légende : {meta.get('caption', 'N/A')}")
                print(f"  📁 Fichier Image : {meta.get('asset_path', 'N/A')}")
                print(f"  📐 Dimensions : {meta.get('dimensions', 'N/A')} (Format {meta.get('format', 'N/A')})")
            elif doc_type == "TABLE":
                print(f"  📊 Légende : {meta.get('caption', 'N/A')}")
                print(f"  📁 Fichier Markdown : {meta.get('asset_path', 'N/A')}")
                print(f"  🔢 Dimensions : {meta.get('rows', '?')} lignes x {meta.get('cols', '?')} colonnes")
                print("  Contenu extrait :")
                # Afficher les 6 premières lignes du tableau
                lines = node.text.split("\n")
                preview = "\n    ".join(lines[:12])
                print(f"    {preview}")
            else:
                # Texte classique
                snippet = node.text[:300].replace("\n", " ")
                print(f"  📄 Extrait : {snippet}...")
