"""
Module de Pipeline RAG (Retrieval-Augmented Generation) pour le projet MAINTiDOC.
Intègre :
  - La recherche sémantique multimodale dans PostgreSQL / PGVector (LlamaIndex + BGE)
  - Le modèle LLM Open Source Qwen (exécuté localement via Ollama)
  - Un prompt de génération strict garantissant des réponses fondées sur les sources retrouvées.
"""

import sys
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama

from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DB,
    PG_TABLE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
)

# Template de prompt RAG pour guider Qwen
RAG_SYSTEM_PROMPT = """Tu es un assistant expert technique pour l'analyse de documents industriels et scientifiques.
Ton rôle est de répondre aux questions de manière précise, factuelle et directement fondée sur le contexte fourni.

Règles strictes :
1. Base TOUTE ton argumentation UNIQUEMENT sur les informations présentes dans le CONTEXTE fourni ci-dessous.
2. Si le contexte ne contient pas l'information demandée, indique clairement : "Cette information n'est pas mentionnée dans le document analysé." Ne spécule pas et n'invente rien.
3. Cite TOUJOURS précisément tes sources : mentionne le numéro de page, le nom du document, et le type d'élément (tableau, extrait de texte ou schéma/image) d'où provient chaque élément de réponse.
4. Réponds dans la langue de la question posée (en français ou en anglais).
5. Sois structuré, clair et synthétique (puces, chiffres exacts, valeurs techniques).
"""

RAG_QA_TEMPLATE = PromptTemplate(
    template="""{system_prompt}

---------------------
CONTEXTE RETROUVÉ DANS LE DOCUMENT :
{context_str}
---------------------

QUESTION : {query_str}

RÉPONSE FONDÉE SUR LE CONTEXTE (avec citations de pages) :
""",
    template_var_mappings={"system_prompt": "system_prompt"}
)


class TechnicalRAGPipeline:
    """
    Pipeline RAG reliant PGVector (PostgreSQL) et Qwen (via Ollama).
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT
    ):
        print(f"[*] Initialisation du pipeline RAG avec le LLM open source : {model_name} (Ollama)...")
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            request_timeout=timeout,
            temperature=0.1  # Faible température pour un maximum de factualité
        )
        
        print(f"[*] Chargement du modèle d'embedding : {EMBEDDING_MODEL_NAME}...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            trust_remote_code=True
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        print(f"[*] Connexion à PGVector (PostgreSQL: {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB})...")
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

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        node_type: Optional[str] = None
    ) -> List[Any]:
        """Récupère les passages les plus pertinents depuis PGVector."""
        filters = None
        if node_type:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="type", value=node_type)]
            )

        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            filters=filters
        )
        return retriever.retrieve(query)

    def ask(
        self,
        query: str,
        top_k: int = 3,
        node_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exécute le cycle RAG complet :
          1. Retrieval des nœuds pertinents dans PGVector
          2. Construction du contexte structuré
          3. Génération de la réponse sourcée par Qwen via Ollama
        """
        # 1. Retrieval
        retrieved_nodes = self.retrieve_context(query=query, top_k=top_k, node_type=node_type)

        if not retrieved_nodes:
            return {
                "query": query,
                "answer": "Aucun passage pertinent n'a été trouvé dans la base vectorielle.",
                "sources": []
            }

        # 2. Construction du contexte avec métadonnées précises
        context_parts = []
        sources = []
        for idx, item in enumerate(retrieved_nodes, start=1):
            node = item.node
            meta = node.metadata
            score = item.score or 0.0
            doc_type = meta.get("type", "inconnu").upper()
            page = meta.get("page_number", "?")
            source_file = meta.get("source_file", "document.pdf")
            caption = meta.get("caption", "")

            source_header = f"[Source #{idx} | Page {page} | Type: {doc_type} | Fichier: {source_file}]"
            if caption:
                source_header += f" (Légende: {caption})"

            context_parts.append(f"{source_header}\n{node.text}")

            sources.append({
                "rank": idx,
                "page": page,
                "type": doc_type,
                "file": source_file,
                "score": score,
                "caption": caption,
                "text": node.text,
                "metadata": meta,
                "text_snippet": node.text[:220].replace("\n", " ") + "..."
            })

        formatted_context = "\n\n".join(context_parts)

        # 3. Génération avec Qwen
        prompt = RAG_QA_TEMPLATE.format(
            system_prompt=RAG_SYSTEM_PROMPT,
            context_str=formatted_context,
            query_str=query
        )

        response = self.llm.complete(prompt)

        return {
            "query": query,
            "answer": response.text.strip(),
            "sources": sources
        }

    def display_answer(self, result: Dict[str, Any]):
        """Affiche élégamment la question, la réponse de Qwen et les sources citées."""
        print("\n" + "=" * 80)
        print(f"❓ QUESTION : {result['query']}")
        print("=" * 80)

        print("\n🤖 RÉPONSE GÉNÉRÉE PAR QWEN (OLLAMA) :")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)

        print("\n📚 SOURCES RETROUVÉES DANS PGVECTOR :")
        for s in result["sources"]:
            print(f"  [#{s['rank']}] Page {s['page']} | Type: {s['type']} | Similarité: {s['score']:.4f}")
            if s["caption"]:
                print(f"       Légende: {s['caption']}")
            print(f"       Extrait: {s['text_snippet']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline RAG avec Qwen et PGVector")
    parser.add_argument("--query", "-q", type=str, help="Question à poser au document")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Nombre de passages à récupérer")
    parser.add_argument("--type", "-t", choices=["text", "table", "image"], default=None, help="Filtrer par type")
    args = parser.parse_args()

    pipeline = TechnicalRAGPipeline()

    if args.query:
        ans = pipeline.ask(query=args.query, top_k=args.top_k, node_type=args.type)
        pipeline.display_answer(ans)
    else:
        # Mode démo automatique
        demo_questions = [
            "Quelle est la formule mathématique de l'attention Scaled Dot-Product et comment fonctionne-t-elle ?",
            "Quels sont les scores BLEU obtenus par le modèle Transformer en comparaison avec les autres modèles ?",
            "Quelle est la périodicité de vidange de l'huile du compresseur et quelle référence d'huile utiliser ?",
            "Quels sont les couples de serrage recommandés pour la culasse et le flasque d'aspiration ?"
        ]
        print("\n--- Mode Démonstration RAG avec Qwen ---")
        for q in demo_questions[:2]:
            ans = pipeline.ask(query=q, top_k=2)
            pipeline.display_answer(ans)

