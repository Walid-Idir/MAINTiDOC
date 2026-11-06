"""
Module d'Orchestration Agentique avec LangChain - Projet MAINTiDOC

Ce module implémente un RAG Agentique complet :
  1. Sélection autonome d'outils de recherche (Texte/Tableaux BGE, Schémas visuels CLIP, Métadonnées documents)
  2. Interrogation dynamique de l'index PostgreSQL / PGVector
  3. Gestion du contexte et du raisonnement multi-étapes (Scratchpad / ReAct)
  4. Synthèse et génération d'une réponse factuelle strictement sourcée avec Qwen (Ollama)
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Encodage UTF-8 pour Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Résolution automatique du chemin local
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import psycopg2
from psycopg2.extras import RealDictCursor

# Composants LangChain & LangChain Ollama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DB,
    PG_TABLE,
    CLIP_TABLE,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
)
from image_processor import CLIPImageProcessor
from search_engine import MultiModalSearchEngine


AGENT_SYSTEM_PROMPT = """You are an expert AI technical assistant in machine maintenance and technical document analysis.
You have access to a PostgreSQL PGVector database containing technical text, maintenance tables, and engineering diagrams.

AVAILABLE TOOLS:
1. `search_text_and_tables(query: str, filter_type: str = None)`: Semantic search in text paragraphs and maintenance tables (torque values, oil change intervals, tolerances, part references, formulas, BLEU scores).
2. `search_visual_clip(visual_query: str)`: Multimodal visual search (CLIP 512D) for engineering diagrams, compressor flowcharts, architecture figures, or technical schemes.
3. `inspect_documents_in_db()`: Lists available documents and vector statistics in PGVector.

EXECUTION RULES:
- When the user asks a question, ALWAYS invoke the appropriate search tool first (`search_text_and_tables` or `search_visual_clip`).
- Do NOT make up any values or guess without consulting tools.
- Once tool observations are received, synthesize a comprehensive, factual answer in French (or the user's language) and systematically cite the source document, page number, and table/figure.
"""


class AgenticRAGOrchestrator:
    """
    Orchestrateur Agentique LangChain gérant le cycle de décision :
    Question -> Sélection d'outils -> Interrogation PGVector -> Gestion Contexte -> Réponse Sourcée
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
        max_iterations: int = 4
    ):
        print(f"[*] Initialisation de l'Agent LangChain avec LLM : {model_name}...")
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            timeout=OLLAMA_TIMEOUT
        )
        self.max_iterations = max_iterations

        # Initialisation des moteurs sous-jacents
        self.search_engine = MultiModalSearchEngine()
        self.clip_processor = CLIPImageProcessor()

        # Enregistrement des outils LangChain
        self.tools = self._build_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _build_tools(self):
        """Définit les outils LangChain disponibles pour l'Agent."""

        search_engine = self.search_engine
        clip_processor = self.clip_processor

        @tool
        def search_text_and_tables(query: str, filter_type: Optional[str] = None) -> str:
            """Semantic search in text and technical maintenance tables in PostgreSQL PGVector.
            Use this tool to find torque specifications, oil change intervals, tolerances,
            mathematical formulas, model architectures, or numerical data.
            Args:
              query: search query keywords or technical question (e.g. 'torque cylinder head', 'oil change periodicity').
              filter_type: optional filter: 'text', 'table', or None for both.
            """
            filtre = None
            if filter_type in ["text", "table"]:
                filtre = filter_type

            results = search_engine.search(query=query, top_k=3, node_type=filtre)
            if not results:
                return f"No results found for query: '{query}'."

            formatted = []
            for item in results:
                node = item.node
                meta = node.metadata
                page = meta.get("page_number", "?")
                doc_type = meta.get("type", "inconnu").upper()
                source_file = meta.get("source_file", "document.pdf")
                caption = meta.get("caption", "")
                score = item.score or 0.0
                content = node.text

                entry = f"--- Source [Page {page} | Type: {doc_type} | Doc: {source_file}] ---\n"
                if caption:
                    entry += f"Légende: {caption}\n"
                entry += f"Score similarité: {score:.4f}\n"
                entry += f"Contenu:\n{content}\n"
                formatted.append(entry)

            return "\n".join(formatted)

        @tool
        def search_visual_clip(visual_query: str) -> str:
            """Multimodal visual search (Text-to-Image) using CLIP 512D embeddings in PostgreSQL PGVector.
            Use this tool when the question mentions a diagram, scheme, figure, flowchart, or visual machine part.
            Args:
              visual_query: description in French or English of the visual diagram (e.g. 'screw compressor piping diagram', 'transformer encoder decoder architecture').
            """
            results = clip_processor.search_images_clip(query=visual_query, top_k=2)
            if not results:
                return f"No images found for visual query: '{visual_query}'."

            formatted = []
            for r in results:
                entry = (
                    f"--- Image Retrouvée [Page {r['page_number']} | Doc: {r['source_file']}] ---\n"
                    f"Fichier image: {r['image_filename']}\n"
                    f"Chemin asset: {r['image_path']}\n"
                    f"Légende: {r.get('caption', 'Sans légende')}\n"
                    f"Tags visuels détectés par CLIP: {r.get('visual_tags', 'N/A')}\n"
                    f"Similarité cosinus CLIP: {r['similarity_score']:.4f}\n"
                    f"Description contextuelle: {r.get('generated_description', '')}\n"
                )
                formatted.append(entry)

            return "\n".join(formatted)

        @tool
        def inspect_documents_in_db() -> str:
            """Inspects PostgreSQL PGVector database to list active documents, page counts, tables, and images.
            Useful to see what documents are loaded or answer high-level overview questions.
            """
            try:
                conn = psycopg2.connect(
                    dbname=PG_DB,
                    user=PG_USER,
                    password=PG_PASSWORD,
                    host=PG_HOST,
                    port=PG_PORT,
                    connect_timeout=3
                )
                cur = conn.cursor(cursor_factory=RealDictCursor)
                real_table = f"data_{PG_TABLE}"
                
                cur.execute(f"""
                    SELECT metadata_->>'source_file' AS filename,
                           COUNT(*) AS total_chunks,
                           COUNT(*) FILTER (WHERE metadata_->>'type' = 'text') AS text_chunks,
                           COUNT(*) FILTER (WHERE metadata_->>'type' = 'table') AS table_chunks,
                           COUNT(*) FILTER (WHERE metadata_->>'type' = 'image') AS image_chunks
                    FROM {real_table}
                    WHERE metadata_->>'source_file' IS NOT NULL
                    GROUP BY metadata_->>'source_file'
                    ORDER BY total_chunks DESC;
                """)
                rows = cur.fetchall()

                cur.execute(f"SELECT COUNT(*) AS total_clip FROM {CLIP_TABLE};")
                clip_row = cur.fetchone()
                clip_total = clip_row["total_clip"] if clip_row else 0

                cur.close()
                conn.close()

                if not rows:
                    return "Aucun document indexé dans PostgreSQL actuellement."

                output = ["=== Documents Disponibles dans PGVector ==="]
                for r in rows:
                    output.append(
                        f"• {r['filename']} : {r['total_chunks']} chunks au total "
                        f"({r['text_chunks']} texte, {r['table_chunks']} tableaux, {r['image_chunks']} images)"
                    )
                output.append(f"• Total des schémas visuels CLIP vectorisés : {clip_total}")
                return "\n".join(output)
            except Exception as e:
                return f"Erreur lors de l'inspection de la base : {e}"

        return [search_text_and_tables, search_visual_clip, inspect_documents_in_db]

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Exécute la boucle agentique complète :
          1. Initialise le fil de messages avec la question
          2. Boucle de raisonnement (Thoughts / Tool Calls / Observations)
          3. Arrêt dès que le LLM génère la réponse finale ou atteint max_iterations
        """
        print("\n" + "=" * 80)
        print(f"🤖 AGENT RAG LANGCHAIN - TRAITEMENT DE LA QUESTION : '{user_query}'")
        print("=" * 80)

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=f"User Question: {user_query}\nPlease use the tools to find factual evidence and answer in French with source citations.")
        ]

        steps_trace = []
        tools_executed = []

        iteration = 0
        final_answer = ""

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n[Étape {iteration}] Réflexion de l'Agent...")

            response: AIMessage = self.llm_with_tools.invoke(messages)

            # Si l'Agent a décidé d'appeler des outils
            if response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    tool_id = tc.get("id", f"call_{iteration}_{tool_name}")

                    print(f"  🔧 Action : Appel de l'outil `{tool_name}`")
                    print(f"     Paramètres : {tool_args}")

                    tool_fn = self.tool_map.get(tool_name)
                    if tool_fn:
                        observation = tool_fn.invoke(tool_args)
                    else:
                        observation = f"Outil '{tool_name}' inconnu."

                    snippet = observation[:220].replace("\n", " ") + "..." if len(observation) > 220 else observation
                    print(f"  👁️ Observation : {snippet}")

                    tools_executed.append({
                        "step": iteration,
                        "tool": tool_name,
                        "args": tool_args,
                        "observation_preview": snippet
                    })

                    steps_trace.append({
                        "step": iteration,
                        "thought": response.content or f"L'agent consulte la base documentaire via {tool_name}.",
                        "action": tool_name,
                        "args": tool_args,
                        "observation": observation
                    })

                    # Injecter l'observation de l'outil dans le contexte
                    messages.append(
                        ToolMessage(
                            content=str(observation),
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
            else:
                # Si turn 1 et aucun outil appelé, forcer l'appel du moteur sémantique pour garantir la factualité
                if iteration == 1 and not tools_executed:
                    print("  [*] Déclenchement automatique de la recherche sémantique pour garantir la factualité...")
                    fallback_tool = "search_text_and_tables"
                    tool_args = {"query": user_query}
                    observation = self.tool_map[fallback_tool].invoke(tool_args)

                    snippet = observation[:220].replace("\n", " ") + "..." if len(observation) > 220 else observation
                    tools_executed.append({
                        "step": iteration,
                        "tool": fallback_tool,
                        "args": tool_args,
                        "observation_preview": snippet
                    })
                    steps_trace.append({
                        "step": iteration,
                        "thought": "Recherche de contexte documentaire pertinent.",
                        "action": fallback_tool,
                        "args": tool_args,
                        "observation": observation
                    })
                    messages.append(
                        AIMessage(
                            content="",
                            tool_calls=[{"name": fallback_tool, "args": tool_args, "id": "fallback_call", "type": "tool_call"}]
                        )
                    )
                    messages.append(
                        ToolMessage(
                            content=str(observation),
                            tool_call_id="fallback_call",
                            name=fallback_tool
                        )
                    )
                    continue

                # L'agent a formulé sa réponse finale
                final_answer = response.content.strip()
                print(f"  ✅ Réponse finale formulée par l'Agent.")
                break

        if not final_answer:
            # Demande explicite de synthèse finale si messages contient des observations
            print("  [*] Synthèse finale du contexte...")
            summary_prompt = HumanMessage(content="Sur la base des observations obtenues des outils ci-dessus, formule ta réponse finale en français avec citations exactes.")
            messages.append(summary_prompt)
            final_resp = self.llm.invoke(messages)
            final_answer = final_resp.content.strip()

        return {
            "query": user_query,
            "answer": final_answer,
            "iterations": iteration,
            "tools_executed": tools_executed,
            "trace": steps_trace
        }


# -----------------------------------------------------------------------------
# Exécution en Ligne de Commande pour Validation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent RAG LangChain - MAINTiDOC")
    parser.add_argument(
        "--query",
        type=str,
        default="Quels sont les couples de serrage pour la culasse et le flasque ?",
        help="Question à poser à l'Agent RAG"
    )
    args = parser.parse_args()

    orchestrator = AgenticRAGOrchestrator()
    result = orchestrator.run(args.query)

    print("\n" + "=" * 80)
    print("📊 RAPPORT D'EXÉCUTION DE L'AGENT LANGCHAIN")
    print("=" * 80)
    print(f"❓ Question : {result['query']}")
    print(f"🔄 Itérations : {result['iterations']}")
    print(f"🛠️ Outils appelés ({len(result['tools_executed'])}) :")
    for t in result["tools_executed"]:
        print(f"   - [Étape {t['step']}] `{t['tool']}` avec {t['args']}")

    print("\n🤖 RÉPONSE FINALE DE L'AGENT :")
    print("-" * 80)
    print(result["answer"])
    print("-" * 80)
