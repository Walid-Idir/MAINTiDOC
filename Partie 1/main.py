"""
Script Principal - Projet MAINTiDOC
Pipeline complet :
  1. Sélection du document PDF
  2. Extraction multi-modale (Texte, Tableaux, Images) avec PyMuPDF & LlamaIndex
  3. Traitement des images avec CLIP :
     - Calcul des représentations vectorielles visuelles (CLIP 512D)
     - Génération automatique de descriptions textuelles enrichies (tags visuels + contexte PDF)
     - Stockage et indexation dans PostgreSQL / PGVector (table image_clip_embeddings)
  4. Calcul des embeddings open source (BAAI/bge-small-en-v1.5) & Indexation dans PGVector
  5. Démonstration de recherche sémantique multi-modale & recherche visuelle CLIP (Text-to-Image)
"""

import sys
import argparse
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

# Assurer l'encodage UTF-8 pour Windows
sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_PDF, PG_TABLE, PG_DB, PG_USER, PG_HOST, PG_PORT, CLIP_TABLE
from extractor import PDFMultiModalExtractor
from image_processor import CLIPImageProcessor
from indexer import PGVectorIndexer
from search_engine import MultiModalSearchEngine
from rag_pipeline import TechnicalRAGPipeline


def get_clip_database_stats() -> int:
    """Compte le nombre d'images indexées dans la table PGVector de CLIP."""
    try:
        conn = psycopg2.connect(
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT COUNT(*) AS total FROM {CLIP_TABLE};")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row["total"] if row else 0
    except Exception:
        return 0


def run_pipeline(pdf_path: Path):
    print("=" * 80)
    print("🚀 PROJET MAINTiDOC - PIPELINE MULTIMODAL AVEC CLIP & PGVECTOR")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Étape 1 : Sélection du Document
    # -------------------------------------------------------------------------
    print(f"\n[ÉTAPE 1] Sélection du document : {pdf_path.name}")
    print(f"  Chemin absolu : {pdf_path.resolve()}")
    if not pdf_path.exists():
        print(f"[❌] Erreur : Le fichier {pdf_path} n'existe pas.")
        sys.exit(1)
    file_size_kb = pdf_path.stat().st_size / 1024
    print(f"  Taille du fichier : {file_size_kb:.2f} Ko")

    # -------------------------------------------------------------------------
    # Étape 2 : Extraction Multi-Modale (Texte, Tableaux, Images)
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 2] Extraction multi-modale (PyMuPDF & LlamaIndex)...")
    extractor = PDFMultiModalExtractor(pdf_path=pdf_path)
    nodes, report = extractor.extract()

    print(f"  Pages analysées : {report['total_pages']}")
    print(f"  ✓ Fragments de texte extraits : {report['text_chunks']}")
    print(f"  ✓ Tableaux structurés extraits : {report['tables_extracted']}")
    print(f"  ✓ Images / figures extraites   : {report['images_extracted']}")

    # -------------------------------------------------------------------------
    # Étape 3 : Traitement des Images avec CLIP & Génération de Descriptions
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 3] Traitement des images avec CLIP & Génération de descriptions...")
    clip_processor = CLIPImageProcessor()
    extracted_images = report.get("extracted_images", [])
    page_texts = report.get("page_texts", {})

    enriched_image_nodes = []
    if extracted_images:
        enriched_image_nodes = clip_processor.process_and_index_images(
            extracted_images=extracted_images,
            doc_pages_text=page_texts
        )
    else:
        print("  Aucune image raster détectée à traiter par CLIP dans ce document.")

    # Remplacer les nœuds images basiques par les nœuds enrichis avec descriptions générées
    final_nodes = [n for n in nodes if n.metadata.get("type") != "image"] + enriched_image_nodes
    print(f"  -> Total de nœuds préparés pour l'indexation globale : {len(final_nodes)}")

    # -------------------------------------------------------------------------
    # Étape 4 : Embeddings Open Source & Indexation PGVector
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 4] Calcul des embeddings textuels & Indexation globale dans PGVector...")
    indexer = PGVectorIndexer()
    indexer.index_nodes(final_nodes)

    # -------------------------------------------------------------------------
    # Étape 5 : Vérification dans la base PostgreSQL / PGVector
    # -------------------------------------------------------------------------
    db_stats = indexer.get_database_stats()
    clip_count = get_clip_database_stats()

    print("\n[ÉTAPE 5] État des tables PGVector dans PostgreSQL :")
    print(f"  Base de données : {PG_DB} sur {PG_HOST}")
    print(f"  1. Table Principale (LlamaIndex BGE 384D) : {db_stats.get('table_name')}")
    print(f"     Total des vecteurs : {db_stats.get('total_rows')}")
    for doc_type, count in db_stats.get("by_type", {}).items():
        print(f"       - {str(doc_type).upper():<10} : {count} vecteurs")
    print(f"  2. Table Visuelle Dédiée (CLIP 512D)      : {CLIP_TABLE}")
    print(f"     Total des vecteurs CLIP : {clip_count} vecteurs")

    # -------------------------------------------------------------------------
    # Étape 6 : Démonstration de Recherche Sémantique & Visuelle CLIP
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 DÉMONSTRATION DE RECHERCHE MULTI-MODALE & VISUELLE CLIP")
    print("=" * 80)

    # A. Recherche sémantique par description
    searcher = MultiModalSearchEngine()

    q1 = "What is the Scaled Dot-Product Attention and how is it computed?"
    res1 = searcher.search(query=q1, top_k=2, node_type="text")
    searcher.display_results(q1, res1, header="Recherche Sémantique Texte")

    q2 = "BLEU scores comparison between Transformer and other models"
    res2 = searcher.search(query=q2, top_k=2, node_type="table")
    searcher.display_results(q2, res2, header="Recherche Sémantique Tableau")

    # B. Recherche d'image via description textuelle indexée
    q3 = "Architecture diagram of the Transformer encoder decoder model"
    res3 = searcher.search(query=q3, top_k=2, node_type="image")
    searcher.display_results(q3, res3, header="Recherche Image via Description Générée (LlamaIndex / PGVector)")

    # C. Recherche Visuelle Directe Zero-Shot avec CLIP (Text-to-Image dans PGVector)
    if clip_count > 0:
        print("\n" + "=" * 70)
        print("🎯 RECHERCHE VISUELLE ZERO-SHOT DIRECTE AVEC CLIP (Text-to-Image)")
        print(f"   Espace vectoriel CLIP (512 dimensions) interrogeant la table '{CLIP_TABLE}'")
        print("=" * 70)

        clip_queries = [
            "neural network encoder decoder architecture block diagram",
            "scaled dot-product and multi-head attention diagrams"
        ]

        for cq in clip_queries:
            clip_res = clip_processor.search_images_clip(query=cq, top_k=2)
            print(f"\n🔍 REQUÊTE CLIP : '{cq}'")
            print(f"📋 Résultats Text-to-Image ({len(clip_res)} images correspondantes) :")
            for rank, r in enumerate(clip_res, start=1):
                print(f"  [#{rank}] FICHIER : {r['image_filename']} | PAGE : {r['page_number']} | SIMILARITÉ CLIP : {r['similarity_score']:.4f}")
                print(f"       Légende doc : {r['caption']}")
                print(f"       Tags CLIP   : {r['visual_tags']}")
                print(f"       Description : {r['generated_description'][:220]}...")
                print(f"       Chemin      : {r['image_path']}")

    # -------------------------------------------------------------------------
    # Étape 7 : Démonstration Question-Réponse RAG avec Qwen via Ollama
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🤖 QUESTION-RÉPONSE & GÉNÉRATION RAG AVEC LE LLM QWEN (OLLAMA)")
    print("=" * 80)

    rag = TechnicalRAGPipeline()
    rag_questions = [
        "Quelle est la formule mathématique de la Scaled Dot-Product Attention et quel est le rôle de dk ?",
        "Quels sont les résultats du modèle Transformer par rapport aux modèles précédents en termes de score BLEU ?"
    ]

    for q in rag_questions:
        qa_result = rag.ask(query=q, top_k=2)
        rag.display_answer(qa_result)

    print("\n" + "=" * 80)
    print("✅ Pipeline d'extraction, indexation, CLIP et RAG Qwen terminé avec succès !")
    print("=" * 80)


if __name__ == "__main__":
    from config import PG_PASSWORD
    parser = argparse.ArgumentParser(description="Pipeline d'extraction, CLIP et indexation PGVector")
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF),
        help=f"Chemin vers le fichier PDF (par défaut: {DEFAULT_PDF.name})"
    )
    args = parser.parse_args()
    run_pipeline(Path(args.pdf))
