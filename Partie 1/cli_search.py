"""
CLI interactif pour tester des requêtes de recherche sémantique, visuelle CLIP,
et de Question-Réponse RAG avec Qwen sur la base PostgreSQL / PGVector.
"""

import sys
import argparse
from search_engine import MultiModalSearchEngine
from image_processor import CLIPImageProcessor

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Recherche Multi-Modale, CLIP & RAG Qwen dans PGVector")
    parser.add_argument("query", nargs="*", help="Texte de la requête ou question")
    parser.add_argument("--ask", "--rag", action="store_true", help="Générer une réponse avec le LLM Qwen fondé sur le contexte RAG")
    parser.add_argument("--clip", action="store_true", help="Utiliser la recherche visuelle directe CLIP (Text-to-Image)")
    parser.add_argument("--type", choices=["text", "table", "image"], default=None, help="Filtrer par type (text, table, image)")
    parser.add_argument("--top_k", type=int, default=3, help="Nombre de résultats / sources (par défaut: 3)")
    args = parser.parse_args()

    print("=" * 80)
    print("🔎 MOTEUR DE RECHERCHE MULTI-MODAL & PIPELINE RAG (MAINTiDOC)")
    print("=" * 80)

    query_str = " ".join(args.query).strip()

    # 1. Mode RAG (Question-Réponse avec Qwen)
    if args.ask:
        from rag_pipeline import TechnicalRAGPipeline
        if not query_str:
            query_str = "Quelle est la formule de la Scaled Dot-Product Attention et comment fonctionne-t-elle ?"
        
        rag = TechnicalRAGPipeline()
        result = rag.ask(query=query_str, top_k=args.top_k, node_type=args.type)
        rag.display_answer(result)
        return

    # 2. Mode recherche visuelle CLIP Text-to-Image
    if args.clip:
        if not query_str:
            query_str = "transformer encoder decoder architecture diagram"
        print(f"🎯 Mode : Recherche Visuelle Zero-Shot CLIP (Text-to-Image)")
        print(f"🔍 Requête : '{query_str}'\n")

        clip_proc = CLIPImageProcessor()
        results = clip_proc.search_images_clip(query=query_str, top_k=args.top_k)

        for rank, r in enumerate(results, start=1):
            print(f"[#{rank}] Image: {r['image_filename']} (Page {r['page_number']}) | Similarité CLIP: {r['similarity_score']:.4f}")
            print(f"     Légende : {r['caption']}")
            print(f"     Tags    : {r['visual_tags']}")
            print(f"     Fichier : {r['image_path']}\n")
        return

    # 3. Mode recherche sémantique standard (LlamaIndex + PGVector)
    searcher = MultiModalSearchEngine()
    if query_str:
        res = searcher.search(query=query_str, top_k=args.top_k, node_type=args.type)
        searcher.display_results(query_str, res, header="Résultats de recherche")
    else:
        # Démonstrations par défaut
        print("Aucune requête spécifiée. Exécution d'un exemple multi-modal :\n")
        
        # Exemple texte
        res_text = searcher.search("Scaled Dot-Product Attention", top_k=1, node_type="text")
        searcher.display_results("Scaled Dot-Product Attention", res_text, header="Exemple Texte")

        # Exemple tableau
        res_tab = searcher.search("BLEU score benchmarks", top_k=1, node_type="table")
        searcher.display_results("BLEU score benchmarks", res_tab, header="Exemple Tableau")

        # Exemple image CLIP
        print("\n" + "-" * 70)
        print("🎯 Exemple Recherche Visuelle CLIP (Text-to-Image) :")
        clip_proc = CLIPImageProcessor()
        c_res = clip_proc.search_images_clip("transformer architecture block diagram", top_k=1)
        if c_res:
            r = c_res[0]
            print(f"  Image trouvée : {r['image_filename']} (Similarité CLIP: {r['similarity_score']:.4f})")
            print(f"  Légende       : {r['caption']}")
            print(f"  Chemin        : {r['image_path']}")


if __name__ == "__main__":
    main()
