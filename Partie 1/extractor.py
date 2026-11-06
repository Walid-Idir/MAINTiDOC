"""
Module d'extraction multi-modale pour documents PDF techniques
Extrait le texte, les tableaux et les images avec leurs métadonnées
et les transforme en nœuds LlamaIndex prêts pour l'indexation.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pymupdf
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

from config import IMAGES_DIR, TABLES_DIR, TEXT_DIR


class PDFMultiModalExtractor:
    """
    Extracteur multimodal pour fichiers PDF:
    - Texte: découpé en chunks contextuels avec LlamaIndex SentenceSplitter
    - Tableaux: détectés automatiquement, convertis en Markdown et associés à leurs légendes
    - Images: extraites sous forme de fichiers PNG/JPEG avec capture des légendes contextuelles (ex: Figure 1)
    """

    def __init__(self, pdf_path: str | Path, chunk_size: int = 512, chunk_overlap: int = 50):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {self.pdf_path}")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    def _find_figure_captions(self, page_text: str) -> List[str]:
        """Recherche des mentions de figures (ex: 'Figure 1: ...') dans le texte de la page."""
        captions = []
        pattern = r"(Figure\s+\d+[:\.\-][^\n]+(?:\n[^\n]+)?)"
        matches = re.findall(pattern, page_text, flags=re.IGNORECASE)
        for m in matches:
            clean_cap = " ".join(m.split())
            if clean_cap not in captions:
                captions.append(clean_cap)
        return captions

    def _find_table_captions(self, page_text: str) -> List[str]:
        """Recherche des mentions de tables (ex: 'Table 1: ...') dans le texte de la page."""
        captions = []
        pattern = r"(Table\s+\d+[:\.\-][^\n]+(?:\n[^\n]+)?)"
        matches = re.findall(pattern, page_text, flags=re.IGNORECASE)
        for m in matches:
            clean_cap = " ".join(m.split())
            if clean_cap not in captions:
                captions.append(clean_cap)
        return captions

    def extract(self) -> Tuple[List[TextNode], Dict[str, Any]]:
        """
        Exécute l'extraction complète et retourne la liste des nœuds LlamaIndex ainsi qu'un rapport.
        """
        doc = pymupdf.open(self.pdf_path)
        nodes: List[TextNode] = []
        
        stats = {
            "source_file": self.pdf_path.name,
            "total_pages": len(doc),
            "text_chunks": 0,
            "tables_extracted": 0,
            "images_extracted": 0,
            "details": []
        }

        full_doc_text = []

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_text = page.get_text()
            full_doc_text.append(f"--- Page {page_num} ---\n" + page_text)
            
            page_info = {
                "page": page_num,
                "text_length": len(page_text),
                "tables": 0,
                "images": 0
            }

            # -------------------------------------------------------------
            # 1. Extraction des Tableaux
            # -------------------------------------------------------------
            table_captions = self._find_table_captions(page_text)
            table_finder = page.find_tables()
            tables = table_finder.tables if table_finder else []
            
            for t_idx, table in enumerate(tables):
                try:
                    md_table = table.to_markdown()
                    caption = table_captions[t_idx] if t_idx < len(table_captions) else f"Tableau {t_idx + 1} de la page {page_num}"
                    
                    # Sauvegarde sur disque
                    table_filename = f"page_{page_num}_table_{t_idx + 1}.md"
                    table_path = TABLES_DIR / table_filename
                    with open(table_path, "w", encoding="utf-8") as tf:
                        tf.write(f"# {caption}\n\n{md_table}")

                    # Création du nœud LlamaIndex
                    table_node_text = (
                        f"Type: Tableau Technique\n"
                        f"Document: {self.pdf_path.name}\n"
                        f"Page: {page_num}\n"
                        f"Légende: {caption}\n"
                        f"Contenu Structuré:\n{md_table}"
                    )
                    
                    node = TextNode(
                        text=table_node_text,
                        metadata={
                            "source_file": self.pdf_path.name,
                            "type": "table",
                            "page_number": page_num,
                            "table_id": t_idx + 1,
                            "caption": caption,
                            "rows": table.row_count,
                            "cols": table.col_count,
                            "asset_path": str(table_path)
                        }
                    )
                    nodes.append(node)
                    stats["tables_extracted"] += 1
                    page_info["tables"] += 1
                except Exception as e:
                    print(f"[!] Avertissement extraction table page {page_num}: {e}")

            # -------------------------------------------------------------
            # 2. Extraction des Images / Figures
            # -------------------------------------------------------------
            img_list = page.get_images(full=True)
            fig_captions = self._find_figure_captions(page_text)
            
            for img_idx, img in enumerate(img_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Filtrer les icônes ou éléments décoratifs minuscules (< 50x50 px)
                    if width < 60 and height < 60:
                        continue

                    img_filename = f"page_{page_num}_img_{img_idx + 1}_{xref}.{image_ext}"
                    img_path = IMAGES_DIR / img_filename
                    with open(img_path, "wb") as imf:
                        imf.write(image_bytes)

                    caption = fig_captions[img_idx] if img_idx < len(fig_captions) else f"Figure/Schéma {img_idx + 1} de la page {page_num}"

                    # Création du nœud LlamaIndex pour l'image
                    image_node_text = (
                        f"Type: Image / Schéma Technique / Figure\n"
                        f"Document: {self.pdf_path.name}\n"
                        f"Page: {page_num}\n"
                        f"Légende / Description: {caption}\n"
                        f"Format: {image_ext.upper()} ({width}x{height} px)\n"
                        f"Fichier image extrait: {img_filename}"
                    )
                    
                    node = TextNode(
                        text=image_node_text,
                        metadata={
                            "source_file": self.pdf_path.name,
                            "type": "image",
                            "page_number": page_num,
                            "image_id": img_idx + 1,
                            "caption": caption,
                            "dimensions": f"{width}x{height}",
                            "format": image_ext,
                            "asset_path": str(img_path)
                        }
                    )
                    nodes.append(node)
                    stats["images_extracted"] += 1
                    page_info["images"] += 1
                    
                    # Enregistrer dans la liste d'images pour le traitement CLIP
                    if "extracted_images" not in stats:
                        stats["extracted_images"] = []
                    stats["extracted_images"].append({
                        "path": str(img_path),
                        "page_number": page_num,
                        "source_file": self.pdf_path.name,
                        "caption": caption,
                        "dimensions": f"{width}x{height}",
                        "format": image_ext
                    })
                except Exception as e:
                    print(f"[!] Avertissement extraction image page {page_num}: {e}")

            # -------------------------------------------------------------
            # 3. Extraction et Découpage du Texte
            # -------------------------------------------------------------
            if page_text.strip():
                if "page_texts" not in stats:
                    stats["page_texts"] = {}
                stats["page_texts"][page_num] = page_text

                # Découpage du texte de la page en chunks avec LlamaIndex SentenceSplitter
                chunks = self.splitter.split_text(page_text)
                for c_idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 40:
                        continue  # ignorer les fragments insignifiants
                    
                    text_node_text = (
                        f"Type: Texte Documentaire\n"
                        f"Document: {self.pdf_path.name}\n"
                        f"Page: {page_num}\n"
                        f"Extrait:\n{chunk}"
                    )
                    
                    node = TextNode(
                        text=text_node_text,
                        metadata={
                            "source_file": self.pdf_path.name,
                            "type": "text",
                            "page_number": page_num,
                            "chunk_index": c_idx + 1
                        }
                    )
                    nodes.append(node)
                    stats["text_chunks"] += 1

            stats["details"].append(page_info)

        # Sauvegarde du texte brut complet pour référence
        with open(TEXT_DIR / f"{self.pdf_path.stem}_fulltext.txt", "w", encoding="utf-8") as ft:
            ft.write("\n\n".join(full_doc_text))

        doc.close()
        return nodes, stats
