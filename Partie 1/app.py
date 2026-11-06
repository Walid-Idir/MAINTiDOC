"""
Interface Utilisateur Professionnelle - PROJET MAINTiDOC Enterprise
Pipeline RAG Multimodal & Orchestration Agentique avec LangChain, PGVector & CLIP
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
from PIL import Image

# Encodage UTF-8 pour Windows et ajout du dossier local au sys.path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

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
    EMBEDDING_MODEL_NAME,
    CLIP_MODEL_NAME,
    PROJECT_DIR,
    IMAGES_DIR,
    TABLES_DIR,
)
from extractor import PDFMultiModalExtractor
from image_processor import CLIPImageProcessor
from indexer import PGVectorIndexer
from rag_pipeline import TechnicalRAGPipeline


# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MAINTiDOC Enterprise - Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Style CSS Professionnel & Thème Moderne
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #1e3a8a 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 26px 32px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4), 0 8px 10px -6px rgba(15, 23, 42, 0.3);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(129, 140, 248, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #c7d2fe;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #e0e7ff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin-top: 6px;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(6px);
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Badges */
    .type-chip {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .chip-text { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .chip-table { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .chip-image { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

    /* Service Status Card */
    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .pulse-dot {
        width: 9px;
        height: 9px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
        display: inline-block;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Agent Thought Step */
    .agent-step {
        background: #f1f5f9;
        border-left: 4px solid #4f46e5;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    .agent-tool-tag {
        background: #e0e7ff;
        color: #3730a3;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 4px;
    }

    /* Image Card */
    .img-result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .img-result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching des Services & Modèles (Singletons)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialisation du Moteur RAG Direct (Qwen via Ollama & BGE)...")
def get_rag_pipeline():
    return TechnicalRAGPipeline()


@st.cache_resource(show_spinner="Chargement du Modèle Multimodal CLIP (512D)...")
def get_clip_processor():
    return CLIPImageProcessor()


@st.cache_resource(show_spinner="Initialisation de l'Orchestrateur Agentique LangChain...")
def get_agentic_orchestrator():
    from agentic_rag import AgenticRAGOrchestrator
    return AgenticRAGOrchestrator()


def get_pgvector_stats():
    """Interroge PostgreSQL pour récupérer les métriques de la base vectorielle."""
    stats = {"total": 0, "by_type": {}, "clip_total": 0, "documents": []}
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

        cur.execute(f"SELECT COUNT(*) AS total FROM {real_table};")
        row = cur.fetchone()
        stats["total"] = row["total"] if row else 0

        cur.execute(f"""
            SELECT metadata_->>'type' AS doc_type, COUNT(*) AS count
            FROM {real_table}
            GROUP BY metadata_->>'type'
            ORDER BY count DESC;
        """)
        for r in cur.fetchall():
            if r["doc_type"]:
                stats["by_type"][r["doc_type"]] = r["count"]

        cur.execute(f"""
            SELECT metadata_->>'source_file' AS doc_name, COUNT(*) AS count
            FROM {real_table}
            WHERE metadata_->>'source_file' IS NOT NULL
            GROUP BY metadata_->>'source_file'
            ORDER BY count DESC;
        """)
        stats["documents"] = cur.fetchall()

        cur.execute(f"SELECT COUNT(*) AS total FROM {CLIP_TABLE};")
        row_clip = cur.fetchone()
        stats["clip_total"] = row_clip["total"] if row_clip else 0

        cur.close()
        conn.close()
    except Exception as e:
        stats["error"] = str(e)
    return stats


def process_uploaded_pdf(pdf_path: Path):
    """Exécute l'extraction multimodale et l'indexation PGVector en temps réel."""
    status = st.status(f"⚙️ Indexation de '{pdf_path.name}' en cours...", expanded=True)
    with status:
        st.write("📄 **Phase 1/3** : Analyse du PDF et extraction contextuelle (PyMuPDF)...")
        extractor = PDFMultiModalExtractor(pdf_path=pdf_path)
        nodes, report = extractor.extract()
        st.write(
            f"✓ `{report['total_pages']}` pages • "
            f"`{report['text_chunks']}` paragraphes • "
            f"`{report['tables_extracted']}` tableaux structurés • "
            f"`{report['images_extracted']}` figures isolées"
        )

        st.write("🖼️ **Phase 2/3** : Encodage CLIP ViT-B/32 & classification visuelle zero-shot...")
        clip_proc = get_clip_processor()
        extracted_images = report.get("extracted_images", [])
        page_texts = report.get("page_texts", {})
        enriched_image_nodes = []
        if extracted_images:
            enriched_image_nodes = clip_proc.process_and_index_images(
                extracted_images=extracted_images,
                doc_pages_text=page_texts
            )
            st.write(f"✓ `{len(enriched_image_nodes)}` figures encodées en 512 dimensions et stockées dans `{CLIP_TABLE}`.")
        else:
            st.write("✓ Aucune image raster détectée dans ce document.")

        st.write("🧠 **Phase 3/3** : Calcul des vecteurs denses BGE (384D) & écriture PostgreSQL...")
        final_nodes = [n for n in nodes if n.metadata.get("type") != "image"] + enriched_image_nodes
        indexer = PGVectorIndexer()
        indexer.index_nodes(final_nodes)
        st.write(f"✓ `{len(final_nodes)}` nœuds vectorisés avec succès dans la table `data_{PG_TABLE}`.")

        status.update(label=f"✅ Document '{pdf_path.name}' indexé et prêt à l'interrogation !", state="complete", expanded=False)


# -----------------------------------------------------------------------------
# Barre Latérale (Sidebar Professionnelle)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ MAINTiDOC Enterprise")
    st.caption("Plateforme d'Intelligence Technique Industrielle")

    st.markdown("---")
    st.markdown("#### 📥 Chargement de Document")
    uploaded_file = st.file_uploader(
        "Importer un nouveau manuel technique (PDF)",
        type=["pdf"],
        help="Téléversez un manuel de maintenance, un schéma mécanique ou un rapport technique."
    )

    existing_pdfs = list(PROJECT_DIR.glob("*.pdf"))
    existing_pdf_names = [p.name for p in existing_pdfs]
    selected_existing = st.selectbox(
        "Ou sélectionner un document déjà présent :",
        options=["-- Aucun --"] + existing_pdf_names,
        index=0
    )

    if st.button("🚀 Lancer l'Indexation PGVector", use_container_width=True, type="primary"):
        target_path = None
        if uploaded_file is not None:
            target_path = PROJECT_DIR / uploaded_file.name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif selected_existing != "-- Aucun --":
            target_path = PROJECT_DIR / selected_existing

        if target_path:
            process_uploaded_pdf(target_path)
            st.success(f"Indexation finalisée pour '{target_path.name}' !")
            st.rerun()
        else:
            st.warning("Veuillez sélectionner ou importer un document PDF.")

    st.markdown("---")
    st.markdown("#### ⚙️ Configuration du RAG")
    rag_mode = st.radio(
        "Mode d'Orchestration :",
        options=[
            "🤖 Agent LangChain (ReAct & Multi-Outils)",
            "⚡ RAG Direct (LlamaIndex & Dense Search)"
        ],
        index=0,
        help="L'Agent LangChain sélectionne dynamiquement ses outils (BGE texte, CLIP visuel, inspection DB) avant de formuler une réponse fondée."
    )

    top_k = st.slider("Passages Documentaires (Top-K)", min_value=1, max_value=6, value=3)
    filter_type = st.selectbox(
        "Filtre de type de contenu :",
        options=["Tous", "text", "table", "image"],
        format_func=lambda x: {
            "Tous": "🌐 Tous les contenus",
            "text": "📝 Texte documentaire",
            "table": "📊 Tableaux de spécifications",
            "image": "🖼️ Schémas & Figures"
        }.get(x, x)
    )

    st.markdown("---")
    st.markdown("#### 🟢 État des Services Locaux")
    st.markdown("""
    <div style="font-size: 0.85rem; line-height: 1.8;">
        <div><span class="pulse-dot"></span><b>PostgreSQL 16</b> (PGVector) : Connecté</div>
        <div><span class="pulse-dot"></span><b>LLM Qwen 2.5</b> (Ollama 1.5B) : Actif</div>
        <div><span class="pulse-dot"></span><b>CLIP ViT-B/32</b> (512D) : Opérationnel</div>
        <div><span class="pulse-dot"></span><b>BGE Embeddings</b> (384D) : Prêt</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# En-tête Hero & Ruban de Métriques
# -----------------------------------------------------------------------------
stats = get_pgvector_stats()
total_vecs = stats.get("total", 0)
clip_vecs = stats.get("clip_total", 0)
num_docs = len(stats.get("documents", []))
tables_count = stats.get("by_type", {}).get("table", 0)

st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">⚡ Architecture Multi-Modale & LangChain Agentic RAG</div>
    <h1 class="hero-title">MAINTiDOC Enterprise</h1>
    <p class="hero-subtitle">
        Plateforme d'intelligence documentaire industrielle : interrogation augmentée (RAG),
        compréhension de tableaux de maintenance et recherche visuelle zero-shot (CLIP).
    </p>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 20px;">
        <div class="kpi-card">
            <div class="kpi-value">{num_docs}</div>
            <div class="kpi-label">Documents Indexés</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{total_vecs}</div>
            <div class="kpi-label">Fragments Vectorisés (PGVector)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{tables_count}</div>
            <div class="kpi-label">Tableaux Techniques</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{clip_vecs}</div>
            <div class="kpi-label">Schémas Vectorisés (CLIP)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Navigation Principale par Onglets
# -----------------------------------------------------------------------------
tab_agent, tab_clip, tab_explorer, tab_monitoring = st.tabs([
    "💬 Assistant Intelligent & Q&A",
    "🎯 Recherche Visuelle CLIP (Text-to-Image)",
    "📑 Galerie des Figures & Tableaux",
    "📊 Monitoring de la Base Vectorielle"
])


# =============================================================================
# Onglet 1 : Assistant Intelligent & Q&A
# =============================================================================
with tab_agent:
    st.markdown("##### 💡 Suggestions de questions techniques :")
    col_q1, col_q2, col_q3 = st.columns(3)
    sample_q = None
    with col_q1:
        if st.button("🔧 Vidange huile compresseur", use_container_width=True):
            sample_q = "Quelle est la periodicite de vidange de l'huile du compresseur et quelle reference d'huile utiliser ?"
    with col_q2:
        if st.button("🔩 Couples de serrage mécanique", use_container_width=True):
            sample_q = "Quels sont les couples de serrage pour la culasse et le flasque ?"
    with col_q3:
        if st.button("📐 Attention Transformer & Formule", use_container_width=True):
            sample_q = "Quelle est la formule de la Scaled Dot-Product Attention et comment est-elle calculee ?"

    # Historique de conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des échanges
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Si le message contient la trace de l'agent
            if msg.get("agentic") and msg.get("trace"):
                with st.expander(f"🛠️ Trace du Raisonnement Agentique ({len(msg['trace'])} étapes)", expanded=False):
                    for step in msg["trace"]:
                        st.markdown(f"**Étape {step['step']} : Action `{step['action']}`**")
                        st.caption(f"Arguments d'appel : `{step['args']}`")
                        st.markdown("**Observation PGVector :**")
                        st.code(step["observation"][:500] + ("..." if len(step["observation"]) > 500 else ""), language="markdown")
                        st.markdown("---")

            st.markdown(msg["content"])

            # Sources documentaires citées
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 {len(msg['sources'])} sources documentaires retrouvées"):
                    for src in msg["sources"]:
                        chip_class = f"chip-{src['type'].lower()}"
                        st.markdown(
                            f"<span class='type-chip {chip_class}'>{src['type']}</span> "
                            f"**Page {src['page']}** • Score similarité : `{src['score']:.4f}` • *{src['file']}*",
                            unsafe_allow_html=True
                        )
                        if src.get("caption"):
                            st.caption(f"📌 {src['caption']}")

                        meta = src.get("metadata", {})
                        asset_path = meta.get("asset_path")

                        if src["type"] == "IMAGE" and asset_path and os.path.exists(asset_path):
                            col_img, col_txt = st.columns([1, 2])
                            with col_img:
                                st.image(asset_path, caption=meta.get("image_filename", "Figure"), use_container_width=True)
                            with col_txt:
                                st.markdown(f"**Tags visuels CLIP :** `{meta.get('visual_tags', 'N/A')}`")
                                st.markdown(f"**Description contextuelle :**\n{meta.get('generated_description', src.get('text', ''))}")
                        elif src["type"] == "TABLE":
                            st.markdown(src.get("text", src.get("text_snippet", "")))
                        else:
                            st.markdown(f"> {src.get('text', src.get('text_snippet', ''))}")
                        st.markdown("---")

    # Champ de saisie
    user_query = st.chat_input("Posez votre question technique sur la documentation indexée...")
    if sample_q:
        user_query = sample_q

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if rag_mode.startswith("🤖"):
                # Mode Orchestration Agentique LangChain
                with st.spinner("🤖 L'Agent LangChain analyse la question et sélectionne les outils PGVector..."):
                    try:
                        agent = get_agentic_orchestrator()
                        result = agent.run(user_query)
                        answer = result.get("answer", "")
                        trace = result.get("trace", [])
                        tools_exec = result.get("tools_executed", [])

                        # Affichage élégant de la trace agentique
                        if trace:
                            with st.expander(f"🛠️ Trace du Raisonnement Agentique ({len(tools_exec)} actions d'outils)", expanded=True):
                                for step in trace:
                                    st.markdown(
                                        f"**Étape {step['step']}** : <span class='agent-tool-tag'>OUTIL : {step['action']}</span>",
                                        unsafe_allow_html=True
                                    )
                                    st.caption(f"Paramètres passés par le LLM : `{step['args']}`")
                                    st.text_area(
                                        "Observation retournée par PostgreSQL :",
                                        step["observation"][:600] + ("..." if len(step["observation"]) > 600 else ""),
                                        height=90,
                                        key=f"trace_{step['step']}_{step['action']}_{hash(user_query)}"
                                    )
                                    st.markdown("---")

                        st.markdown(answer)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "trace": trace,
                            "tools": tools_exec,
                            "agentic": True
                        })
                    except Exception as e:
                        st.error(f"Erreur Agentique : {e}")
            else:
                # Mode RAG Direct LlamaIndex
                with st.spinner("⚡ Recherche directe dans PGVector & Génération factuelle par Qwen..."):
                    try:
                        rag_pipe = get_rag_pipeline()
                        selected_type = None if filter_type == "Tous" else filter_type
                        result = rag_pipe.ask(query=user_query, top_k=top_k, node_type=selected_type)

                        answer = result.get("answer", "")
                        sources = result.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📚 {len(sources)} sources documentaires retrouvées (PGVector)", expanded=False):
                                for src in sources:
                                    chip_class = f"chip-{src['type'].lower()}"
                                    st.markdown(
                                        f"<span class='type-chip {chip_class}'>{src['type']}</span> "
                                        f"**Page {src['page']}** • Score similarité : `{src['score']:.4f}` • *{src['file']}*",
                                        unsafe_allow_html=True
                                    )
                                    if src.get("caption"):
                                        st.caption(f"📌 {src['caption']}")

                                    meta = src.get("metadata", {})
                                    asset_path = meta.get("asset_path")

                                    if src["type"] == "IMAGE" and asset_path and os.path.exists(asset_path):
                                        col_img, col_txt = st.columns([1, 2])
                                        with col_img:
                                            st.image(asset_path, caption=meta.get("image_filename", "Figure"), use_container_width=True)
                                        with col_txt:
                                            st.markdown(f"**Tags visuels CLIP :** `{meta.get('visual_tags', 'N/A')}`")
                                            st.markdown(f"**Description contextuelle :**\n{meta.get('generated_description', src.get('text', ''))}")
                                    elif src["type"] == "TABLE":
                                        st.markdown(src.get("text", src.get("text_snippet", "")))
                                    else:
                                        st.markdown(f"> {src.get('text', src.get('text_snippet', ''))}")
                                    st.markdown("---")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "agentic": False
                        })
                    except Exception as e:
                        st.error(f"Erreur lors de la génération RAG : {e}")


# =============================================================================
# Onglet 2 : Recherche Visuelle Zero-Shot CLIP (Text-to-Image)
# =============================================================================
with tab_clip:
    st.subheader("🎯 Recherche Visuelle Text-to-Image avec CLIP (512D)")
    st.markdown(
        "Interrogez directement l'espace vectoriel d'images stocké dans la table PostgreSQL `image_clip_embeddings`. "
        "Le modèle **CLIP** projette votre description textuelle et les pixels des schémas dans un même espace sémantique."
    )

    c_col1, c_col2 = st.columns([3, 1])
    with c_col1:
        clip_query = st.text_input(
            "Décrivez visuellement le schéma recherché :",
            value="schéma compresseur industriel et moteur électrique",
            help="Exemples: 'transformer model architecture diagram', 'circuit de lubrification', 'scaled dot-product attention'"
        )
    with c_col2:
        clip_top = st.number_input("Nombre de figures max :", min_value=1, max_value=6, value=2)

    # Suggestions de requêtes visuelles rapides
    col_v1, col_v2, col_v3 = st.columns(3)
    if col_v1.button("🖼️ Schéma Compresseur à Vis"):
        clip_query = "industrial screw compressor mechanical flow and piping schematic"
    if col_v2.button("📐 Architecture Transformer"):
        clip_query = "transformer model architecture diagram with encoder and decoder blocks"
    if col_v3.button("🔍 Mécanisme Attention"):
        clip_query = "scaled dot-product attention and multi-head attention mechanism diagram"

    if st.button("🔍 Exécuter la Recherche Visuelle CLIP", type="primary"):
        with st.spinner("Calcul du vecteur CLIP de la requête & interrogation cosinus dans PGVector..."):
            try:
                clip_proc = get_clip_processor()
                results = clip_proc.search_images_clip(query=clip_query, top_k=int(clip_top))

                if not results:
                    st.info("Aucune figure ne correspond à cette description visuelle.")
                else:
                    st.success(f"{len(results)} figure(s) identifiée(s) avec succès via distance cosinus !")
                    cols = st.columns(len(results))
                    for idx, res in enumerate(results):
                        with cols[idx]:
                            st.markdown(f"#### Figure #{idx + 1}")
                            img_path = res.get("image_path")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)

                            st.markdown(f"**Similarité CLIP :** `{res['similarity_score']:.4f}`")
                            st.markdown(f"**Document :** *{res['source_file']}* (Page {res['page_number']})")
                            if res.get("caption"):
                                st.markdown(f"**Légende :** *{res['caption']}*")
                            if res.get("visual_tags"):
                                st.caption(f"🏷️ **Tags Zero-Shot :** {res['visual_tags']}")
                            with st.expander("Voir la description technique générée"):
                                st.write(res.get("generated_description", "N/A"))
            except Exception as e:
                st.error(f"Erreur recherche CLIP : {e}")


# =============================================================================
# Onglet 3 : Galerie des Figures & Tableaux
# =============================================================================
with tab_explorer:
    st.subheader("📑 Galerie des Schémas et Tableaux Extraits")
    sub_t1, sub_t2 = st.tabs(["🖼️ Galerie des Schémas Techniques", "📊 Tableaux Structurés Markdown"])

    with sub_t1:
        images_found = list(IMAGES_DIR.glob("*.*"))
        if not images_found:
            st.info("Aucune figure extraite pour le moment.")
        else:
            st.caption(f"Total : {len(images_found)} figures sauvegardées dans `{IMAGES_DIR.name}` :")
            img_cols = st.columns(3)
            for idx, img_p in enumerate(images_found):
                with img_cols[idx % 3]:
                    st.image(str(img_p), caption=img_p.name, use_container_width=True)

    with sub_t2:
        tables_found = list(TABLES_DIR.glob("*.md"))
        if not tables_found:
            st.info("Aucun tableau extrait pour le moment.")
        else:
            st.caption(f"Total : {len(tables_found)} tableaux convertis en Markdown structuré :")
            for tbl_p in tables_found:
                with st.expander(f"📋 {tbl_p.name}"):
                    with open(tbl_p, "r", encoding="utf-8") as tf:
                        st.markdown(tf.read())


# =============================================================================
# Onglet 4 : Monitoring de la Base Vectorielle
# =============================================================================
with tab_monitoring:
    st.subheader("📊 Monitoring Technique de la Base PostgreSQL PGVector")

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("#### Répartition par Type d'Élément")
        by_type = stats.get("by_type", {})
        if by_type:
            for t_name, count in by_type.items():
                st.write(f"• **{str(t_name).upper()}** : `{count}` vecteurs dans `data_{PG_TABLE}`")
        else:
            st.info("Aucune statistique disponible.")

    with m_col2:
        st.markdown("#### Documents Déposés")
        docs = stats.get("documents", [])
        if docs:
            for d in docs:
                st.write(f"📄 **{d['doc_name']}** (`{d['count']}` fragments indexés)")
        else:
            st.info("Aucun document chargé.")

    st.markdown("---")
    st.markdown("#### Schémas des Tables PostgreSQL")
    st.code(f"""
-- Table Principale LlamaIndex (Embeddings BGE 384D)
CREATE TABLE data_{PG_TABLE} (
    id VARCHAR PRIMARY KEY,
    text TEXT,
    metadata_ JSONB,
    node_id VARCHAR,
    embedding vector(384)
);

-- Table Multimodale Dédiée aux Images (Vecteurs CLIP 512D)
CREATE TABLE {CLIP_TABLE} (
    id SERIAL PRIMARY KEY,
    image_filename VARCHAR(255) UNIQUE,
    image_path TEXT NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    page_number INT NOT NULL,
    dimensions VARCHAR(50),
    caption TEXT,
    visual_tags TEXT,
    generated_description TEXT NOT NULL,
    clip_embedding vector(512) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
    """, language="sql")
