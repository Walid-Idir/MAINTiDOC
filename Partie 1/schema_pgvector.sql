-- =============================================================================
-- PROJET MAINTiDOC - SCHÉMAS POSTGRESQL / PGVECTOR
-- =============================================================================

-- 1. Activation de l'extension PGVector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table Principale LlamaIndex (Vecteurs Denses BGE 384D - Texte et Tableaux)
CREATE TABLE IF NOT EXISTS data_document_embeddings (
    id VARCHAR PRIMARY KEY,
    text TEXT,
    metadata_ JSONB,
    node_id VARCHAR,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_data_document_embeddings_hnsw 
ON data_document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 3. Table Multimodale Dédiée aux Schémas et Figures (Vecteurs CLIP 512D)
CREATE TABLE IF NOT EXISTS image_clip_embeddings (
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

CREATE INDEX IF NOT EXISTS idx_image_clip_embeddings_hnsw 
ON image_clip_embeddings USING hnsw (clip_embedding vector_cosine_ops);
