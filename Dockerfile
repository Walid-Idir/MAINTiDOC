# =============================================================================
# MAINTiDOC - Dockerfile Multi-Modal RAG & Interface Streamlit
# =============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système pour PyMuPDF, Pillow et communications réseau
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copie du code applicatif
COPY . .

# Port exposé pour Streamlit
EXPOSE 8501

# Santé du conteneur Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancement de l'application Streamlit
CMD ["streamlit", "run", "Partie 1/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
