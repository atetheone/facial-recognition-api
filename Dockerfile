FROM python:3.10-slim as builder

WORKDIR /app

# Installer les dépendances système nécessaires pour la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers requis
COPY requirements.txt .

# Installer les dépendances Python dans un environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Image finale plus légère
FROM python:3.10-slim

# Copier les dépendances depuis l'image builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installer les dépendances runtime minimales
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Créer la structure de répertoires
RUN mkdir -p data/uploads data/known_faces models

# Copier le code source
COPY app/ ./app/
COPY client/ ./client/
COPY app.py requirements.txt ./

# Variables d'environnement
ENV FLASK_ENV=prod
ENV PORT=5000
ENV HOST=0.0.0.0

# Exposer le port
EXPOSE 5000

# Commande de démarrage
CMD ["python", "app.py"]