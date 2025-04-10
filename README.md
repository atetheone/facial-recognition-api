# API de Reconnaissance Faciale

Une API robuste pour l'enregistrement et la reconnaissance de visages, développée en Python avec Flask et des techniques avancées de vision par ordinateur.

## Fonctionnalités

- Enregistrement de visages avec un nom associé
- Reconnaissance de visages dans des images avec plusieurs méthodes
- Extraction de caractéristiques faciales avec différentes approches (HOG, CNN, LBP)
- Modèle CNN personnalisé pour la reconnaissance faciale
- Gestion d'une base de données locale de visages connus
- Affichage des résultats avec les noms et le niveau de confiance
- Interface API RESTful modulaire

## Méthodes d'extraction de caractéristiques

L'API implémente plusieurs méthodes d'extraction de caractéristiques faciales:

1. **HOG (Histogramme de Gradients Orientés)** - Rapide et efficace pour la détection faciale
2. **CNN (Réseau de Neurones Convolutifs)** - Plus précis mais plus lent, basé sur le modèle CNN de dlib
3. **Custom HOG** - Implémentation personnalisée utilisant OpenCV
4. **LBP (Local Binary Patterns)** - Méthode légère et robuste aux variations d'éclairage

## Architecture du projet

```
facial-recognition-api/
│
├── app/                          # Dossier principal de l'application
│   ├── __init__.py               # Initialise le package Python
│   ├── config.py                 # Configuration de l'application
│   ├── routes/                   # Endpoints de l'API
│   │   ├── __init__.py
│   │   ├── face_routes.py        # Routes liées aux visages
│   │   └── utils_routes.py       # Routes utilitaires
│   │
│   ├── services/                 # Logique métier
│   │   ├── __init__.py
│   │   ├── face_service.py       # Service de gestion des visages
│   │   ├── recognition_service.py # Service de reconnaissance 
│   │   ├── feature_extraction.py # Extraction de caractéristiques
│   │   └── cnn_model.py          # Modèle CNN personnalisé
│   │
│   └── utils/                    # Utilitaires
│       ├── __init__.py
│       ├── file_utils.py         # Gestion des fichiers
│       └── image_utils.py        # Traitement d'images
│
├── data/                         # Données persistantes
│   ├── known_faces/              # Visages enregistrés
│   └── uploads/                  # Fichiers temporaires
│
├── models/                       # Modèles entraînés
│
├── client/                       # Client de test
│   └── client.py
│
├── app.py                        # Point d'entrée de l'application
├── requirements.txt              # Dépendances
├── Dockerfile                    # Configuration Docker
├── docker-compose.yml            # Configuration Docker Compose
└── README.md                     # Documentation
```

## Prérequis

- Python 3.8+
- Docker (optionnel)
- OpenCV
- dlib
- TensorFlow (pour le modèle CNN personnalisé)

## Installation

### Méthode 1: Installation directe

1. Cloner le dépôt:
   ```bash
   git clone https://github.com/votre-nom/facial-recognition-api.git
   cd facial-recognition-api
   ```

2. Créer les dossiers nécessaires:
   ```bash
   mkdir -p data/known_faces data/uploads models
   ```

3. Installer les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

4. Lancer l'application:
   ```bash
   python app.py
   ```

### Méthode 2: Utilisation de Docker

1. Utiliser Docker Compose (recommandé):
   ```bash
   docker-compose up -d
   ```

2. Ou construire et lancer manuellement:
   ```bash
   docker build -t facial-recognition-api .
   docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models facial-recognition-api
   ```

## Utilisation de l'API

### Enregistrer un visage

```
POST /register_face
```

Paramètres:
- `file`: Fichier image contenant un visage (.jpg, .jpeg, .png)
- `name`: Nom de la personne

Exemple avec curl:
```bash
curl -X POST -F "file=@photo.jpg" -F "name=John" http://localhost:5000/register_face
```

### Reconnaître des visages dans une image

```
POST /recognize
```

Paramètres:
- `file`: Fichier image à analyser (.jpg, .jpeg, .png)
- `method`: (optionnel) Méthode d'extraction ('hog', 'cnn', 'custom_hog', 'lbp')
- `use_cnn_model`: (optionnel) Utiliser le modèle CNN personnalisé (true/false)

Exemple avec curl:
```bash
# Méthode par défaut (HOG)
curl -X POST -F "file=@group_photo.jpg" http://localhost:5000/recognize

# Avec CNN de dlib
curl -X POST -F "file=@group_photo.jpg" -F "method=cnn" http://localhost:5000/recognize

# Avec le modèle CNN personnalisé
curl -X POST -F "file=@group_photo.jpg" -F "use_cnn_model=true" http://localhost:5000/recognize
```

### Lister les visages connus

```
GET /list_known_faces
```

Exemple avec curl:
```bash
curl http://localhost:5000/list_known_faces
```

### Supprimer un visage

```
GET /delete_face/<nom>
```

Exemple avec curl:
```bash
curl http://localhost:5000/delete_face/John
```

## Utilisation du client de test

Un script client est fourni pour tester facilement l'API:

```bash
# Enregistrer un visage
python client/client.py register photo.jpg "John Doe"

# Reconnaître des visages dans une image
python client/client.py recognize group_photo.jpg

# Lister les visages enregistrés
python client/client.py list

# Supprimer un visage
python client/client.py delete "John Doe"
```

## Tests et évaluation

Pour tester les différentes méthodes d'extraction de caractéristiques:

```bash
# Méthode HOG (par défaut)
curl -X POST -F "file=@photo.jpg" -F "method=hog" http://localhost:5000/recognize

# Méthode CNN de dlib
curl -X POST -F "file=@photo.jpg" -F "method=cnn" http://localhost:5000/recognize

# Méthode personnalisée HOG
curl -X POST -F "file=@photo.jpg" -F "method=custom_hog" http://localhost:5000/recognize

# Méthode LBP
curl -X POST -F "file=@photo.jpg" -F "method=lbp" http://localhost:5000/recognize
```

Pour entraîner et utiliser le modèle CNN personnalisé:

```bash
# Enregistrer plusieurs visages d'abord
curl -X POST -F "file=@personne1.jpg" -F "name=Personne1" http://localhost:5000/register_face
curl -X POST -F "file=@personne2.jpg" -F "name=Personne2" http://localhost:5000/register_face

# Utiliser le modèle CNN pour la reconnaissance
curl -X POST -F "file=@test.jpg" -F "use_cnn_model=true" http://localhost:5000/recognize
```

## Notes techniques

- L'API est structurée de manière modulaire pour faciliter la maintenance et l'extension
- Le système utilise plusieurs méthodes d'extraction de caractéristiques pour s'adapter à différents cas d'usage
- L'architecture modulaire permet d'ajouter facilement de nouvelles méthodes de reconnaissance
- Le modèle CNN personnalisé utilise le transfer learning avec MobileNetV2 pour des performances optimales

## Limitations

- Les performances dépendent de la qualité des images
- Nécessite une bonne luminosité et des visages clairement visibles
- L'entraînement du modèle CNN nécessite idéalement plusieurs images par personne
- Sensibilité aux changements d'apparence (lunettes, barbe, etc.)

## Améliorations possibles

- Ajouter une interface web pour la gestion des visages
- Permettre plusieurs photos de référence par personne
- Implémenter une détection de vivacité (anti-spoofing)
- Ajouter une authentification à l'API
- Optimiser les performances pour de grandes bases de données
- Ajouter plus de méthodes d'extraction de caractéristiques