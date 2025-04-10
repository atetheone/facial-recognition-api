import os

class Config:
    """Configuration de base de l'application"""
    # Dossiers de données
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'data/uploads')
    KNOWN_FACES_FOLDER = os.environ.get('KNOWN_FACES_FOLDER', 'data/known_faces')
    
    # Limites et contraintes
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Paramètres de reconnaissance faciale
    FACE_RECOGNITION_THRESHOLD = 0.6  # Seuil de confiance pour la reconnaissance
    MODEL_COMPLEXITY = 1  # 0=moins précis mais plus rapide, 1=plus précis mais plus lent

class DevelopmentConfig(Config):
    """Configuration de développement"""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Configuration de test"""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = 'tests/data/uploads'
    KNOWN_FACES_FOLDER = 'tests/data/known_faces'

class ProductionConfig(Config):
    """Configuration de production"""
    DEBUG = False
    TESTING = False
    
    # En production, on peut utiliser des chemins absolus ou des volumes Docker
    UPLOAD_FOLDER = '/app/data/uploads'
    KNOWN_FACES_FOLDER = '/app/data/known_faces'

# Configuration par défaut
config_by_name = {
    'dev': DevelopmentConfig,
    'test': TestingConfig,
    'prod': ProductionConfig,
}

# Configuration active (basée sur une variable d'environnement ou dev par défaut)
Config = config_by_name[os.environ.get('FLASK_ENV', 'dev')]