from flask import Flask
from flask_cors import CORS
import os
from app.config import Config

def create_app(config_class=Config):
    """Factory pattern pour créer l'application Flask"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Activer CORS
    CORS(app)
    
    # Créer les dossiers nécessaires
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['KNOWN_FACES_FOLDER'], exist_ok=True)
    
    # Enregistrer les blueprints
    from app.routes.face_routes import face_bp
    from app.routes.utils_routes import utils_bp
    
    app.register_blueprint(face_bp)
    app.register_blueprint(utils_bp)
    
    return app