from flask import Blueprint, jsonify, current_app

# Créer le blueprint
utils_bp = Blueprint('utils', __name__)

@utils_bp.route('/')
def index():
    """
    Route de base pour vérifier que l'API est en ligne
    ---
    tags:
      - Utils
    responses:
      200:
        description: API en ligne
    """
    return jsonify({
        "status": "online",
        "message": "API de reconnaissance faciale",
        "version": "1.0.0"
    })

@utils_bp.route('/health')
def health_check():
    """
    Vérification de santé de l'API
    ---
    tags:
      - Utils
    responses:
      200:
        description: État de santé de l'API
    """
    return jsonify({
        "status": "healthy",
        "environment": current_app.config.get('ENV', 'development')
    })

@utils_bp.errorhandler(404)
def not_found(e):
    """Gestionnaire d'erreur pour les routes non trouvées"""
    return jsonify({"success": False, "error": "Route non trouvée"}), 404

@utils_bp.errorhandler(500)
def server_error(e):
    """Gestionnaire d'erreur pour les erreurs serveur"""
    return jsonify({"success": False, "error": "Erreur serveur interne"}), 500

@utils_bp.errorhandler(413)
def request_entity_too_large(e):
    """Gestionnaire d'erreur pour les fichiers trop volumineux"""
    return jsonify({
        "success": False, 
        "error": "Fichier trop volumineux. Limite: 16MB"
    }), 413