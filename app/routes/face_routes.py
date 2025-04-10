from flask import Blueprint, request, jsonify, send_from_directory, current_app
from app.services.face_service import register_face, delete_face, list_faces, load_known_faces
from app.services.recognition_service import recognize_faces

# Créer le blueprint
face_bp = Blueprint('face', __name__)

# Charger les visages au démarrage
@face_bp.before_app_first_request
def initialize():
    """Initialiser les données au démarrage de l'application"""
    count = load_known_faces()
    current_app.logger.info(f"{count} visages chargés au démarrage")

@face_bp.route('/register_face', methods=['POST'])
def register_face_route():
    """
    Enregistrer un nouveau visage
    ---
    tags:
      - Faces
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Image contenant un visage
      - name: name
        in: formData
        type: string
        required: true
        description: Nom de la personne
    responses:
      200:
        description: Visage enregistré avec succès
      400:
        description: Erreur dans la requête
      500:
        description: Erreur serveur
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Aucun fichier n'a été soumis"}), 400
    
    file = request.files['file']
    name = request.form.get('name')
    
    result, status_code = register_face(file, name)
    return jsonify(result), status_code

@face_bp.route('/recognize', methods=['POST'])
def recognize_faces_route():
    """
    Reconnaître les visages dans une image
    ---
    tags:
      - Faces
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Image à analyser
    responses:
      200:
        description: Résultats de la reconnaissance
      400:
        description: Erreur dans la requête
      500:
        description: Erreur serveur
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Aucun fichier n'a été soumis"}), 400
    
    file = request.files['file']
    
    # Option pour désactiver la sauvegarde de l'image (par défaut: activée)
    save_result = request.form.get('save_result', 'true').lower() != 'false'
    
    result, status_code = recognize_faces(file, save_result)
    return jsonify(result), status_code

@face_bp.route('/get_image/<filename>')
def get_image_route(filename):
    """
    Récupérer une image traitée
    ---
    tags:
      - Faces
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Nom du fichier image
    responses:
      200:
        description: Image
      404:
        description: Image non trouvée
    """
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@face_bp.route('/list_known_faces')
def list_faces_route():
    """
    Lister tous les visages connus
    ---
    tags:
      - Faces
    responses:
      200:
        description: Liste des visages connus
    """
    result, status_code = list_faces()
    return jsonify(result), status_code

@face_bp.route('/delete_face/<name>')
def delete_face_route(name):
    """
    Supprimer un visage
    ---
    tags:
      - Faces
    parameters:
      - name: name
        in: path
        type: string
        required: true
        description: Nom de la personne
    responses:
      200:
        description: Visage supprimé avec succès
      404:
        description: Visage non trouvé
      500:
        description: Erreur serveur
    """
    result, status_code = delete_face(name)
    return jsonify(result), status_code