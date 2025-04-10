# Dans app/routes/recognition.py

from flask import Blueprint, request, jsonify
from app.services.face_detector import detect_face
from app.services.traditional import verify_face_traditional
from app.services.cnn import verify_face_cnn

recognition_bp = Blueprint('recognition', __name__)

@recognition_bp.route('/enroll', methods=['POST'])
def enroll_user():
    """Enrôler un nouvel utilisateur avec ses données faciales"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    user_id = request.form.get('user_id')
    image = request.files['image']
    
    # Traitement de l'image et enregistrement des données faciales
    # ...
    
    return jsonify({'success': True, 'message': 'User enrolled successfully'})

@recognition_bp.route('/verify', methods=['POST'])
def verify_user():
    """Vérifier l'identité d'un utilisateur via reconnaissance faciale"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    user_id = request.form.get('user_id')
    image = request.files['image']
    method = request.form.get('method', 'cnn')  # Méthode par défaut: CNN
    
    if method == 'traditional':
        result = verify_face_traditional(user_id, image)
    else:
        result = verify_face_cnn(user_id, image)
    
    return jsonify(result)

@recognition_bp.route('/detect', methods=['POST'])
def detect_faces():
    """Détecter les visages dans une image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    faces = detect_face(image)
    
    return jsonify({'faces': faces})