import os
import face_recognition
from werkzeug.utils import secure_filename
from flask import current_app
from app.utils.file_utils import allowed_file

# Variables globales pour stocker les visages connus
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Charger tous les visages connus depuis le dossier de stockage"""
    global known_face_encodings, known_face_names
    
    # Réinitialiser les listes
    known_face_encodings = []
    known_face_names = []
    
    known_faces_folder = current_app.config['KNOWN_FACES_FOLDER']
    
    for filename in os.listdir(known_faces_folder):
        if allowed_file(filename):
            # Extraire le nom de la personne du nom de fichier
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(known_faces_folder, filename)
            
            try:
                # Charger l'image
                image = face_recognition.load_image_file(image_path)
                
                # Essayer de détecter un visage
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    # Prendre le premier visage trouvé
                    encoding = face_encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    current_app.logger.info(f"Visage chargé: {name}")
                else:
                    current_app.logger.warning(f"Aucun visage détecté dans {filename}")
            except Exception as e:
                current_app.logger.error(f"Erreur lors du chargement de {filename}: {str(e)}")
    
    return len(known_face_names)

def register_face(file, name):
    """Enregistrer un nouveau visage"""
    if not file or not name:
        return {'success': False, 'error': "Le fichier et le nom sont requis"}, 400
    
    if not allowed_file(file.filename):
        return {'success': False, 'error': "Type de fichier non autorisé"}, 400
    
    try:
        # Sécuriser le nom de fichier
        filename = f"{secure_filename(name)}.jpg"
        known_faces_folder = current_app.config['KNOWN_FACES_FOLDER']
        filepath = os.path.join(known_faces_folder, filename)
        
        # Sauvegarder le fichier
        file.save(filepath)
        
        # Vérifier si un visage est détecté
        image = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            # Supprimer le fichier si aucun visage n'est détecté
            os.remove(filepath)
            return {'success': False, 'error': "Aucun visage détecté dans l'image"}, 400
        
        # Recharger tous les visages
        load_known_faces()
        
        return {'success': True, 'message': f"Visage de {name} enregistré avec succès"}, 200
    
    except Exception as e:
        current_app.logger.error(f"Erreur lors de l'enregistrement du visage: {str(e)}")
        return {'success': False, 'error': f"Erreur lors de l'enregistrement: {str(e)}"}, 500

def delete_face(name):
    """Supprimer un visage enregistré"""
    filename = f"{secure_filename(name)}.jpg"
    known_faces_folder = current_app.config['KNOWN_FACES_FOLDER']
    filepath = os.path.join(known_faces_folder, filename)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            # Recharger les visages
            load_known_faces()
            return {'success': True, 'message': f"Visage de {name} supprimé avec succès"}, 200
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la suppression du visage: {str(e)}")
            return {'success': False, 'error': f"Erreur lors de la suppression: {str(e)}"}, 500
    else:
        return {'success': False, 'error': f"Aucun visage trouvé pour {name}"}, 404

def list_faces():
    """Lister tous les visages enregistrés"""
    return {
        'success': True,
        'known_faces': known_face_names,
        'count': len(known_face_names)
    }, 200

def get_known_faces():
    """Retourner les visages connus pour le service de reconnaissance"""
    return known_face_encodings, known_face_names