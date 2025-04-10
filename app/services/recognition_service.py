import face_recognition
import numpy as np
import cv2
import os
from flask import current_app
from werkzeug.utils import secure_filename
from app.services.face_service import get_known_faces
from app.services.feature_extraction import FeatureExtractor, FeatureExtractionMethod
from app.services.cnn_model import CNNFaceModel

def recognize_faces(file, save_result=True, method='hog', use_cnn_model=False):
    """
    Reconnaître les visages dans une image
    
    Args:
        file: Fichier image à analyser
        save_result: Booléen indiquant s'il faut sauvegarder l'image annotée
        method: Méthode d'extraction de caractéristiques ('hog', 'cnn', 'custom_hog', 'lbp')
        use_cnn_model: Utiliser le modèle CNN personnalisé pour la reconnaissance
        
    Returns:
        Tuple (results, status_code)
    """
    if not file:
        return {'success': False, 'error': "Aucun fichier n'a été soumis"}, 400
    
    try:
        # Sauvegarder temporairement le fichier
        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Charger l'image
        image = face_recognition.load_image_file(filepath)
        
        # Initialiser l'extracteur de caractéristiques
        feature_extractor = FeatureExtractor(method=method)
        
        # Initialiser le modèle CNN si demandé
        cnn_model = None
        if use_cnn_model:
            cnn_model = CNNFaceModel()
            if not cnn_model.is_trained:
                # Entraîner le modèle si nécessaire
                known_faces_folder = current_app.config['KNOWN_FACES_FOLDER']
                cnn_model.train(known_faces_folder)
        
        # Détecter les visages avec la méthode spécifiée
        face_locations = feature_extractor.detect_faces(image)
        
        # Si aucun visage n'est détecté avec la méthode spécifiée, essayer une autre méthode
        if not face_locations and method != 'hog':
            current_app.logger.info(f"Aucun visage détecté avec la méthode {method}, essai avec HOG")
            feature_extractor.set_method(FeatureExtractionMethod.HOG)
            face_locations = feature_extractor.detect_faces(image)
        
        # Obtenir les données des visages connus (pour la méthode standard)
        known_face_encodings, known_face_names = get_known_faces()
        
        # Extraire les caractéristiques faciales
        # (sauf si on utilise le modèle CNN qui a sa propre méthode)
        if not use_cnn_model:
            face_encodings = feature_extractor.extract_features(image, face_locations)
        else:
            face_encodings = [True] * len(face_locations)  # Placeholder
        
        # Initialiser les résultats
        results = []
        output_filename = None
        
        # Si des visages sont détectés et que l'enregistrement est activé
        if face_locations and save_result:
            # Convertir l'image pour dessiner dessus
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Traiter chaque visage trouvé
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = face_location
            
            # Initialiser avec valeurs par défaut
            name = "Inconnu"
            confidence = 0.0
            
            # Si on utilise le modèle CNN personnalisé
            if use_cnn_model and cnn_model.is_trained:
                # Extraire le visage
                face_img = image[top:bottom, left:right]
                
                # Prédire avec le modèle CNN
                name, confidence = cnn_model.predict(face_img)
                
            # Sinon, utiliser la méthode standard
            elif known_face_encodings:
                # Calculer les distances aux visages connus
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                # Trouver le meilleur match
                best_match_index = np.argmin(face_distances)
                confidence = 1.0 - face_distances[best_match_index]
                
                # Vérifier si le match est suffisamment bon selon le seuil de confiance
                threshold = current_app.config['FACE_RECOGNITION_THRESHOLD']
                if confidence > threshold:
                    name = known_face_names[best_match_index]
            
            # Ajouter les résultats
            results.append({
                "id": i,
                "name": name,
                "confidence": float(confidence),
                "method": method if not use_cnn_model else "cnn_model",
                "location": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left)
                }
            })
            
            # Dessiner sur l'image si demandé
            if save_result and face_locations:
                # Rectangle autour du visage
                cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Afficher le nom et le score sous le visage
                label = f"{name} ({confidence:.2%})"
                cv2.putText(image_cv, label, (left, bottom + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Sauvegarder l'image avec les annotations si demandé
        if save_result and face_locations:
            output_filename = f"result_{filename}"
            output_path = os.path.join(upload_folder, output_filename)
            cv2.imwrite(output_path, image_cv)
        
        # Préparer la réponse
        response = {
            "success": True,
            "faces_detected": len(face_locations),
            "results": results,
            "method": method if not use_cnn_model else "cnn_model"
        }
        
        # Ajouter le chemin de l'image résultante si disponible
        if output_filename:
            response["output_image"] = output_filename
        
        return response, 200
        
    except Exception as e:
        current_app.logger.error(f"Erreur lors de la reconnaissance: {str(e)}")
        return {'success': False, 'error': str(e)}, 500