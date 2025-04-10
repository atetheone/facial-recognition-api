import cv2
import numpy as np
from PIL import Image
import io

def resize_image(image_data, max_size=1024):
    """
    Redimensionne une image si elle est trop grande
    
    Args:
        image_data: Données binaires de l'image
        max_size: Taille maximale en pixels (largeur ou hauteur)
        
    Returns:
        bytes: Données de l'image redimensionnée au format JPEG
    """
    image = Image.open(io.BytesIO(image_data))
    width, height = image.size
    
    # Si l'image est déjà assez petite, la retourner telle quelle
    if width <= max_size and height <= max_size:
        return image_data
    
    # Calculer le ratio de redimensionnement
    ratio = min(max_size / width, max_size / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Redimensionner l'image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Convertir en bytes
    output = io.BytesIO()
    resized_image.save(output, format='JPEG', quality=85)
    
    return output.getvalue()

def enhance_image_for_detection(image_array):
    """
    Améliore une image pour la détection faciale
    - Normalisation de la luminosité
    - Augmentation du contraste
    
    Args:
        image_array: Tableau numpy de l'image
        
    Returns:
        numpy.ndarray: Image améliorée
    """
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Égalisation d'histogramme adaptative (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Reconvertir en RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb