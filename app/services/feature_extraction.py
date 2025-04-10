import cv2
import numpy as np
import face_recognition
from enum import Enum
import os
import dlib
from flask import current_app
from app.utils.image_utils import enhance_image_for_detection


class FeatureExtractionMethod(Enum):
	"""Méthodes disponibles pour l'extraction de caractéristiques faciales"""

	HOG = "hog"  # Histogramme de Gradients Orientés (plus rapide, moins précis)
	CNN = "cnn"  # Réseau de Neurones Convolutifs (plus précis, plus lent)
	CUSTOM_HOG = "custom_hog"  # Notre implémentation personnalisée HOG
	LBP = "lbp"  # Local Binary Patterns (rapide, simple)


class FeatureExtractor:
    """
    Classe pour gérer l'extraction de caractéristiques faciales
    avec différentes méthodes
    """

    def __init__(self, method=FeatureExtractionMethod.HOG):
        """
        Initialiser l'extracteur avec la méthode spécifiée

        Args:
            method: Méthode d'extraction (par défaut: HOG)
        """
        self.set_method(method)

        # Charger le modèle CNN pré-entraîné si nécessaire
        if method == FeatureExtractionMethod.CNN:
            # Vérifier si le modèle est disponible
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "models",
                "mmod_human_face_detector.dat",
            )
            if os.path.exists(model_path):
                self.cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)
            else:
                current_app.logger.warning(
                    f"Modèle CNN non trouvé à {model_path}, retour à HOG"
                )
                self.method = FeatureExtractionMethod.HOG

    def set_method(self, method):
        """
        Définir la méthode d'extraction

        Args:
            method: Méthode d'extraction
        """
        if isinstance(method, str):
            try:
                self.method = FeatureExtractionMethod(method.lower())
            except ValueError:
                self.method = FeatureExtractionMethod.HOG
                current_app.logger.warning(
                    f"Méthode '{method}' non reconnue, utilisation de HOG par défaut"
                )
        else:
            self.method = method

    def detect_faces(self, image):
        """
        Détecter les visages dans une image avec la méthode configurée

        Args:
            image: Image au format numpy array

        Returns:
            List de rectangles (top, right, bottom, left) ou liste vide
        """
        if self.method == FeatureExtractionMethod.HOG:
            # Utiliser la méthode HOG de face_recognition
            face_locations = face_recognition.face_locations(image, model="hog")
            return face_locations

        elif self.method == FeatureExtractionMethod.CNN:
            # Utiliser la méthode CNN de face_recognition
            face_locations = face_recognition.face_locations(image, model="cnn")
            return face_locations

        elif self.method == FeatureExtractionMethod.CUSTOM_HOG:
            # Notre implémentation personnalisée HOG
            return self._custom_hog_detection(image)

        elif self.method == FeatureExtractionMethod.LBP:
            # Méthode LBP avec OpenCV
            return self._lbp_detection(image)

        else:
            # Méthode par défaut (HOG)
            return face_recognition.face_locations(image)

    def extract_features(self, image, face_locations):
        """
        Extraire les caractéristiques faciales à partir de visages détectés

        Args:
            image: Image au format numpy array
            face_locations: Liste de rectangles (top, right, bottom, left)

        Returns:
            Liste d'encodages faciaux (vecteurs de caractéristiques)
        """
        if (
            self.method == FeatureExtractionMethod.CUSTOM_HOG
            or self.method == FeatureExtractionMethod.LBP
        ):
            # Utiliser notre propre méthode d'extraction de caractéristiques
            return self._custom_feature_extraction(image, face_locations)
        else:
            # Utiliser l'encodage standard de face_recognition
            return face_recognition.face_encodings(image, face_locations)

    def _custom_hog_detection(self, image):
        """
        Implémentation personnalisée de détection de visage avec HOG

        Args:
            image: Image au format numpy array

        Returns:
            Liste de rectangles (top, right, bottom, left)
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Chargement du classifier Haar Cascade pour la détection de visages
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Détection de visages
        opencv_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Convertir au format face_recognition (top, right, bottom, left)
        face_locations = []
        for x, y, w, h in opencv_faces:
            face_locations.append((y, x + w, y + h, x))  # top, right, bottom, left

        return face_locations

    def _lbp_detection(self, image):
        """
        Détection de visage avec Local Binary Patterns (LBP)

        Args:
            image: Image au format numpy array

        Returns:
            Liste de rectangles (top, right, bottom, left)
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Chargement du classifier LBP pour la détection de visages
        lbp_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "lbpcascade_frontalface.xml"
        )

        # Détection de visages avec LBP
        opencv_faces = lbp_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Convertir au format face_recognition (top, right, bottom, left)
        face_locations = []
        for x, y, w, h in opencv_faces:
            face_locations.append((y, x + w, y + h, x))  # top, right, bottom, left

        return face_locations

    def _custom_feature_extraction(self, image, face_locations):
        """
        Extraction personnalisée de caractéristiques faciales
        Combine HOG + LBP pour un vecteur de caractéristiques robuste

        Args:
            image: Image au format numpy array
            face_locations: Liste de rectangles (top, right, bottom, left)

        Returns:
            Liste d'encodages faciaux (vecteurs de caractéristiques)
        """
        features = []

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = gray[top:bottom, left:right]

            # Redimensionner pour standardiser
            face_image = cv2.resize(face_image, (128, 128))

            # 1. Caractéristiques HOG
            hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            h = hog.compute(cv2.resize(face_image, (64, 64)))

            # 2. Caractéristiques LBP
            lbp_features = self._extract_lbp_features(face_image)

            # 3. Combiner les caractéristiques et normaliser
            combined_features = np.concatenate([h.flatten(), lbp_features.flatten()])

            # Normalisation
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm

            features.append(combined_features)

        return features

    def _extract_lbp_features(self, image):
        """
        Extraire les caractéristiques LBP d'une image

        Args:
            image: Image en niveaux de gris

        Returns:
            Vecteur de caractéristiques LBP
        """
        lbp_image = np.zeros_like(image)

        # Pour chaque pixel (sauf les bords)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0

                # Parcourir les 8 voisins
                code |= (image[i - 1, j - 1] >= center) << 7
                code |= (image[i - 1, j] >= center) << 6
                code |= (image[i - 1, j + 1] >= center) << 5
                code |= (image[i, j + 1] >= center) << 4
                code |= (image[i + 1, j + 1] >= center) << 3
                code |= (image[i + 1, j] >= center) << 2
                code |= (image[i + 1, j - 1] >= center) << 1
                code |= (image[i, j - 1] >= center) << 0

                lbp_image[i, j] = code

        # Calculer l'histogramme LBP
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])

        # Normaliser l'histogramme
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7

        return hist
