import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import current_app
import pickle

class CNNFaceModel:
    """
    Classe pour la gestion d'un modèle CNN personnalisé pour la reconnaissance faciale
    Utilise une architecture basée sur MobileNetV2 (transfer learning)
    """
    
    def __init__(self):
        """Initialisation du modèle CNN"""
        self.model = None
        self.face_labels = []
        self.is_trained = False
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        
        # Créer le dossier des modèles s'il n'existe pas
        os.makedirs(self.model_path, exist_ok=True)
        
        # Chemins pour sauvegarder/charger le modèle
        self.model_file = os.path.join(self.model_path, 'face_recognition_model.h5')
        self.labels_file = os.path.join(self.model_path, 'face_labels.pkl')
        
        # Essayer de charger un modèle existant
        self._load_model()
    
    def _create_model(self, num_classes):
        """
        Créer l'architecture du modèle CNN
        
        Args:
            num_classes: Nombre de classes (personnes) à reconnaître
            
        Returns:
            Model Keras configuré
        """
        # Utiliser MobileNetV2 comme base (efficace et léger)
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Geler les couches du modèle de base
        base_model.trainable = False
        
        # Construire le modèle complet
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _load_model(self):
        """
        Charger un modèle pré-entraîné et ses labels s'ils existent
        
        Returns:
            bool: True si le modèle a été chargé, False sinon
        """
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
                # Charger le modèle
                self.model = models.load_model(self.model_file)
                
                # Charger les labels
                with open(self.labels_file, 'rb') as f:
                    self.face_labels = pickle.load(f)
                
                self.is_trained = True
                current_app.logger.info(f"Modèle CNN chargé avec {len(self.face_labels)} personnes")
                return True
        except Exception as e:
            current_app.logger.error(f"Erreur lors du chargement du modèle CNN: {str(e)}")
        
        self.is_trained = False
        return False
    
    def _save_model(self):
        """Sauvegarder le modèle et ses labels"""
        try:
            if self.model is not None:
                # Sauvegarder le modèle
                self.model.save(self.model_file)
                
                # Sauvegarder les labels
                with open(self.labels_file, 'wb') as f:
                    pickle.dump(self.face_labels, f)
                
                current_app.logger.info(f"Modèle CNN sauvegardé avec {len(self.face_labels)} personnes")
                return True
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la sauvegarde du modèle CNN: {str(e)}")
        
        return False
    
    def _preprocess_image(self, image):
        """
        Prétraiter une image pour l'entrée du modèle
        
        Args:
            image: Image numpy array au format BGR (OpenCV)
            
        Returns:
            Image prétraitée au format approprié pour le modèle
        """
        # Convertir BGR en RGB si nécessaire
        if image.shape[2] == 3 and image[0,0,0] != image[0,0,2]:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner à la taille attendue par le modèle
        image = cv2.resize(image, (224, 224))
        
        # Normaliser les valeurs des pixels
        image = image.astype(np.float32) / 255.0
        
        # Prétraitement spécifique à MobileNetV2
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        
        # Ajouter la dimension du lot (batch)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def train(self, known_faces_folder, epochs=10, batch_size=32):
        """
        Entraîner le modèle CNN sur les visages connus
        
        Args:
            known_faces_folder: Dossier contenant les images de visages connus
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des lots
            
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        try:
            # Vérifier si des visages existent
            if not os.path.exists(known_faces_folder) or len(os.listdir(known_faces_folder)) == 0:
                current_app.logger.warning("Aucun visage trouvé pour l'entraînement")
                return False
            
            # Préparer les données d'entraînement
            # Supposons que le dossier known_faces contient un fichier par personne
            # Nous créons un dossier temporaire avec la structure appropriée pour l'entraînement
            
            train_dir = os.path.join(self.model_path, 'train_data')
            os.makedirs(train_dir, exist_ok=True)
            
            # Nettoyer les anciennes données
            for person_dir in os.listdir(train_dir):
                person_path = os.path.join(train_dir, person_dir)
                if os.path.isdir(person_path):
                    for file in os.listdir(person_path):
                        os.remove(os.path.join(person_path, file))
                    os.rmdir(person_path)
            
            # Scanner le dossier des visages connus
            self.face_labels = []
            for filename in os.listdir(known_faces_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extraire le nom de la personne du nom de fichier
                    person_name = os.path.splitext(filename)[0]
                    
                    if person_name not in self.face_labels:
                        self.face_labels.append(person_name)
                        
                        # Créer un dossier pour cette personne
                        person_dir = os.path.join(train_dir, person_name)
                        os.makedirs(person_dir, exist_ok=True)
                        
                        # Copier et augmenter l'image pour l'entraînement
                        image_path = os.path.join(known_faces_folder, filename)
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Enregistrer l'image originale
                            cv2.imwrite(os.path.join(person_dir, 'original.jpg'), image)
                            
                            # Générer des versions augmentées
                            for i in range(10):  # Créer 10 variations
                                # Rotation aléatoire
                                angle = np.random.uniform(-15, 15)
                                rows, cols = image.shape[:2]
                                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                                rotated = cv2.warpAffine(image, M, (cols, rows))
                                
                                # Ajustement aléatoire de luminosité
                                brightness = np.random.uniform(0.7, 1.3)
                                adjusted = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)
                                
                                # Enregistrer l'image augmentée
                                cv2.imwrite(os.path.join(person_dir, f'aug_{i}.jpg'), adjusted)
            
            # S'il n'y a pas assez de personnes, impossible d'entraîner
            if len(self.face_labels) < 2:
                current_app.logger.warning("Il faut au moins 2 personnes pour entraîner le modèle CNN")
                return False
            
            # Créer le générateur de données avec augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2  # 20% des données pour la validation
            )
            
            # Générateur pour l'ensemble d'entraînement
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            
            # Générateur pour l'ensemble de validation
            validation_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
            
            # Créer le modèle avec le bon nombre de classes
            self.model = self._create_model(len(self.face_labels))
            
            # Entraîner le modèle
            self.model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // batch_size,
                epochs=epochs
            )
            
            # Sauvegarder le modèle
            self._save_model()
            
            self.is_trained = True
            current_app.logger.info(f"Modèle CNN entraîné avec succès sur {len(self.face_labels)} personnes")
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Erreur lors de l'entraînement du modèle CNN: {str(e)}")
            return False
    
    def predict(self, face_image):
        """
        Prédire l'identité d'un visage
        
        Args:
            face_image: Image du visage (numpy array)
            
        Returns:
            Tuple (nom prédit, score de confiance) ou (None, 0) si pas de prédiction
        """
        if not self.is_trained or self.model is None:
            return None, 0
        
        try:
            # Prétraiter l'image
            processed_image = self._preprocess_image(face_image)
            
            # Faire la prédiction
            predictions = self.model.predict(processed_image)[0]
            
            # Trouver la classe avec la plus haute probabilité
            best_index = np.argmax(predictions)
            confidence = predictions[best_index]
            
            # Vérifier si la confiance est suffisante
            threshold = 0.6
            if confidence >= threshold:
                return self.face_labels[best_index], float(confidence)
            else:
                return "Inconnu", float(confidence)
            
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return "Erreur", 0