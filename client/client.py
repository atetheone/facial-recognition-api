#!/usr/bin/env python3
"""
Client de test pour l'API de reconnaissance faciale.
Permet d'interagir facilement avec l'API via la ligne de commande.
"""

import requests
import argparse
import os
import json
import sys
from tabulate import tabulate

API_URL = "http://localhost:5000"

def register_face(image_path, name, verbose=True):
    """
    Enregistrer un nouveau visage dans la base de données
    
    Args:
        image_path: Chemin vers l'image contenant le visage
        name: Nom de la personne
        verbose: Afficher les détails de la réponse
    
    Returns:
        bool: Succès ou échec
    """
    if not os.path.exists(image_path):
        print(f"Erreur: Le fichier {image_path} n'existe pas")
        return False
    
    url = f"{API_URL}/register_face"
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
            data = {'name': name}
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                if verbose:
                    if result.get('success'):
                        print(f"✅ {result.get('message')}")
                    else:
                        print(f"❌ {result.get('error')}")
                return result.get('success', False)
            else:
                if verbose:
                    print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
                return False
    except Exception as e:
        if verbose:
            print(f"❌ Erreur lors de la connexion à l'API: {str(e)}")
        return False

def recognize_face(image_path, method=None, use_cnn_model=False, verbose=True):
    """
    Reconnaître les visages dans une image
    
    Args:
        image_path: Chemin vers l'image à analyser
        method: Méthode d'extraction de caractéristiques ('hog', 'cnn', 'custom_hog', 'lbp')
        use_cnn_model: Utiliser le modèle CNN personnalisé
        verbose: Afficher les détails de la réponse
    
    Returns:
        dict: Résultats de la reconnaissance ou None
    """
    if not os.path.exists(image_path):
        print(f"Erreur: Le fichier {image_path} n'existe pas")
        return None
    
    url = f"{API_URL}/recognize"
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
            data = {}
            
            if method:
                data['method'] = method
            
            if use_cnn_model:
                data['use_cnn_model'] = 'true'
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if verbose:
                    if result.get('success'):
                        print(f"Analyse de l'image: {os.path.basename(image_path)}")
                        print(f"Méthode: {method or 'hog'}{' (Modèle CNN)' if use_cnn_model else ''}")
                        print(f"Visages détectés: {result.get('faces_detected', 0)}\n")
                        
                        if result.get('faces_detected', 0) > 0:
                            # Préparer les données pour le tableau
                            table_data = []
                            for face in result.get('results', []):
                                confidence = face.get('confidence', 0) * 100
                                location = face.get('location', {})
                                coords = f"({location.get('left')},{location.get('top')})-({location.get('right')},{location.get('bottom')})"
                                table_data.append([
                                    face.get('id', ''),
                                    face.get('name', 'Inconnu'),
                                    f"{confidence:.2f}%",
                                    coords
                                ])
                            
                            # Afficher les résultats dans un tableau
                            print(tabulate(
                                table_data, 
                                headers=['ID', 'Personne', 'Confiance', 'Position'], 
                                tablefmt='pretty'
                            ))
                            
                            if result.get('output_image'):
                                print(f"\nImage annotée disponible à: {API_URL}/get_image/{result.get('output_image')}")
                        else:
                            print("Aucun visage détecté dans l'image.")
                    else:
                        print(f"❌ {result.get('error')}")
                
                return result
            else:
                if verbose:
                    print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
                return None
    except Exception as e:
        if verbose:
            print(f"❌ Erreur lors de la connexion à l'API: {str(e)}")
        return None

def list_known_faces(verbose=True):
    """
    Lister tous les visages connus
    
    Args:
        verbose: Afficher les détails de la réponse
    
    Returns:
        list: Liste des visages connus ou None
    """
    url = f"{API_URL}/list_known_faces"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            
            if verbose:
                if result.get('success'):
                    faces = result.get('known_faces', [])
                    count = result.get('count', 0)
                    
                    if count > 0:
                        print(f"Visages enregistrés ({count}):")
                        for i, face in enumerate(faces, 1):
                            print(f"  {i}. {face}")
                    else:
                        print("Aucun visage enregistré.")
                else:
                    print(f"❌ {result.get('error')}")
            
            return result.get('known_faces', [])
        else:
            if verbose:
                print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
            return None
    except Exception as e:
        if verbose:
            print(f"❌ Erreur lors de la connexion à l'API: {str(e)}")
        return None

def delete_face(name, verbose=True):
    """
    Supprimer un visage de la base de données
    
    Args:
        name: Nom de la personne à supprimer
        verbose: Afficher les détails de la réponse
    
    Returns:
        bool: Succès ou échec
    """
    url = f"{API_URL}/delete_face/{name}"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            
            if verbose:
                if result.get('success'):
                    print(f"✅ {result.get('message')}")
                else:
                    print(f"❌ {result.get('error')}")
            
            return result.get('success', False)
        else:
            if verbose:
                print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        if verbose:
            print(f"❌ Erreur lors de la connexion à l'API: {str(e)}")
        return False

def check_api_status(verbose=True):
    """
    Vérifier si l'API est en ligne
    
    Args:
        verbose: Afficher les détails de la réponse
    
    Returns:
        bool: API en ligne ou non
    """
    try:
        response = requests.get(API_URL)
        
        if response.status_code == 200:
            if verbose:
                result = response.json()
                print(f"✅ API en ligne: {result.get('message')} (Status: {result.get('status')})")
            return True
        else:
            if verbose:
                print(f"❌ L'API a répondu avec le code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        if verbose:
            print(f"❌ Impossible de se connecter à l'API à {API_URL}")
            print("   Assurez-vous que l'API est en cours d'exécution.")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Erreur lors de la vérification de l'API: {str(e)}")
        return False

def main():
    """Fonction principale du client"""
    parser = argparse.ArgumentParser(description='Client pour l\'API de reconnaissance faciale')
    
    # Commande principale
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Commande pour vérifier le statut de l'API
    subparsers.add_parser('status', help='Vérifier si l\'API est en ligne')
    
    # Commande pour enregistrer un visage
    register_parser = subparsers.add_parser('register', help='Enregistrer un nouveau visage')
    register_parser.add_argument('image', help='Chemin vers l\'image contenant le visage')
    register_parser.add_argument('name', help='Nom de la personne')
    
    # Commande pour reconnaître des visages
    recognize_parser = subparsers.add_parser('recognize', help='Reconnaître les visages dans une image')
    recognize_parser.add_argument('image', help='Chemin vers l\'image à analyser')
    recognize_parser.add_argument('--method', choices=['hog', 'cnn', 'custom_hog', 'lbp'], 
                                help='Méthode d\'extraction de caractéristiques')
    recognize_parser.add_argument('--use-cnn-model', action='store_true', 
                                help='Utiliser le modèle CNN personnalisé')
    
    # Commande pour lister les visages
    subparsers.add_parser('list', help='Lister les visages enregistrés')
    
    # Commande pour supprimer un visage
    delete_parser = subparsers.add_parser('delete', help='Supprimer un visage')
    delete_parser.add_argument('name', help='Nom de la personne à supprimer')
    
    # Commande pour définir l'URL de l'API
    url_parser = subparsers.add_parser('set-url', help='Définir l\'URL de l\'API')
    url_parser.add_argument('url', help='URL de l\'API (ex: http://localhost:5000)')
    
    args = parser.parse_args()
    
    # Si aucune commande n'est fournie, afficher l'aide
    if not args.command:
        parser.print_help()
        return
    
    # Vérifier la connexion à l'API (sauf pour set-url)
    if args.command != 'set-url' and args.command != 'status':
        if not check_api_status(verbose=False):
            print(f"❌ Impossible de se connecter à l'API à {API_URL}")
            print("   Assurez-vous que l'API est en cours d'exécution ou utilisez 'set-url' pour changer l'URL.")
            return
    
    # Exécuter la commande appropriée
    if args.command == 'status':
        check_api_status()
    elif args.command == 'register':
        register_face(args.image, args.name)
    elif args.command == 'recognize':
        recognize_face(args.image, args.method, args.use_cnn_model)
    elif args.command == 'list':
        list_known_faces()
    elif args.command == 'delete':
        delete_face(args.name)
    elif args.command == 'set-url':
        # Sauvegarder l'URL dans un fichier de configuration
        config_dir = os.path.expanduser('~/.config/facial-recognition-client')
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, 'config.json')
        
        with open(config_file, 'w') as f:
            json.dump({'api_url': args.url}, f)
        
        print(f"✅ URL de l'API définie à: {args.url}")
        
        # Vérifier la connexion avec la nouvelle URL
        # global API_URL
        API_URL = args.url
        check_api_status()

if __name__ == "__main__":
    # Charger l'URL de l'API depuis le fichier de configuration s'il existe
    config_file = os.path.expanduser('~/.config/facial-recognition-client/config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'api_url' in config:
                    API_URL = config['api_url']
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la configuration: {str(e)}")
    
    main()