import os
from app import create_app

# Déterminer l'environnement
env = os.environ.get('FLASK_ENV', 'dev')

# Créer l'application
app = create_app()

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    print(f"=== API de Reconnaissance Faciale ===")
    print(f"Environnement: {env}")
    print(f"Démarrage sur {host}:{port}")
    
    app.run(host=host, port=port, debug=(env != 'prod'))