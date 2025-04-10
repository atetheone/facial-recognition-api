from flask import current_app

def allowed_file(filename):
    """
    Vérifier si l'extension du fichier est autorisée
    
    Args:
        filename: Nom du fichier à vérifier
        
    Returns:
        Boolean: True si l'extension est autorisée, False sinon
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']