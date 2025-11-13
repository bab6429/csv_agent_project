"""
Configuration pour l'agent CSV
"""
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class Config:
    """Configuration globale de l'application"""
    
    # Clé API Google Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # Paramètres du modèle Gemini
    MODEL_NAME = "gemini-2.0-flash"
    TEMPERATURE = 0  # 0 = déterministe, 1 = créatif
    
    # Paramètres de l'agent (réduits pour éviter l'épuisement des ressources)
    MAX_ITERATIONS = 5  # Nombre max d'itérations de l'agent (réduit de 10 à 5)
    MAX_EXECUTION_TIME = 30  # Timeout en secondes (réduit de 60 à 30)
    VERBOSE = False  # Afficher les étapes de raisonnement
    
    # Délai entre les appels LLM pour éviter les erreurs 429 (resource exhausted)
    LLM_REQUEST_DELAY = 1.5  # Délai en secondes entre chaque appel à l'API Gemini
    
    # Paramètres d'affichage
    MAX_ROWS_DISPLAY = 100  # Nombre max de lignes à afficher
    FLOAT_PRECISION = 2  # Précision des nombres décimaux
    
    # Langues
    LANGUAGE = "fr"  # Langue des réponses
    
    @classmethod
    def validate(cls):
        """Valide la configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY manquante. "
                "Créez un fichier .env avec votre clé API."
            )
        return True
    
    @classmethod
    def display(cls):
        """Affiche la configuration actuelle"""
        print("⚙️  Configuration actuelle :")
        print(f"   - Modèle : {cls.MODEL_NAME}")
        print(f"   - Temperature : {cls.TEMPERATURE}")
        print(f"   - Max iterations : {cls.MAX_ITERATIONS}")
        print(f"   - Timeout : {cls.MAX_EXECUTION_TIME}s")
        print(f"   - Délai entre appels LLM : {cls.LLM_REQUEST_DELAY}s")
        print(f"   - Verbose : {cls.VERBOSE}")
        print(f"   - Langue : {cls.LANGUAGE}")
        if cls.GOOGLE_API_KEY:
            key_preview = cls.GOOGLE_API_KEY[:10] + "..." + cls.GOOGLE_API_KEY[-4:]
            print(f"   - API Key : {key_preview}")
        else:
            print(f"   - API Key : ❌ Non configurée")

