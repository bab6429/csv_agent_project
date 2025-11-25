"""
Configuration pour l'agent CSV
"""
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class Config:
    """Configuration globale de l'application"""
    
    # Clé API Google Gemini (optionnel si Ollama est utilisé)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # Paramètres du modèle Gemini (utilisé en fallback si Ollama n'est pas disponible)
    MODEL_NAME = "gemini-2.0-flash"
    TEMPERATURE = 0  # 0 = déterministe, 1 = créatif
    
    # Paramètres Ollama (utilisé en priorité si disponible)
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")  # Modèle Ollama par défaut
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # URL d'Ollama
    
    # Paramètres de l'agent (limites désactivées)
    MAX_ITERATIONS = None  # Nombre max d'itérations de l'agent (None = pas de limite)
    MAX_ITERATIONS_GEMINI = 15  # Nombre max d'itérations quand on utilise Gemini (limite pour éviter les coûts)
    MAX_EXECUTION_TIME = None  # Timeout en secondes (None = pas de limite)
    VERBOSE = False  # Afficher les étapes de raisonnement
    
    # Délai entre les appels LLM pour éviter les erreurs 429 (resource exhausted)
    # Note: Ce délai s'applique principalement à Gemini, Ollama local est généralement plus tolérant
    LLM_REQUEST_DELAY = 1.5  # Délai en secondes entre chaque appel à l'API
    

