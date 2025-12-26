"""
Factory pour créer des instances LLM avec support Ollama et fallback Gemini
"""
import os
import requests
from typing import Optional
from langchain_core.language_models import BaseChatModel
from config import Config


def _test_ollama_connection(base_url: str = None, timeout: int = 2) -> bool:
    """
    Teste si Ollama est disponible et accessible
    
    Args:
        base_url: URL de base d'Ollama (par défaut http://localhost:11434)
        timeout: Timeout en secondes pour le test de connexion
        
    Returns:
        True si Ollama est accessible, False sinon
    """
    if base_url is None:
        base_url = Config.OLLAMA_BASE_URL
    
    try:
        # Test simple de connexion à l'API Ollama
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return False


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: int = 2,
    api_key: Optional[str] = None,
    verbose: bool = True
) -> BaseChatModel:
    """
    Crée une instance LLM en essayant Ollama en premier, puis fallback vers Gemini
    
    Args:
        model_name: Nom du modèle (si None, utilise Config.MODEL_NAME)
        temperature: Température du modèle (si None, utilise Config.TEMPERATURE)
        max_output_tokens: Nombre max de tokens de sortie (optionnel)
        max_retries: Nombre de tentatives en cas d'erreur
        api_key: Clé API Google (optionnel, pour Gemini)
        verbose: Si True, affiche des messages informatifs
        
    Returns:
        Instance de BaseChatModel (ChatOllama ou ChatGoogleGenerativeAI)
        
    Raises:
        ValueError: Si ni Ollama ni Gemini ne sont disponibles
    """
    # Utiliser les valeurs par défaut de la config si non spécifiées
    if model_name is None:
        model_name = Config.MODEL_NAME
    if temperature is None:
        temperature = Config.TEMPERATURE
    
    # Vérifier si on force l'utilisation d'un provider spécifique
    use_ollama = os.getenv("USE_OLLAMA", "").lower()
    use_gemini = os.getenv("USE_GEMINI", "").lower()
    
    # Si USE_OLLAMA est explicitement défini à "true", utiliser Ollama
    if use_ollama == "true":
        if verbose:
            print("🔧 Mode Ollama forcé via USE_OLLAMA=true")
        return _create_ollama_llm(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            verbose=verbose
        )
    
    # Si USE_GEMINI est explicitement défini à "true", utiliser Gemini
    if use_gemini == "true":
        if verbose:
            print("🔧 Mode Gemini forcé via USE_GEMINI=true")
        return _create_gemini_llm(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            api_key=api_key,
            verbose=verbose
        )
    
    # Sinon, essayer Ollama en premier (par défaut)
    if verbose:
        print("🔍 Vérification de la disponibilité d'Ollama...")
    
    if _test_ollama_connection():
        if verbose:
            print(f"✅ Ollama détecté ! Utilisation du modèle local: {Config.OLLAMA_MODEL_NAME}")
        return _create_ollama_llm(
            model_name=Config.OLLAMA_MODEL_NAME,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            verbose=verbose
        )
    else:
        # Fallback vers Gemini
        if verbose:
            print("⚠️ Ollama non disponible, utilisation de Gemini en fallback...")
        return _create_gemini_llm(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            api_key=api_key,
            verbose=verbose
        )


def _create_ollama_llm(
    model_name: str,
    temperature: float,
    max_output_tokens: Optional[int],
    max_retries: int,
    verbose: bool
) -> BaseChatModel:
    """Crée une instance ChatOllama"""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama n'est pas installé. "
            "Installez-le avec: pip install langchain-ollama"
        )
    
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "base_url": Config.OLLAMA_BASE_URL,
    }
    
    # max_output_tokens n'est pas supporté par ChatOllama de la même manière
    # On peut utiliser num_predict à la place si nécessaire
    if max_output_tokens:
        llm_kwargs["num_predict"] = max_output_tokens
    
    if verbose:
        print(f"🤖 Initialisation de ChatOllama avec le modèle: {model_name}")
    
    return ChatOllama(**llm_kwargs)


def _create_gemini_llm(
    model_name: str,
    temperature: float,
    max_output_tokens: Optional[int],
    max_retries: int,
    api_key: Optional[str],
    verbose: bool
) -> BaseChatModel:
    """Crée une instance ChatGoogleGenerativeAI"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai n'est pas installé. "
            "Installez-le avec: pip install langchain-google-genai"
        )
    
    # Configuration de la clé API
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" not in os.environ:
        raise ValueError(
            "Clé API Google manquante pour Gemini. "
            "Définissez-la via GOOGLE_API_KEY dans .env ou passez-la en paramètre. "
            "Ou installez et démarrez Ollama pour utiliser un LLM local."
        )
    
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_retries": max_retries,
    }
    
    if max_output_tokens:
        llm_kwargs["max_output_tokens"] = max_output_tokens
    
    if verbose:
        print(f"🤖 Initialisation de ChatGoogleGenerativeAI avec le modèle: {model_name}")
    
    return ChatGoogleGenerativeAI(**llm_kwargs)



