"""
Agent orchestrateur qui route les questions vers les agents spécialisés
Utilise un LLM pour un routing intelligent
"""
import os
import time
from typing import Optional
from csv_tools import CSVTools
from .time_series_agent import TimeSeriesAgent
from .transformation_agent import TransformationAgent
from config import Config
from llm_factory import get_llm


class OrchestratorAgent:
    """
    Agent principal qui orchestre les agents spécialisés
    Route les questions vers l'agent approprié selon le type de question
    Utilise un LLM pour un routing intelligent et contextuel
    """
    
    def __init__(self, csv_path: str, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialise l'agent orchestrateur
        
        Args:
            csv_path: Chemin vers le fichier CSV
            api_key: Clé API Google (optionnel)
            verbose: Si True, affiche les étapes de raisonnement
        """
        self.csv_path = csv_path
        self.verbose = verbose
        self.last_llm_call_time = 0
        
        # Initialisation du LLM pour le routing (léger, rapide)
        # Utilise Ollama si disponible, sinon fallback vers Gemini
        print("🤖 Initialisation du LLM de routing...")
        try:
            self.routing_llm = get_llm(
                model_name=Config.MODEL_NAME,
                temperature=0,  # Déterministe pour le routing
                max_output_tokens=50,  # Très court, juste pour choisir l'agent
                max_retries=2,
                api_key=api_key,
                verbose=verbose
            )
        except ValueError as e:
            # Si ni Ollama ni Gemini ne sont disponibles, on essaie quand même
            # avec Gemini en forçant la clé API
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            raise ValueError(
                f"Impossible d'initialiser un LLM. {str(e)}\n"
                "Solutions:\n"
                "1. Installez et démarrez Ollama (recommandé pour usage local)\n"
                "2. Définissez GOOGLE_API_KEY dans .env ou passez-la en paramètre"
            )
        
        # Initialisation des outils CSV (partagés par tous les agents)
        print("🔧 Initialisation des outils d'analyse...")
        self.csv_tools = CSVTools(csv_path)
        
        # Initialisation des agents spécialisés
        print("🤖 Initialisation des agents spécialisés...")
        self.time_series_agent = TimeSeriesAgent(
            csv_tools=self.csv_tools,
            api_key=api_key,
            verbose=verbose
        )
        
        self.transformation_agent = TransformationAgent(
            csv_tools=self.csv_tools,
            api_key=api_key,
            verbose=verbose
        )
        
        print("✅ Orchestrateur prêt !\n")
    
    def _detect_agent_type(self, question: str) -> str:
        """
        Détecte quel agent spécialisé doit traiter la question en utilisant un LLM
        
        Args:
            question: La question de l'utilisateur
            
        Returns:
            'time_series' ou 'transformation'
        """
        # Gestion du délai entre appels LLM
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        if time_since_last_call < Config.LLM_REQUEST_DELAY:
            delay_needed = Config.LLM_REQUEST_DELAY - time_since_last_call
            time.sleep(delay_needed)
        
        self.last_llm_call_time = time.time()
        
        # Prompt pour le LLM de routing
        routing_prompt = f"""Tu es un routeur intelligent. Analyse cette question et détermine quel agent spécialisé doit la traiter.

Question: "{question}"

Agents disponibles:
1. time_series - Pour les questions sur:
   - Tendances, croissance, décroissance
   - Séries temporelles, données temporelles
   - Moyennes mobiles, lissage
   - Agrégations par période (jour, semaine, mois, année)
   - Saisonnalité, patterns temporels
   - Prévisions, forecasts
   - Taux de croissance temporels
   - Anomalies dans des séries temporelles

2. transformation - Pour les questions sur:
   - Structure du fichier, colonnes, types de données
   - Aperçu des données (premières lignes)
   - Statistiques descriptives (moyenne, médiane, etc.)
   - Valeurs manquantes, qualité des données
   - Corrélations entre colonnes
   - Filtrage, groupement de données
   - Manipulation et transformation de données

Réponds UNIQUEMENT par un seul mot: "time_series" ou "transformation"
Ne réponds rien d'autre, juste le nom de l'agent."""
        
        try:
            response = self.routing_llm.invoke(routing_prompt)
            agent_type = response.content.strip().lower()
            
            # Validation et normalisation
            if 'time_series' in agent_type or 'timeseries' in agent_type:
                return 'time_series'
            elif 'transformation' in agent_type:
                return 'transformation'
            else:
                # Fallback: utiliser transformation par défaut
                if self.verbose:
                    print(f"⚠️ Réponse LLM non reconnue: '{agent_type}', utilisation de 'transformation' par défaut")
                return 'transformation'
                
        except Exception as e:
            # En cas d'erreur, fallback vers transformation
            if self.verbose:
                print(f"⚠️ Erreur lors du routing LLM: {e}, utilisation de 'transformation' par défaut")
            return 'transformation'
    
    def query(self, question: str) -> str:
        """
        Traite une question en la routant vers l'agent approprié
        
        Args:
            question: La question de l'utilisateur
            
        Returns:
            La réponse de l'agent spécialisé
        """
        # Détecter quel agent doit traiter la question
        agent_type = self._detect_agent_type(question)
        
        if self.verbose:
            print(f"🔀 Routing vers l'agent: {agent_type}")
        
        # Router vers l'agent approprié
        if agent_type == 'time_series':
            return self.time_series_agent.query(question)
        elif agent_type == 'transformation':
            return self.transformation_agent.query(question)
        else:
            # Par défaut, utiliser l'agent transformation
            return self.transformation_agent.query(question)
    
    def get_dataframe(self):
        """Retourne le DataFrame pandas pour un accès direct si nécessaire"""
        return self.csv_tools.df

