"""
Agent spécialisé dans l'analyse de séries temporelles
"""
import os
import time
from typing import Optional
from callbacks import LLMIterationCounter
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from csv_tools import CSVTools
from config import Config
from llm_factory import get_llm


class TimeSeriesAgent:
    """
    Agent spécialisé dans l'analyse de séries temporelles
    """
    
    def __init__(self, csv_tools: CSVTools, api_key: Optional[str] = None, verbose: bool = True, llm_counter: Optional[dict] = None):
        """
        Initialise l'agent Time Series
        
        Args:
            csv_tools: Instance de CSVTools avec le DataFrame chargé
            api_key: Clé API Google (optionnel)
            verbose: Si True, affiche les étapes de raisonnement
        """
        self.csv_tools = csv_tools
        self.verbose = verbose
        self.last_llm_call_time = 0
        self.callbacks = [LLMIterationCounter(llm_counter)] if llm_counter is not None else None
        
        # Initialisation du LLM (Ollama en priorité, fallback Gemini)
        self.llm = get_llm(
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_output_tokens=2048,
            max_retries=2,
            api_key=api_key,
            verbose=verbose
        )
        
        # Outils spécialisés pour les séries temporelles
        self.tools = self._create_tools()
        
        # Création du prompt spécialisé
        self.prompt = self._create_prompt()
        
        # Création de l'agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Déterminer la limite d'itérations selon le provider utilisé
        max_iterations = self._get_max_iterations()
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            max_execution_time=Config.MAX_EXECUTION_TIME,
            return_intermediate_steps=True,
            early_stopping_method="force",
            callbacks=self.callbacks,
        )
    
    def _get_max_iterations(self) -> Optional[int]:
        """
        Détermine la limite d'itérations selon le provider LLM utilisé
        Applique une limite spécifique pour Gemini
        """
        # Vérifier si on utilise Gemini en vérifiant le type de LLM
        llm_type = type(self.llm).__name__
        if "GoogleGenerativeAI" in llm_type or "Gemini" in llm_type:
            if self.verbose:
                print(f"🔒 Limite d'itérations Gemini activée: {Config.MAX_ITERATIONS_GEMINI}")
            return Config.MAX_ITERATIONS_GEMINI
        else:
            # Pour Ollama ou autres, utiliser la limite générale
            return Config.MAX_ITERATIONS
    
    def _create_tools(self) -> list:
        """Crée les outils spécialisés pour les séries temporelles"""
        return [
            Tool(
                name="get_csv_info",
                func=self.csv_tools.get_info,
                description="Obtient les informations générales sur le fichier CSV (colonnes, types, dimensions). Utilise cet outil pour découvrir les colonnes disponibles dans le dataset. Input: vide."
            ),
            Tool(
                name="detect_time_columns",
                func=self.csv_tools.detect_time_columns,
                description="Détecte automatiquement les colonnes contenant des dates/timestamps. Input: vide."
            ),
            Tool(
                name="calculate_trend",
                func=self.csv_tools.calculate_trend,
                description="Calcule la tendance (croissance/décroissance) d'une série temporelle. Input: 'nom_colonne' ou 'nom_colonne,colonne_temps'. Utilise le nom RÉEL de la colonne du dataset. Exemple: 'Global_active_power' ou 'Global_active_power,Date'."
            ),
            Tool(
                name="calculate_moving_average",
                func=self.csv_tools.calculate_moving_average,
                description="Calcule la moyenne mobile pour lisser les données. Input: 'nom_colonne,fenetre' ou 'nom_colonne,fenetre,colonne_temps'. Utilise le nom RÉEL de la colonne du dataset. Exemple: 'Voltage,7' pour une moyenne mobile sur 7 périodes."
            ),
            Tool(
                name="aggregate_by_period",
                func=self.csv_tools.aggregate_by_period,
                description="Agrège les données par période (jour=D, semaine=W, mois=M, trimestre=Q, année=Y). Input: 'nom_colonne,periode,colonne_temps,fonction_agreg'. Utilise les noms RÉELS des colonnes du dataset. Exemple: 'Global_intensity,M,Date,sum' pour sommer par mois."
            ),
        ]
    
    def _create_prompt(self):
        """Crée le prompt spécialisé pour les séries temporelles"""
        template = """Tu es un expert en analyse de séries temporelles. 
Tu aides l'utilisateur à analyser des données temporelles en répondant à ses questions.

Tu as accès aux outils suivants pour analyser les séries temporelles :

{tools}

Utilise le format suivant :

Question: la question de l'utilisateur
Thought: réfléchis BRIÈVEMENT (1 phrase max)
Action: l'action à prendre parmi [{tool_names}]
Action Input: l'entrée de l'action
Observation: le résultat
... (maximum 3-4 actions)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale

⚠️ IMPORTANT : 
- Sois EFFICACE. Maximum 3-4 actions avant la réponse finale.
- INCLUS TOUJOURS les résultats numériques complets dans ta réponse finale (tendances, moyennes mobiles, agrégations, etc.)
- Ne résume PAS les résultats des outils, copie-les tels quels dans ta réponse
- Si la question demande des tendances temporelles sans spécifier de colonne, détecte d'abord les colonnes temporelles et numériques disponibles, puis analyse les tendances pour toutes les colonnes numériques pertinentes
- N'INVENTE JAMAIS de noms de colonnes. Utilise uniquement les colonnes RÉELLES du dataset

RÈGLES IMPORTANTES :
1. Réponds TOUJOURS en français
2. Sois PRÉCIS et CONCIS dans tes explications, MAIS inclut TOUJOURS les résultats bruts des outils
3. Pour les calculs, les tendances, les agrégations : copie TOUJOURS les valeurs numériques complètes de l'Observation dans ta réponse finale
4. ADAPTE-TOI au dataset : utilise les noms de colonnes RÉELS du fichier, pas des exemples génériques
5. Si tu ne connais pas les colonnes disponibles, utilise d'abord 'get_csv_info' pour découvrir toutes les colonnes, puis 'detect_time_columns' pour les colonnes temporelles
6. Si la question mentionne des tendances temporelles sans spécifier de colonne, détecte d'abord les colonnes disponibles avec 'get_csv_info' et 'detect_time_columns', puis analyse les tendances pour toutes les colonnes numériques pertinentes
7. Pour les tendances, utilise 'calculate_trend' avec le nom RÉEL de la colonne - INCLUS les valeurs dans ta réponse
8. Pour lisser les données, utilise 'calculate_moving_average' avec le nom RÉEL de la colonne - INCLUS les valeurs dans ta réponse
9. Pour agréger par période, utilise 'aggregate_by_period' avec les noms RÉELS des colonnes - INCLUS les valeurs dans ta réponse

STRATÉGIE D'ANALYSE :
- Étape 1 : Si tu ne connais pas la structure du dataset, utilise d'abord 'get_csv_info' pour découvrir toutes les colonnes disponibles
- Étape 2 : Utilise 'detect_time_columns' pour identifier les colonnes temporelles
- Étape 3 : Identifie les colonnes numériques pertinentes pour l'analyse demandée
- Étape 4 : Utilise les outils avec les noms de colonnes RÉELS trouvés dans le dataset
- Étape 5 : Inclus TOUJOURS les résultats complets dans ta réponse finale

Exemples de format (remplace par les noms RÉELS de colonnes) :

Pour découvrir les colonnes disponibles :
Action: get_csv_info
Action Input: 

Pour détecter les colonnes temporelles :
Action: detect_time_columns
Action Input: 

Pour calculer la tendance d'une colonne numérique (ex: Global_active_power) :
Action: calculate_trend
Action Input: Global_active_power

Pour une moyenne mobile (ex: sur 7 périodes pour Voltage) :
Action: calculate_moving_average
Action Input: Voltage,7

Pour agréger par mois (ex: Global_intensity par mois) :
Action: aggregate_by_period
Action Input: Global_intensity,M,date,sum

Commence maintenant !

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        )
    
    def query(self, question: str) -> str:
        """Pose une question à l'agent Time Series"""
        # Gestion du délai entre appels
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        if time_since_last_call < Config.LLM_REQUEST_DELAY:
            delay_needed = Config.LLM_REQUEST_DELAY - time_since_last_call
            time.sleep(delay_needed)
        
        self.last_llm_call_time = time.time()
        
        # Retry avec backoff exponentiel
        max_retries = 3
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = self.agent_executor.invoke({"input": question})
                return response.get("output", "")
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str or "resource exhausted" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if self.verbose:
                            print(f"⚠️ Erreur 429. Nouvelle tentative dans {delay}s...")
                        time.sleep(delay)
                        self.last_llm_call_time = time.time()
                        continue
                    else:
                        return f"❌ Erreur : Limite de taux API atteinte. Veuillez patienter."
                else:
                    return f"❌ Erreur : {error_str}"
        
        return "❌ Erreur : Échec après plusieurs tentatives"

