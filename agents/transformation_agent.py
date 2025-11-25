"""
Agent spécialisé dans la transformation et le filtrage de données
"""
import os
import time
from typing import Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from csv_tools import CSVTools
from config import Config
from llm_factory import get_llm


class TransformationAgent:
    """
    Agent spécialisé dans la transformation, filtrage et manipulation de données
    """
    
    def __init__(self, csv_tools: CSVTools, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialise l'agent Transformation
        
        Args:
            csv_tools: Instance de CSVTools avec le DataFrame chargé
            api_key: Clé API Google (optionnel)
            verbose: Si True, affiche les étapes de raisonnement
        """
        self.csv_tools = csv_tools
        self.verbose = verbose
        self.last_llm_call_time = 0
        
        # Initialisation du LLM (Ollama en priorité, fallback Gemini)
        self.llm = get_llm(
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_output_tokens=2048,
            max_retries=2,
            api_key=api_key,
            verbose=verbose
        )
        
        # Outils spécialisés pour la transformation
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
        """Crée les outils spécialisés pour la transformation"""
        return [
            Tool(
                name="get_csv_info",
                func=self.csv_tools.get_info,
                description="Obtient les informations générales sur le fichier CSV (colonnes, types, dimensions). Input: vide."
            ),
            Tool(
                name="get_head",
                func=self.csv_tools.get_head,
                description="Affiche les n premières lignes. Input: nombre de lignes (ex: '10')."
            ),
            Tool(
                name="get_statistics",
                func=self.csv_tools.get_statistics,
                description="Calcule les statistiques descriptives. Input: nom de colonne (ou vide pour toutes)."
            ),
            Tool(
                name="count_missing_values",
                func=self.csv_tools.count_missing,
                description="Compte les valeurs manquantes dans chaque colonne. Input: vide."
            ),
        ]
    
    def _create_prompt(self):
        """Crée le prompt spécialisé pour la transformation"""
        template = """Tu es un expert en transformation et manipulation de données. 
Tu aides l'utilisateur à filtrer, transformer et analyser des données CSV.

Tu as accès aux outils suivants :

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
- INCLUS TOUJOURS les résultats numériques complets dans ta réponse finale (statistiques, corrélations, données, etc.)
- Ne résume PAS les résultats des outils, copie-les tels quels dans ta réponse

RÈGLES IMPORTANTES :
1. Réponds TOUJOURS en français
2. Sois PRÉCIS et CONCIS dans tes explications, MAIS inclut TOUJOURS les résultats bruts des outils
3. Pour les statistiques, les corrélations, les données : copie TOUJOURS les valeurs numériques complètes de l'Observation dans ta réponse finale
4. ADAPTE-TOI au dataset : utilise les noms de colonnes RÉELS du fichier, pas des exemples génériques
5. Si tu ne connais pas les colonnes disponibles, utilise d'abord 'get_csv_info' pour découvrir la structure
6. Utilise 'get_csv_info' pour comprendre la structure si nécessaire
7. Utilise 'get_head' pour voir un aperçu des données
8. Utilise 'get_statistics' pour les statistiques descriptives - INCLUS TOUJOURS les valeurs dans ta réponse
9. Utilise 'count_missing_values' pour vérifier la qualité des données

STRATÉGIE D'ANALYSE :
- Étape 1 : Si tu ne connais pas la structure, utilise 'get_csv_info' pour découvrir les colonnes disponibles
- Étape 2 : Identifie les colonnes pertinentes pour la question posée
- Étape 3 : Utilise les outils avec les noms de colonnes RÉELS trouvés dans le dataset
- Étape 4 : Inclus TOUJOURS les résultats complets dans ta réponse finale

Exemples de format (remplace par les noms RÉELS de colonnes) :

Pour voir la structure :
Action: get_csv_info
Action Input: 

Pour voir les premières lignes :
Action: get_head
Action Input: 10

Pour les statistiques d'une colonne spécifique (ex: Global_active_power) :
Action: get_statistics
Action Input: Global_active_power

Pour les statistiques de toutes les colonnes numériques :
Action: get_statistics
Action Input: 

Pour vérifier les valeurs manquantes :
Action: count_missing_values
Action Input: 

Commence maintenant !

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        )
    
    def query(self, question: str) -> str:
        """Pose une question à l'agent Transformation"""
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

