"""
Agent IA pour l'analyse de fichiers CSV avec LangChain et Gemini
"""
import os
import time
from typing import Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from csv_tools import CSVTools
from config import Config


class CSVAgent:
    """
    Agent intelligent pour analyser des fichiers CSV et répondre à des questions en langage naturel
    
    Architecture:
    - Utilise le modèle Gemini de Google pour comprendre les questions
    - Dispose d'outils personnalisés pour manipuler et analyser les données CSV
    - Utilise le pattern ReAct (Reasoning + Acting) pour orchestrer les outils
    """
    
    def __init__(self, csv_path: str, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialise l'agent CSV
        
        Args:
            csv_path: Chemin vers le fichier CSV à analyser
            api_key: Clé API Google (optionnel, peut être définie via variable d'environnement)
            verbose: Si True, affiche les étapes de raisonnement de l'agent
        """
        # Configuration de la clé API
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "Clé API Google manquante. "
                "Définissez-la via GOOGLE_API_KEY dans .env ou passez-la en paramètre."
            )
        
        self.csv_path = csv_path
        self.verbose = verbose
        self.last_llm_call_time = 0  # Timestamp du dernier appel LLM pour gérer les délais
        
        # Initialisation des outils CSV
        print("🔧 Initialisation des outils d'analyse...")
        self.csv_tools = CSVTools(csv_path)
        self.tools = self.csv_tools.get_tools()
        print(f"🧰 Outils disponibles: {[t.name for t in self.tools]}")
        
        # Initialisation du modèle Gemini avec limite de tokens
        print("🤖 Connexion à Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_output_tokens=2048,  # Limite de tokens pour éviter l'épuisement
            max_retries=2  # Limite les tentatives en cas d'erreur
        )
        
        # Création du prompt pour l'agent ReAct
        self.prompt = self._create_prompt()
        
        # Création de l'agent
        print("⚡ Création de l'agent...")
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Création de l'exécuteur d'agent avec gestion d'erreurs
        # Limites réduites pour éviter l'épuisement des ressources
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=Config.MAX_ITERATIONS,  # Utilise la config (10 par défaut, réduit à 5)
            max_execution_time=Config.MAX_EXECUTION_TIME,  # Utilise la config (60s par défaut)
            return_intermediate_steps=True,
            early_stopping_method="force",  # Force l'arrêt si limite atteinte
        )
        
        print("✅ Agent prêt à analyser vos données !\n")
    
    def _create_prompt(self):
        """
        Crée le prompt template pour l'agent ReAct
        
        Le prompt ReAct suit le format :
        - Question: La question de l'utilisateur
        - Thought: Ce que l'agent pense faire
        - Action: L'outil à utiliser
        - Action Input: Les paramètres de l'outil
        - Observation: Le résultat de l'outil
        - ... (répète Thought/Action/Observation si nécessaire)
        - Thought: Je connais maintenant la réponse finale
        - Final Answer: La réponse à l'utilisateur
        """
        template = """Tu es un assistant IA expert en analyse de données, SPÉCIALISÉ dans l'analyse de SÉRIES TEMPORELLES (time series).
Tu aides l'utilisateur à analyser un fichier CSV en répondant à ses questions en langage naturel, avec une expertise particulière pour les données temporelles.

Tu as accès aux outils suivants pour analyser les données :

{tools}

Utilise le format suivant pour raisonner et agir :

Question: la question que l'utilisateur te pose
Thought: réfléchis BRIÈVEMENT à ce que tu dois faire (1 phrase max)
Action: l'action à prendre, doit être parmi [{tool_names}]
Action Input: l'entrée de l'action
Observation: le résultat de l'action
... (ce processus peut se répéter MAXIMUM 3-4 fois, puis tu DOIS donner la réponse finale)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale à la question originale de l'utilisateur

⚠️ IMPORTANT : Sois EFFICACE. Ne fais pas plus de 3-4 actions. Si tu as les informations nécessaires, donne la réponse finale immédiatement.

RÈGLES IMPORTANTES :
1. Réponds TOUJOURS en français
2. Sois PRÉCIS, CONCIS et DIRECT - évite les réflexions inutiles
3. LIMITE : Maximum 3-4 actions (Thought/Action/Observation) avant de donner la réponse finale
4. Si tu dois faire des calculs ou analyses complexes, utilise l'outil 'python_code_executor'
5. Pour filtrer des données, utilise 'python_code_executor' avec du code pandas
6. Utilise 'get_csv_info' UNIQUEMENT si tu as vraiment besoin de connaître la structure (évite si possible)
7. Formate bien tes réponses finales pour qu'elles soient lisibles
8. Fournis une justification/explication COURTE (2-3 points max) :
   - cite les colonnes utilisées et la méthode (ex: groupby, mean, count)
   - donne 1-2 chiffres clés (moyenne, total, top catégorie, etc.) si pertinent
   - mentionne d'éventuels filtres appliqués

🎯 EXPERTISE SÉRIES TEMPORELLES (TIME SERIES) - RÈGLES CRITIQUES :
10. TU ES UN EXPERT EN SÉRIES TEMPORELLES : Tu comprends parfaitement les concepts de temps, dates, heures, jours, semaines, mois, années
11. GESTION DES DATES ET HEURES :
    - Identifie TOUJOURS les colonnes contenant des dates/heures (même si elles sont en format texte)
    - Convertis-les en format datetime avec pd.to_datetime() pour pouvoir faire des opérations temporelles
    - Reconnais différents formats : 'YYYY-MM-DD', 'DD/MM/YYYY', 'YYYY-MM-DD HH:MM:SS', timestamps Unix, etc.
    - Extrais les composantes temporelles (jour, mois, année, heure, jour de la semaine) quand nécessaire
12. COMPRÉHENSION DU TEMPS QUI PASSE :
    - Comprends que le temps est séquentiel et continu
    - Reconnais les concepts de "jour", "semaine", "mois", "année", "heure", "minute"
    - Gère les agrégations temporelles (par jour, par semaine, par mois, par heure, etc.)
    - Comprends les comparaisons temporelles : "avant", "après", "entre", "pendant", "au cours de"
    - Identifie les tendances, saisonnalités, et patterns temporels
13. ANALYSES TEMPORELLES SPÉCIFIQUES :
    - Pour les questions sur des périodes : utilise pd.to_datetime() puis filtrage avec .between() ou comparaisons
    - Pour les agrégations temporelles : utilise .resample() ou groupby avec pd.Grouper(freq='D', 'H', 'M', etc.)
    - Pour les calculs de durée : utilise des soustractions de datetime (ex: df['date_fin'] - df['date_debut'])
    - Pour les analyses par jour de la semaine : utilise .dt.dayofweek ou .dt.day_name()
    - Pour les analyses par heure : utilise .dt.hour
14. GRAPHIQUES TEMPORELS - RÈGLES CRITIQUES :
    - Pour les séries temporelles, utilise TOUJOURS l'axe X pour le temps (dates/heures)
    - OBLIGATOIRE: Convertir les dates en datetime avec pd.to_datetime() AVANT toute opération
    - OBLIGATOIRE: TRIER les données par date chronologiquement AVANT de tracer (sort_values('date'))
    - OBLIGATOIRE: Utiliser mode='lines+markers' pour voir la courbe complète avec tous les points
    - Si tu fais un groupby sur des dates, CONVERTIR en datetime AVANT le groupby, puis TRIER après
    - Utilise px.line() avec markers=True pour les séries temporelles continues
    - Utilise px.scatter() pour voir tous les points individuels
    - Utilise px.bar() pour les agrégations par période (jour, semaine, mois)
    - NE JAMAIS tracer sans trier les dates - cela crée des droites au lieu de courbes!

9. OBLIGATOIRE : Pour créer des graphiques, TU DOIS utiliser Plotly (plotly.express ou plotly.graph_objects) UNIQUEMENT. 
   - N'utilise JAMAIS matplotlib pour créer des graphiques
   - Plotly permet un affichage interactif dynamique (zoom, pan, hover, etc.)
   - Utilise plotly.express (px) pour des graphiques simples et rapides
   - Utilise plotly.graph_objects (go) pour plus de contrôle
   - Exécute toujours le code avec 'python_code_executor', ne renvoie jamais de code seul
   - Assigne la figure à la variable 'fig' : fig = px.xxx(...) ou fig = go.Figure(...)
   - Assigne result = 'graph_ok' à la fin
   - RÈGLE CRITIQUE POUR LES COURBES: 
     * Convertir les dates en datetime AVANT toute opération
     * TRIER les données par date AVANT de tracer (sort_values('date'))
     * Utiliser mode='lines+markers' pour voir la courbe complète
     * Si tu agrèges (groupby), trier APRÈS l'agrégation

Exemple pour filtrer :
Action: python_code_executor
Action Input: result = df[df['prix'] > 100]

Exemple pour compter :
Action: python_code_executor
Action Input: result = df[df['categorie'] == 'A'].shape[0]

Exemple pour un histogramme avec Plotly (OBLIGATOIRE pour les histogrammes) :
Action: python_code_executor
Action Input: 
    import plotly.express as px
    # Vérifier que la colonne existe et contient des données
    if 'age' in df.columns:
        df_age = df[df['age'].notna()]  # Filtrer les valeurs manquantes
        if len(df_age) > 0:
            fig = px.histogram(df_age, x='age', nbins=20, title='Répartition des âges')
            fig.update_xaxes(title_text='Âge')
            fig.update_yaxes(title_text='Fréquence')
            result = 'graph_ok'
        else:
            result = 'Aucune donnée disponible pour tracer l\'histogramme'
    else:
        result = 'La colonne "age" n\'existe pas. Colonnes disponibles: ' + str(list(df.columns))

Exemple pour une courbe avec Plotly (OBLIGATOIRE pour les courbes/lignes) :
Action: python_code_executor
Action Input: 
    import plotly.express as px
    import pandas as pd
    # IMPORTANT: Convertir en datetime AVANT le groupby
    df['date'] = pd.to_datetime(df['date'])
    # Faire l'agrégation
    data = df.groupby('date')['Global_active_power'].sum().reset_index()
    # S'assurer que la colonne date reste en datetime (au cas où)
    data['date'] = pd.to_datetime(data['date'])
    # TRIER par date pour avoir une courbe continue (CRITIQUE!)
    data = data.sort_values('date')
    # Utiliser mode='lines+markers' pour voir les points ET la ligne
    fig = px.line(data, x='date', y='Global_active_power', title='Consommation totale par jour', markers=True)
    fig.update_traces(mode='lines+markers')  # Afficher lignes ET points
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='kW')
    result = 'graph_ok'

Exemple pour analyser des données temporelles (dates/heures) :
Action: python_code_executor
Action Input: 
    import pandas as pd
    # Convertir la colonne date en datetime si ce n'est pas déjà fait
    df['date'] = pd.to_datetime(df['date'])
    # Filtrer par période (ex: janvier 2024)
    df_filtered = df[(df['date'] >= '2024-01-01') & (df['date'] < '2024-02-01')]
    # Agréger par jour
    data = df_filtered.groupby(df_filtered['date'].dt.date)['valeur'].sum().reset_index()
    data.columns = ['date', 'valeur_totale']
    result = data

Exemple pour analyser par jour de la semaine :
Action: python_code_executor
Action Input: 
    import pandas as pd
    df['date'] = pd.to_datetime(df['date'])
    df['jour_semaine'] = df['date'].dt.day_name()
    data = df.groupby('jour_semaine')['valeur'].mean().reset_index()
    result = data

Exemple pour analyser par heure de la journée :
Action: python_code_executor
Action Input: 
    import pandas as pd
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['heure'] = df['datetime'].dt.hour
    data = df.groupby('heure')['valeur'].mean().reset_index()
    result = data

Exemple pour une série temporelle avec Plotly (dates sur l'axe X) :
Action: python_code_executor
Action Input: 
    import plotly.express as px
    import pandas as pd
    # Convertir en datetime et TRIER chronologiquement (OBLIGATOIRE!)
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values('date')
    # Tracer la série temporelle avec markers pour voir tous les points
    fig = px.line(df_sorted, x='date', y='valeur', title='Évolution dans le temps', markers=True)
    fig.update_traces(mode='lines+markers')  # Lignes ET points pour voir la courbe complète
    fig.update_xaxes(title_text='Date/Heure')
    fig.update_yaxes(title_text='Valeur')
    result = 'graph_ok'

Exemple pour tracer DIRECTEMENT sans agrégation (si les données sont déjà temporelles) :
Action: python_code_executor
Action Input: 
    import plotly.express as px
    import pandas as pd
    # Si les données ont déjà une colonne datetime, convertir et trier
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_sorted = df.sort_values('datetime')
    # Tracer DIRECTEMENT toutes les données (pas de groupby) pour voir la vraie courbe
    fig = px.line(df_sorted, x='datetime', y='valeur', title='Évolution complète', markers=True)
    fig.update_traces(mode='lines+markers', line_shape='linear')  # Courbe linéaire entre points
    fig.update_xaxes(title_text='Date/Heure')
    fig.update_yaxes(title_text='Valeur')
    result = 'graph_ok'

Exemple pour un scatter plot avec Plotly (OBLIGATOIRE pour les scatter plots) :
Action: python_code_executor
Action Input: 
    import plotly.express as px
    fig = px.scatter(df, x='salaire', y='age', title='Salaire vs Âge')
    result = 'graph_ok'

Exemple pour un graphique en barres avec Plotly (OBLIGATOIRE pour les barres) :
Action: python_code_executor
Action Input: 
    import plotly.express as px
    data = df.groupby('categorie')['montant'].sum().reset_index()
    fig = px.bar(data, x='categorie', y='montant', title='Montant par catégorie')
    result = 'graph_ok'

FORMAT DE SORTIE RECOMMANDÉ :

Final Answer: <réponse directe à la question>
Explications:
- <méthode/colonnes utilisées>
- <1-2 chiffres clés résumant l'analyse>
- <notes, hypothèses ou filtres, si pertinents>

Commence maintenant !

Question: {input}
Thought: {agent_scratchpad}"""
        
        # IMPORTANT: create_react_agent attend que le PromptTemplate expose
        # 'tools' et 'tool_names' comme variables. On ne les remplace pas ici,
        # on déclare simplement qu'elles font partie des variables du template.
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        )
        return prompt
    
    def query(self, question: str) -> str:
        """
        Pose une question à l'agent sur les données CSV
        
        Args:
            question: La question en langage naturel
            
        Returns:
            La réponse de l'agent
        """
        # Ajouter un délai entre les appels LLM pour éviter les erreurs 429
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        if time_since_last_call < Config.LLM_REQUEST_DELAY:
            delay_needed = Config.LLM_REQUEST_DELAY - time_since_last_call
            if self.verbose:
                print(f"⏳ Délai de {delay_needed:.2f}s pour respecter les limites de taux...")
            time.sleep(delay_needed)
        
        self.last_llm_call_time = time.time()
        
        # Retry avec backoff exponentiel pour les erreurs 429
        max_retries = 3
        base_delay = 5  # Délai de base en secondes pour les erreurs 429
        
        for attempt in range(max_retries):
            try:
                response = self.agent_executor.invoke({"input": question})
                final_text = response.get("output", "")
                # Récupère les observations d'outils pour extraire d'éventuels marqueurs PLOT
                intermediates = response.get("intermediate_steps", [])
                plot_markers = []
                for step in intermediates:
                    # step est (AgentAction, observation)
                    if isinstance(step, (list, tuple)) and len(step) == 2:
                        observation = step[1]
                        if isinstance(observation, str) and ("PLOT::" in observation or "PLOT_B64::" in observation or "PLOTLY_JSON::" in observation):
                            # extraire toutes les lignes contenant les marqueurs
                            for line in observation.splitlines():
                                if line.startswith("PLOT::") or line.startswith("PLOT_B64::") or line.startswith("PLOTLY_JSON::"):
                                    plot_markers.append(line)
                if plot_markers:
                    final_text = final_text + "\n" + "\n".join(plot_markers)
                return final_text
            except Exception as e:
                error_str = str(e)
                # Détecter les erreurs 429 (Resource Exhausted)
                if "429" in error_str or "ResourceExhausted" in error_str or "resource exhausted" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Calculer le délai avec backoff exponentiel
                        delay = base_delay * (2 ** attempt)
                        if self.verbose:
                            print(f"⚠️ Erreur 429 détectée. Nouvelle tentative dans {delay}s... (tentative {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        self.last_llm_call_time = time.time()  # Mettre à jour après le délai
                        continue
                    else:
                        return f"❌ Erreur : Limite de taux API atteinte (429). Veuillez patienter quelques minutes avant de réessayer.\nDétails : {error_str}"
                else:
                    # Pour les autres erreurs, retourner immédiatement
                    return f"❌ Erreur : {error_str}"
        
        return "❌ Erreur : Échec après plusieurs tentatives"
    
    def chat(self):
        """
        Lance une session de chat interactive avec l'agent
        """
        print("=" * 70)
        print("💬 Mode Chat Interactif")
        print("=" * 70)
        print("Posez vos questions sur le fichier CSV.")
        print("Tapez 'quit', 'exit' ou 'q' pour quitter.\n")
        
        while True:
            try:
                question = input("🧑 Vous : ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', 'quitter']:
                    print("\n👋 Au revoir !")
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Agent : ", end="")
                response = self.query(question)
                print(response)
                print("\n" + "-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"\n❌ Erreur : {str(e)}\n")
    
    def get_dataframe(self):
        """
        Retourne le DataFrame pandas pour un accès direct si nécessaire
        """
        return self.csv_tools.df


if __name__ == "__main__":
    # Exemple d'utilisation
    print("🚀 Démonstration de l'agent CSV\n")
    
    # Vérifie si un fichier CSV existe pour la démo
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Usage: python csv_agent.py <fichier.csv>")
        print("\nOu créez un fichier d'exemple et relancez.")
        sys.exit(1)
    
    # Création de l'agent
    agent = CSVAgent(csv_file, verbose=False)
    
    # Mode chat interactif
    agent.chat()

