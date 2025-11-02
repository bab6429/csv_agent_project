"""
Agent IA pour l'analyse de fichiers CSV avec LangChain et Gemini
"""
import os
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
        
        # Initialisation des outils CSV
        print("🔧 Initialisation des outils d'analyse...")
        self.csv_tools = CSVTools(csv_path)
        self.tools = self.csv_tools.get_tools()
        print(f"🧰 Outils disponibles: {[t.name for t in self.tools]}")
        
        # Initialisation du modèle Gemini
        print("🤖 Connexion à Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE
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
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=10,  # Limite pour éviter les boucles infinies
            max_execution_time=60,  # Timeout de 60 secondes
            return_intermediate_steps=True,
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
        template = """Tu es un assistant IA expert en analyse de données. 
Tu aides l'utilisateur à analyser un fichier CSV en répondant à ses questions en langage naturel.

Tu as accès aux outils suivants pour analyser les données :

{tools}

Utilise le format suivant pour raisonner et agir :

Question: la question que l'utilisateur te pose
Thought: tu dois toujours réfléchir à ce que tu dois faire
Action: l'action à prendre, doit être parmi [{tool_names}]
Action Input: l'entrée de l'action
Observation: le résultat de l'action
... (ce processus Thought/Action/Action Input/Observation peut se répéter N fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale à la question originale de l'utilisateur

RÈGLES IMPORTANTES :
1. Réponds TOUJOURS en français
2. Sois précis et concis dans tes réponses
3. Si tu dois faire des calculs ou analyses complexes, utilise l'outil 'python_code_executor'
4. Pour filtrer des données, utilise 'python_code_executor' avec du code pandas
5. Commence toujours par comprendre la structure des données avec 'get_csv_info' si nécessaire
6. Formate bien tes réponses finales pour qu'elles soient lisibles
7. Fournis systématiquement une courte justification/explication (2-4 points max) :
   - cite les colonnes utilisées et la méthode (ex: groupby, mean, count)
   - donne 1-2 chiffres clés (moyenne, total, top catégorie, etc.) si pertinent
   - mentionne d’éventuels filtres appliqués
8. Pour créer des graphiques, UTILISE et EXÉCUTE 'python_code_executor' avec matplotlib (plt). Ne renvoie PAS de code seul : exécute-le. Crée d'abord une figure (ex: fig, ax = plt.subplots()) puis trace. La figure sera automatiquement affichée dans l'interface.

Exemple pour filtrer :
Action: python_code_executor
Action Input: result = df[df['prix'] > 100]

Exemple pour compter :
Action: python_code_executor
Action Input: result = df[df['categorie'] == 'A'].shape[0]

Exemple pour tracer un graphique (matplotlib) :
Action: python_code_executor
Action Input: 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    df.groupby('date')['Global_active_power'].sum().plot(ax=ax)
    ax.set_title('Consommation totale par jour')
    ax.set_xlabel('Date')
    ax.set_ylabel('kW')
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
                    if isinstance(observation, str) and ("PLOT::" in observation or "PLOT_B64::" in observation):
                        # extraire toutes les lignes contenant les marqueurs
                        for line in observation.splitlines():
                            if line.startswith("PLOT::") or line.startswith("PLOT_B64::"):
                                plot_markers.append(line)
            if plot_markers:
                final_text = final_text + "\n" + "\n".join(plot_markers)
            return final_text
        except Exception as e:
            return f"❌ Erreur : {str(e)}"
    
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

