"""
Agent IA pour l'analyse de fichiers CSV avec LangChain et Gemini
Architecture multi-agents avec orchestrateur
"""
import os
from typing import Optional
from agents.orchestrator_agent import OrchestratorAgent


class CSVAgent:
    """
    Agent intelligent pour analyser des fichiers CSV et répondre à des questions en langage naturel
    
    Architecture:
    - Utilise un système multi-agents avec orchestrateur
    - Route les questions vers des agents spécialisés (Time Series, Transformation, etc.)
    - Chaque agent spécialisé utilise le pattern ReAct avec ses propres outils
    """
    
    def __init__(self, csv_path: str, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialise l'agent CSV avec architecture multi-agents
        
        Args:
            csv_path: Chemin vers le fichier CSV à analyser
            api_key: Clé API Google (optionnel, peut être définie via variable d'environnement)
            verbose: Si True, affiche les étapes de raisonnement de l'agent
        """
        # Utilise l'orchestrateur qui gère tous les agents spécialisés
        self.orchestrator = OrchestratorAgent(
            csv_path=csv_path,
            api_key=api_key,
            verbose=verbose
        )
        
        self.csv_path = csv_path
        self.verbose = verbose
    
    def query(self, question: str) -> str:
        """
        Pose une question à l'agent sur les données CSV
        L'orchestrateur route automatiquement vers l'agent spécialisé approprié
        
        Args:
            question: La question en langage naturel
            
        Returns:
            La réponse de l'agent spécialisé
        """
        # Délègue à l'orchestrateur qui route vers le bon agent
        return self.orchestrator.query(question)
    
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
        return self.orchestrator.get_dataframe()

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


