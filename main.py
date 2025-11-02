"""
Script principal pour utiliser l'agent CSV
"""
import sys
import os
from dotenv import load_dotenv
from csv_agent import CSVAgent

# Charger les variables d'environnement
load_dotenv()


def main():
    """Point d'entrée principal"""
    
    print("=" * 70)
    print("🤖 AGENT IA D'ANALYSE CSV")
    print("   Propulsé par LangChain + Google Gemini")
    print("=" * 70)
    print()
    
    # Vérifier si un fichier CSV est fourni
    if len(sys.argv) < 2:
        print("❌ Erreur : Aucun fichier CSV spécifié")
        print("\n📖 Usage:")
        print("   python main.py <fichier.csv>")
        print("\n💡 Exemple:")
        print("   python main.py ventes.csv")
        print("   python main.py data/produits.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Vérifier que le fichier existe
    if not os.path.exists(csv_file):
        print(f"❌ Erreur : Le fichier '{csv_file}' n'existe pas")
        sys.exit(1)
    
    # Vérifier la clé API
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Erreur : Clé API Google manquante")
        print("\n📖 Pour configurer votre clé API :")
        print("   1. Créez un fichier .env dans le répertoire du projet")
        print("   2. Ajoutez : GOOGLE_API_KEY=votre_cle_ici")
        print("   3. Obtenez une clé sur : https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    try:
        # Créer l'agent
        agent = CSVAgent(csv_file, verbose=False)
        
        # Lancer le mode interactif
        agent.chat()
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'initialisation : {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

