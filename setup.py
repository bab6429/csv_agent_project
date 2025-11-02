"""
Script d'installation et de configuration du projet
"""
import os
import sys


def create_env_file():
    """Crée le fichier .env si nécessaire"""
    if os.path.exists(".env"):
        print("✅ Le fichier .env existe déjà")
        return True
    
    print("📝 Création du fichier .env...")
    print("\n" + "=" * 70)
    print("Pour obtenir une clé API Google Gemini :")
    print("1. Visitez : https://makersuite.google.com/app/apikey")
    print("2. Connectez-vous avec votre compte Google")
    print("3. Créez une nouvelle clé API")
    print("4. Copiez la clé")
    print("=" * 70)
    
    api_key = input("\n🔑 Entrez votre clé API Google Gemini : ").strip()
    
    if not api_key:
        print("❌ Aucune clé fournie. Configuration annulée.")
        return False
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"GOOGLE_API_KEY={api_key}\n")
    
    print("✅ Fichier .env créé avec succès !")
    return True


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("\n🔍 Vérification des dépendances...")
    
    required = [
        "langchain",
        "langchain_google_genai",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "dotenv"
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package if package != "dotenv" else "python_dotenv")
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} manquant")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Dépendances manquantes : {', '.join(missing)}")
        print("Installez-les avec : pip install -r requirements.txt")
        return False
    
    print("\n✅ Toutes les dépendances sont installées !")
    return True


def create_sample_data():
    """Demande si on doit créer des données d'exemple"""
    if os.path.exists("ventes_exemple.csv"):
        print("\n✅ Les fichiers d'exemple existent déjà")
        return True
    
    print("\n📊 Voulez-vous créer des fichiers CSV d'exemple ?")
    response = input("   (o/n) : ").strip().lower()
    
    if response in ['o', 'oui', 'y', 'yes']:
        print("\n🎨 Création des fichiers d'exemple...")
        os.system(f"{sys.executable} create_sample_data.py")
        return True
    
    return False


def main():
    """Script principal d'installation"""
    print("=" * 70)
    print("🚀 INSTALLATION DE L'AGENT CSV")
    print("   Powered by LangChain + Google Gemini")
    print("=" * 70)
    
    # Étape 1 : Vérifier les dépendances
    if not check_dependencies():
        print("\n❌ Installation incomplète. Installez les dépendances d'abord.")
        return
    
    # Étape 2 : Créer le fichier .env
    if not create_env_file():
        print("\n❌ Configuration de la clé API échouée.")
        return
    
    # Étape 3 : Créer des données d'exemple
    create_sample_data()
    
    # Résumé
    print("\n" + "=" * 70)
    print("✅ INSTALLATION TERMINÉE !")
    print("=" * 70)
    print("\n📚 Prochaines étapes :")
    print("   1. Pour tester l'agent : python main.py ventes_exemple.csv")
    print("   2. Pour voir des exemples : python exemple_usage.py")
    print("   3. Pour l'intégrer dans votre code, consultez exemple_usage.py")
    print("\n💡 Consultez README.md et EXPLICATION.md pour plus d'informations")
    print()


if __name__ == "__main__":
    main()

