"""
Script de test pour vérifier que Ollama fonctionne correctement
"""
import requests
import sys
from config import Config

def test_ollama_connection():
    """Teste la connexion à Ollama"""
    print("🔍 Test de connexion à Ollama...")
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama est accessible !")
            return True
        else:
            print(f"❌ Erreur : Code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à Ollama")
        print(f"   Vérifiez qu'Ollama tourne sur {Config.OLLAMA_BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

def test_model_available():
    """Vérifie si le modèle configuré est disponible"""
    print(f"\n🔍 Vérification du modèle {Config.OLLAMA_MODEL_NAME}...")
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Chercher le modèle (peut être avec ou sans tag)
            model_found = False
            for model_name in model_names:
                if Config.OLLAMA_MODEL_NAME in model_name:
                    print(f"✅ Modèle trouvé : {model_name}")
                    model_found = True
                    break
            
            if not model_found:
                print(f"❌ Modèle '{Config.OLLAMA_MODEL_NAME}' non trouvé")
                print(f"   Modèles disponibles : {', '.join(model_names) if model_names else 'Aucun'}")
                print(f"\n   Pour télécharger le modèle :")
                print(f"   ollama pull {Config.OLLAMA_MODEL_NAME}")
                return False
            
            return True
        else:
            print(f"❌ Erreur lors de la vérification : Code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

def test_model_generation():
    """Teste une génération simple avec le modèle"""
    print(f"\n🔍 Test de génération avec {Config.OLLAMA_MODEL_NAME}...")
    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL_NAME,
                "prompt": "Dis bonjour en français en une phrase.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "")
            print(f"✅ Génération réussie !")
            print(f"   Réponse : {answer.strip()}")
            return True
        else:
            print(f"❌ Erreur lors de la génération : Code {response.status_code}")
            print(f"   Réponse : {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Timeout : Le modèle met trop de temps à répondre")
        print("   Votre PC est peut-être trop lent pour ce modèle")
        return False
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("  Test de configuration Ollama")
    print("=" * 60)
    print()
    
    # Test 1 : Connexion
    if not test_ollama_connection():
        print("\n💡 Solution : Assurez-vous qu'Ollama est démarré")
        print("   Windows : Cherchez 'Ollama' dans le menu Démarrer")
        sys.exit(1)
    
    # Test 2 : Modèle disponible
    if not test_model_available():
        sys.exit(1)
    
    # Test 3 : Génération
    if not test_model_generation():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  ✅ Tous les tests sont passés !")
    print("=" * 60)
    print("\nVous pouvez maintenant utiliser l'application avec Ollama.")
    print("Lancez : streamlit run app.py")

if __name__ == "__main__":
    main()

