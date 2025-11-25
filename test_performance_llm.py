"""
Script de test de performance pour mesurer le temps de réponse du modèle LLM
"""
import time
import sys
import requests
from config import Config
from llm_factory import get_llm

# Fix encoding pour Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_direct_api(question: str) -> tuple:
    """Test direct via l'API Ollama (sans LangChain)"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL_NAME,
                "prompt": question,
                "stream": False
            },
            timeout=300  # 5 minutes max
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "")
            tokens_per_sec = result.get("eval_count", 0) / elapsed_time if elapsed_time > 0 else 0
            return True, elapsed_time, answer, tokens_per_sec
        else:
            return False, elapsed_time, f"Erreur {response.status_code}", 0
    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, elapsed_time, str(e), 0

def test_langchain_llm(question: str) -> tuple:
    """Test via LangChain (comme dans l'application)"""
    start_time = time.time()
    try:
        llm = get_llm(
            model_name=Config.OLLAMA_MODEL_NAME,
            temperature=0,
            max_output_tokens=2048,
            verbose=False
        )
        response = llm.invoke(question)
        elapsed_time = time.time() - start_time
        
        answer = response.content if hasattr(response, 'content') else str(response)
        return True, elapsed_time, answer, 0
    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, elapsed_time, str(e), 0

def format_time(seconds: float) -> str:
    """Formate le temps en format lisible"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"

def main():
    """Fonction principale de test"""
    print("=" * 70)
    print("  Test de Performance - Modèle LLM")
    print("=" * 70)
    print(f"Modèle testé : {Config.OLLAMA_MODEL_NAME}")
    print(f"URL Ollama : {Config.OLLAMA_BASE_URL}")
    print()
    
    # Questions de test de complexité croissante
    test_questions = [
        ("Question simple", "Dis bonjour en français."),
        ("Question courte", "Qu'est-ce que Python ? Réponds en une phrase."),
        ("Question moyenne", "Explique brièvement la différence entre une liste et un tuple en Python."),
        ("Question complexe", "Explique comment fonctionne l'analyse de séries temporelles. Donne un exemple concret."),
        ("Question très complexe", "Décris en détail le processus d'analyse de données CSV avec un agent IA, incluant les étapes de préparation, transformation et visualisation."),
    ]
    
    print("📊 Test 1 : API Directe (sans LangChain)")
    print("-" * 70)
    results_direct = []
    
    for name, question in test_questions:
        print(f"\n🔍 {name}...")
        print(f"   Question : {question[:60]}...")
        success, elapsed, answer, tokens_per_sec = test_direct_api(question)
        
        if success:
            answer_preview = answer[:100].replace('\n', ' ') + "..." if len(answer) > 100 else answer
            print(f"   ✅ Réussi en {format_time(elapsed)}")
            if tokens_per_sec > 0:
                print(f"   📈 Vitesse : {tokens_per_sec:.1f} tokens/seconde")
            print(f"   💬 Réponse : {answer_preview}")
            results_direct.append((name, elapsed, tokens_per_sec))
        else:
            print(f"   ❌ Échec après {format_time(elapsed)}")
            print(f"   Erreur : {answer}")
            results_direct.append((name, elapsed, 0))
    
    print("\n" + "=" * 70)
    print("📊 Test 2 : Via LangChain (comme dans l'application)")
    print("-" * 70)
    results_langchain = []
    
    for name, question in test_questions:
        print(f"\n🔍 {name}...")
        print(f"   Question : {question[:60]}...")
        success, elapsed, answer, _ = test_langchain_llm(question)
        
        if success:
            answer_preview = answer[:100].replace('\n', ' ') + "..." if len(answer) > 100 else answer
            print(f"   ✅ Réussi en {format_time(elapsed)}")
            print(f"   💬 Réponse : {answer_preview}")
            results_langchain.append((name, elapsed))
        else:
            print(f"   ❌ Échec après {format_time(elapsed)}")
            print(f"   Erreur : {answer}")
            results_langchain.append((name, elapsed))
    
    # Résumé
    print("\n" + "=" * 70)
    print("📈 RÉSUMÉ DES PERFORMANCES")
    print("=" * 70)
    
    print("\n🔹 API Directe :")
    print(f"{'Question':<25} {'Temps':<15} {'Tokens/s':<15}")
    print("-" * 55)
    for name, elapsed, tokens_per_sec in results_direct:
        tokens_str = f"{tokens_per_sec:.1f}" if tokens_per_sec > 0 else "N/A"
        print(f"{name:<25} {format_time(elapsed):<15} {tokens_str:<15}")
    
    print("\n🔹 Via LangChain :")
    print(f"{'Question':<25} {'Temps':<15}")
    print("-" * 40)
    for name, elapsed in results_langchain:
        print(f"{name:<25} {format_time(elapsed):<15}")
    
    # Moyennes
    if results_direct:
        avg_direct = sum(e[1] for e in results_direct) / len(results_direct)
        print(f"\n⏱️  Temps moyen (API Directe) : {format_time(avg_direct)}")
    
    if results_langchain:
        avg_langchain = sum(e[1] for e in results_langchain) / len(results_langchain)
        print(f"⏱️  Temps moyen (LangChain) : {format_time(avg_langchain)}")
    
    if results_direct and results_langchain:
        overhead = avg_langchain - avg_direct
        overhead_pct = (overhead / avg_direct * 100) if avg_direct > 0 else 0
        print(f"📊 Surcharge LangChain : {format_time(overhead)} ({overhead_pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ Test terminé !")
    print("=" * 70)

if __name__ == "__main__":
    main()

