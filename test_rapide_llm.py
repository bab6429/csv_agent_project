"""
Test rapide de performance - Une seule question
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

def test_rapide():
    """Test rapide avec une seule question"""
    print("=" * 70)
    print("  Test Rapide - Temps de Réponse LLM")
    print("=" * 70)
    print(f"Modèle : {Config.OLLAMA_MODEL_NAME}")
    print(f"URL : {Config.OLLAMA_BASE_URL}")
    print()
    
    # Question de test
    question = "Explique brièvement comment analyser un fichier CSV avec Python."
    
    print(f"Question : {question}")
    print()
    
    # Test 1 : API Directe
    print("Test 1 : API Directe (sans LangChain)...")
    print("-" * 70)
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL_NAME,
                "prompt": question,
                "stream": False
            },
            timeout=300
        )
        elapsed_direct = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "")
            eval_count = result.get("eval_count", 0)
            tokens_per_sec = eval_count / elapsed_direct if elapsed_direct > 0 else 0
            
            print(f"✅ Réussi en {elapsed_direct:.2f} secondes")
            print(f"📊 Tokens générés : {eval_count}")
            print(f"📈 Vitesse : {tokens_per_sec:.1f} tokens/seconde")
            print(f"\n💬 Réponse :")
            print("-" * 70)
            print(answer)
            print("-" * 70)
        else:
            print(f"❌ Erreur {response.status_code} : {response.text}")
            elapsed_direct = time.time() - start_time
    except Exception as e:
        elapsed_direct = time.time() - start_time
        print(f"❌ Erreur : {e}")
        print(f"   Temps écoulé : {elapsed_direct:.2f}s")
    
    print()
    print("=" * 70)
    
    # Test 2 : Via LangChain
    print("Test 2 : Via LangChain (comme dans l'application)...")
    print("-" * 70)
    start_time = time.time()
    
    try:
        llm = get_llm(
            model_name=Config.OLLAMA_MODEL_NAME,
            temperature=0,
            max_output_tokens=2048,
            verbose=False
        )
        response = llm.invoke(question)
        elapsed_langchain = time.time() - start_time
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        print(f"✅ Réussi en {elapsed_langchain:.2f} secondes")
        print(f"\n💬 Réponse :")
        print("-" * 70)
        print(answer)
        print("-" * 70)
    except Exception as e:
        elapsed_langchain = time.time() - start_time
        print(f"❌ Erreur : {e}")
        print(f"   Temps écoulé : {elapsed_langchain:.2f}s")
    
    # Résumé
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"API Directe    : {elapsed_direct:.2f} secondes")
    print(f"Via LangChain  : {elapsed_langchain:.2f} secondes")
    
    if elapsed_direct > 0 and elapsed_langchain > 0:
        overhead = elapsed_langchain - elapsed_direct
        overhead_pct = (overhead / elapsed_direct * 100) if elapsed_direct > 0 else 0
        print(f"Surcharge      : {overhead:.2f}s ({overhead_pct:.1f}%)")
    
    print("=" * 70)

if __name__ == "__main__":
    test_rapide()

