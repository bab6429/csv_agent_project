# 🦙 Guide d'installation Ollama - Modèles légers (≤7B)

## 📥 Étape 1 : Installer Ollama

### Windows
1. Téléchargez l'installateur depuis : https://ollama.com/download/windows
2. Exécutez l'installateur (OllamaSetup.exe)
3. Ollama se lancera automatiquement après l'installation

### Vérifier l'installation
Ouvrez un terminal (PowerShell ou CMD) et tapez :
```bash
ollama --version
```

Si vous voyez un numéro de version, c'est bon ! ✅

---

## 🎯 Étape 2 : Choisir un modèle léger (≤7B)

Pour un PC peu puissant, voici les meilleurs modèles recommandés :

### ⭐ **Recommandation #1 : Phi-3 Mini (3.8B)**
- **Taille** : ~2.3 GB
- **RAM nécessaire** : ~4-6 GB
- **Qualité** : Excellente pour sa taille
- **Vitesse** : Très rapide
- **Commande** : `ollama pull phi3`

### ⭐ **Recommandation #2 : Llama 3.2 (3B)**
- **Taille** : ~2.0 GB
- **RAM nécessaire** : ~4-6 GB
- **Qualité** : Très bonne
- **Vitesse** : Rapide
- **Commande** : `ollama pull llama3.2`

### ⭐ **Recommandation #3 : Mistral 7B**
- **Taille** : ~4.1 GB
- **RAM nécessaire** : ~8 GB
- **Qualité** : Excellente
- **Vitesse** : Moyenne
- **Commande** : `ollama pull mistral`

### Autres options légères :
- **Gemma 2B** : `ollama pull gemma2:2b` (~1.4 GB)
- **TinyLlama 1.1B** : `ollama pull tinyllama` (~637 MB) - Très rapide mais qualité limitée

---

## 📦 Étape 3 : Télécharger le modèle

Ouvrez un terminal et exécutez :

```bash
# Pour Phi-3 Mini (recommandé pour PC peu puissant)
ollama pull phi3

# OU pour Llama 3.2 (alternative)
ollama pull llama3.2
```

Le téléchargement peut prendre quelques minutes selon votre connexion internet.

### Vérifier les modèles téléchargés
```bash
ollama list
```

Vous devriez voir votre modèle dans la liste.

---

## 🚀 Étape 4 : Tester le modèle

### Test rapide en ligne de commande
```bash
ollama run phi3
# Ou
ollama run llama3.2
```

Tapez une question et appuyez sur Entrée. Tapez `/bye` pour quitter.

### Test avec Python
```python
import requests

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'phi3',  # ou 'llama3.2'
        'prompt': 'Bonjour, peux-tu te présenter ?',
        'stream': False
    }
)
print(response.json()['response'])
```

---

## ⚙️ Étape 5 : Configurer votre projet

### Option A : Via fichier .env
Créez un fichier `.env` à la racine du projet :

```env
# Utiliser Phi-3 Mini
OLLAMA_MODEL_NAME=phi3

# Ou utiliser Llama 3.2
# OLLAMA_MODEL_NAME=llama3.2

# URL par défaut (ne changez que si nécessaire)
OLLAMA_BASE_URL=http://localhost:11434
```

### Option B : Via variable d'environnement système
```bash
# Windows PowerShell
$env:OLLAMA_MODEL_NAME="phi3"

# Windows CMD
set OLLAMA_MODEL_NAME=phi3
```

---

## 🎮 Étape 6 : Lancer votre application

```bash
# Installer les dépendances (si pas déjà fait)
pip install -r requirements.txt

# Lancer l'application Streamlit
streamlit run app.py

# OU lancer en ligne de commande
python csv_agent.py votre_fichier.csv
```

L'application détectera automatiquement Ollama et utilisera votre modèle local ! 🎉

---

## 💡 Optimisations pour PC peu puissant

### 1. Réduire le nombre de threads CPU
Ollama utilise tous les cœurs CPU par défaut. Vous pouvez limiter :

**Windows** : Modifier les variables d'environnement système
```bash
# PowerShell
$env:OLLAMA_NUM_THREAD="4"  # Utilise 4 threads au lieu de tous
```

### 2. Utiliser un modèle quantifié (plus léger)
Certains modèles ont des versions quantifiées plus légères :
```bash
# Exemple avec Llama 3.2 quantifié (si disponible)
ollama pull llama3.2:q4_0  # Version quantifiée 4-bit
```

### 3. Fermer les autres applications
Libérez de la RAM en fermant les applications inutiles.

### 4. Vérifier la RAM disponible
```bash
# Windows PowerShell
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory
```

**Recommandations RAM** :
- Phi-3 / Llama 3.2 : Minimum 4 GB RAM libre
- Mistral 7B : Minimum 8 GB RAM libre

---

## 🐛 Dépannage

### Ollama ne démarre pas
```bash
# Vérifier si Ollama tourne
ollama list

# Si erreur, redémarrer Ollama
# Windows : Cherchez "Ollama" dans le menu Démarrer et relancez
```

### Le modèle est trop lent
- Essayez un modèle plus petit (Phi-3 au lieu de Mistral)
- Réduisez le nombre de threads CPU
- Fermez les autres applications

### Erreur "out of memory"
- Utilisez un modèle plus petit
- Fermez les autres applications
- Redémarrez votre PC pour libérer la RAM

### Le modèle n'est pas trouvé
```bash
# Vérifier les modèles installés
ollama list

# Si le modèle n'est pas là, téléchargez-le
ollama pull phi3
```

---

## 📊 Comparaison des modèles légers

| Modèle | Taille | RAM min | Vitesse | Qualité | Recommandation |
|--------|--------|---------|---------|---------|----------------|
| **Phi-3 Mini** | 2.3 GB | 4 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **Meilleur choix** |
| **Llama 3.2** | 2.0 GB | 4 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Excellent |
| **Gemma 2B** | 1.4 GB | 3 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Très léger |
| **Mistral 7B** | 4.1 GB | 8 GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ Si vous avez 8GB+ RAM |
| **TinyLlama** | 637 MB | 2 GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⚠️ Qualité limitée |

---

## ✅ Checklist de démarrage

- [ ] Ollama installé et fonctionnel (`ollama --version`)
- [ ] Modèle téléchargé (`ollama pull phi3` ou `llama3.2`)
- [ ] Modèle testé (`ollama run phi3`)
- [ ] Fichier `.env` créé avec `OLLAMA_MODEL_NAME=phi3`
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Application lancée (`streamlit run app.py`)

---

## 🎯 Ma recommandation finale

Pour un PC peu puissant, je recommande **Phi-3 Mini** :

```bash
ollama pull phi3
```

Puis dans votre `.env` :
```env
OLLAMA_MODEL_NAME=phi3
```

C'est le meilleur compromis entre taille, vitesse et qualité pour un PC avec peu de ressources ! 🚀

