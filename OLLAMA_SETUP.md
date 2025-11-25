# 🦙 Configuration Ollama pour l'Agent CSV

Ce projet supporte maintenant **Ollama** pour utiliser des LLM en local, avec un fallback automatique vers **Google Gemini** si Ollama n'est pas disponible.

## 📋 Prérequis

1. **Installer Ollama** : https://ollama.com/download
2. **Télécharger un modèle** :
   ```bash
   ollama pull llama3.2
   ```
   Autres modèles recommandés :
   - `ollama pull mistral` - Bon équilibre performance/taille
   - `ollama pull phi3` - Très léger (~2GB)
   - `ollama pull codellama` - Spécialisé pour le code

## ⚙️ Configuration

### Variables d'environnement

Créez un fichier `.env` à la racine du projet avec :

```env
# Modèle Ollama à utiliser (par défaut: llama3.2)
OLLAMA_MODEL_NAME=llama3.2

# URL d'Ollama (par défaut: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Clé API Google Gemini (optionnelle si Ollama est installé)
GOOGLE_API_KEY=your_google_api_key_here

# Forcer un provider spécifique (optionnel)
# USE_OLLAMA=true  # Force Ollama
# USE_GEMINI=true  # Force Gemini
```

### Comportement par défaut

1. **Auto-détection** : Le système essaie d'abord Ollama
2. **Si Ollama est disponible** : Utilise le modèle Ollama configuré
3. **Si Ollama n'est pas disponible** : Fallback automatique vers Gemini (nécessite `GOOGLE_API_KEY`)

## 🚀 Utilisation

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Démarrer Ollama

Assurez-vous qu'Ollama est en cours d'exécution :

```bash
# Vérifier qu'Ollama fonctionne
ollama list

# Si Ollama n'est pas démarré, il se lancera automatiquement
# Sinon, démarrez-le manuellement selon votre OS
```

### Lancer l'application

```bash
# Interface Streamlit
streamlit run app.py

# Ou en ligne de commande
python csv_agent.py fichier.csv
```

## 🔧 Forcer un provider spécifique

### Forcer Ollama

```bash
export USE_OLLAMA=true
python csv_agent.py fichier.csv
```

### Forcer Gemini

```bash
export USE_GEMINI=true
python csv_agent.py fichier.csv
```

## 📊 Avantages d'Ollama

- ✅ **Gratuit** : Pas de limite de requêtes
- ✅ **Local** : Données restent sur votre machine (confidentialité)
- ✅ **Hors ligne** : Fonctionne sans connexion internet
- ✅ **Pas de clé API** : Pas besoin de configurer une clé API

## ⚠️ Limitations

- ⚠️ **Ressources** : Nécessite de la RAM (4-8GB recommandés selon le modèle)
- ⚠️ **Performance** : Généralement plus lent que les API cloud
- ⚠️ **Qualité** : Varie selon le modèle choisi

## 🎯 Recommandations de modèles

Pour un agent CSV, nous recommandons :

- **llama3.2** : Bon équilibre performance/taille (~2GB)
- **mistral** : Excellente qualité, un peu plus lourd
- **phi3** : Très léger, bon pour les machines avec peu de RAM
- **codellama** : Optimisé pour générer du code Python

## 🐛 Dépannage

### Ollama n'est pas détecté

1. Vérifiez qu'Ollama est installé : `ollama --version`
2. Vérifiez qu'Ollama tourne : `ollama list`
3. Vérifiez l'URL dans `.env` : `OLLAMA_BASE_URL=http://localhost:11434`

### Erreur "Connection refused"

- Assurez-vous qu'Ollama est démarré
- Vérifiez que le port 11434 n'est pas bloqué par un firewall

### Le modèle n'existe pas

- Téléchargez le modèle : `ollama pull llama3.2`
- Vérifiez les modèles disponibles : `ollama list`
- Mettez à jour `OLLAMA_MODEL_NAME` dans `.env`

