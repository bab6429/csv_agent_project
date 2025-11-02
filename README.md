# 🤖 Agent IA d'Analyse CSV avec LangChain & Gemini

Un agent intelligent capable d'analyser des fichiers CSV et de répondre à des questions en langage naturel via une interface Streamlit ou en ligne de commande.

## 📋 Table des matières

1. [Installation](#-installation)
2. [Configuration](#-configuration)
3. [Utilisation](#-utilisation)
4. [Capacités](#-capacités-de-lagent)
5. [Interface Streamlit](#-interface-streamlit)
6. [Architecture](#-architecture)
7. [Dépannage](#-dépannage)

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- Une clé API Google Gemini (gratuite) : [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Créer le fichier `.env`

Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :

```env
GOOGLE_API_KEY=votre_cle_api_ici
```

**Optionnel :** Utilisez le script de configuration :

```bash
python setup.py
```

---

## 📝 Utilisation

### Mode 1 : Interface Streamlit (Recommandé)

**Windows :**
```bash
# Double-cliquez sur run_streamlit.bat
# ou en ligne de commande :
streamlit run app.py
```

**Linux/Mac :**
```bash
./run_streamlit.sh
# ou
streamlit run app.py
```

L'application s'ouvre à **http://localhost:8501**

**Fonctionnalités :**
- Upload de fichiers CSV via glisser-déposer
- Aperçu des données (métriques, statistiques, colonnes)
- Chat interactif avec l'agent IA
- Mode verbeux pour voir le raisonnement

### Mode 2 : Ligne de commande interactive

```bash
python main.py votre_fichier.csv
```

Puis posez vos questions dans la console.

### Mode 3 : Utilisation programmatique

```python
from csv_agent import CSVAgent

# Créer l'agent
agent = CSVAgent("data.csv")

# Poser des questions
reponse = agent.query("Quelle est la moyenne de la colonne 'prix' ?")
print(reponse)

# Accéder au DataFrame directement
df = agent.get_dataframe()
print(df.head())
```

---

## 🎯 Capacités de l'agent

L'agent peut :
- ✅ Lire et analyser des fichiers CSV
- ✅ Calculer des statistiques descriptives (moyenne, médiane, écart-type, etc.)
- ✅ Filtrer et interroger les données
- ✅ Identifier les valeurs manquantes
- ✅ Effectuer des analyses de corrélation
- ✅ Exécuter du code Python personnalisé pour des analyses complexes
- ✅ Répondre en langage naturel (français)

### Exemples de questions

**Structure :**
- "Quelle est la structure du fichier ?"
- "Combien y a-t-il de lignes et de colonnes ?"
- "Montre-moi les 10 premières lignes"

**Statistiques :**
- "Calcule la moyenne de la colonne prix"
- "Quelle est la médiane des salaires ?"
- "Donne-moi les statistiques descriptives"

**Filtrage :**
- "Combien de ventes ont un prix supérieur à 100 euros ?"
- "Affiche les produits de la catégorie Électronique"
- "Quelle est la moyenne des prix pour les Laptops ?"

**Agrégations :**
- "Quelle est la somme totale des montants ?"
- "Quelle est la moyenne des prix par région ?"
- "Quel produit génère le plus de revenus ?"

**Analyses :**
- "Quelle est la corrélation entre prix et quantité ?"
- "Affiche-moi les 10 salaires les plus élevés"
- "Y a-t-il des valeurs manquantes ?"

---

## 🌐 Interface Streamlit

### Guide complet

L'interface Streamlit offre une expérience utilisateur complète :

1. **Upload de fichier** : Glissez-déposez votre CSV ou sélectionnez-le
2. **Aperçu des données** : Métriques, onglets (Données, Statistiques, Colonnes)
3. **Chat avec l'agent** : Posez vos questions en français
4. **Mode verbeux** : Activez dans la barre latérale pour voir le raisonnement

### Configuration de la clé API dans Streamlit

Si vous n'avez pas de fichier `.env`, vous pouvez entrer votre clé API directement dans la barre latérale de l'interface.

### Options disponibles

- **Mode verbeux** : Affiche le processus de raisonnement de l'agent (Thought → Action → Observation)
- **Aperçu des données** : Slider pour choisir le nombre de lignes à afficher
- **Statistiques** : Vue complète des statistiques descriptives

---

## 🏗️ Architecture

### Composants principaux

```
┌─────────────────────────────────────────┐
│            CSVAgent                     │
│  • Orchestrateur principal              │
│  • Gère LangChain + Gemini              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│            CSVTools                     │
│  • get_csv_info()                       │
│  • get_head()                           │
│  • get_statistics()                     │
│  • count_missing_values()               │
│  • get_correlation()                    │
│  • python_code_executor()               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Pandas DataFrame                │
│  • Données CSV chargées                 │
└─────────────────────────────────────────┘
```

### Pattern ReAct (Reasoning + Acting)

L'agent utilise le pattern ReAct pour raisonner :

1. **Thought** : Analyse la question
2. **Action** : Choisit l'outil approprié
3. **Observation** : Examine le résultat
4. **Final Answer** : Formule la réponse

**Exemple :**

```
Question: "Quelle est la moyenne des prix pour les Laptops ?"

Thought: Je dois filtrer les lignes où produit='Laptop', puis calculer la moyenne

Action: python_code_executor
Action Input: result = df[df['produit'] == 'Laptop']['prix'].mean()

Observation: 1234.56

Final Answer: La moyenne des prix pour les Laptops est de 1234.56€
```

### Technologies utilisées

- **LangChain** : Framework pour construire des agents IA
- **Google Gemini** : Modèle de langage pour comprendre et raisonner
- **Pandas** : Manipulation et analyse de données
- **Streamlit** : Interface web interactive

---

## 🔧 Dépannage

### Problème : "ImportError: No module named langchain"

**Solution :**
```bash
pip install -r requirements.txt
```

### Problème : "Clé API Google manquante"

**Solutions :**
1. Créez un fichier `.env` avec `GOOGLE_API_KEY=votre_cle`
2. Ou entrez la clé directement dans l'interface Streamlit
3. Obtenez une clé sur [Google AI Studio](https://makersuite.google.com/app/apikey)

### Problème : "File not found: fichier.csv"

**Solution :** Vérifiez que le chemin du fichier est correct. Utilisez un chemin absolu si nécessaire.

### Problème : L'agent ne répond pas correctement

**Solutions :**
1. Activez le mode verbeux pour voir le raisonnement
2. Reformulez votre question plus clairement
3. Vérifiez que les noms de colonnes sont corrects
4. Soyez plus précis dans votre question

### Problème : Erreur de quota API

**Solution :** Gemini a un quota gratuit. Si vous le dépassez :
1. Attendez quelques minutes
2. Ou créez une nouvelle clé API

### Problème : Port déjà utilisé (Streamlit)

**Solution :**
```bash
# Utiliser un autre port
streamlit run app.py --server.port 8502
```

### Problème : Erreur d'encodage CSV

**Solution :** Votre CSV doit être en UTF-8. Ouvrez-le avec Notepad++ et convertissez l'encodage si nécessaire.

---

## 📊 Structure du projet

```
csv_agent_project/
├── csv_agent.py          # Agent principal
├── csv_tools.py          # Outils d'analyse CSV
├── config.py             # Configuration
├── main.py               # Point d'entrée CLI
├── app.py                # Interface Streamlit
├── setup.py              # Script de configuration
├── requirements.txt      # Dépendances
└── README.md             # Ce fichier
```

---

## 🔐 Sécurité

⚠️ **Important :** L'outil `python_code_executor` exécute du code Python arbitraire. En production :

- Limitez les opérations permises
- Utilisez un sandbox
- Validez les inputs
- Loggez les actions pour audit

---

## 🚀 Prochaines étapes

1. ✅ Testez avec vos propres fichiers CSV
2. ✅ Explorez l'interface Streamlit
3. ✅ Adaptez le code à vos besoins
4. ✅ Consultez le code source pour comprendre l'architecture

---

## 💡 Astuces

- **Pour de meilleurs résultats** : Soyez précis dans vos questions et utilisez les noms exacts des colonnes
- **Mode verbeux** : Utilisez-le pour comprendre comment l'agent raisonne
- **Fichiers volumineux** : L'agent fonctionne avec des fichiers de plusieurs milliers de lignes, mais restez sous 100 MB pour de meilleures performances

---

## 📞 Support

Pour toute question ou problème :
1. Consultez la section Dépannage ci-dessus
2. Activez le mode verbeux pour voir le raisonnement
3. Vérifiez les logs dans le terminal

---

**Bon analyse ! 📊✨**
