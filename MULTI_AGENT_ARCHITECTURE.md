# Architecture Multi-Agents - Documentation

## 📋 Vue d'ensemble

Le système a été refactorisé pour utiliser une **architecture multi-agents** au lieu d'un seul agent avec de nombreux outils. Cette approche offre plusieurs avantages :

- ✅ **Prompts plus courts et spécialisés** : Chaque agent a un prompt ciblé
- ✅ **Moins de confusion** : L'agent ne voit que les outils pertinents
- ✅ **Meilleure performance** : Moins d'itérations nécessaires
- ✅ **Maintenance facilitée** : Modifications isolées par agent
- ✅ **Extensibilité** : Ajout facile de nouveaux agents

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      CSVAgent (Interface)           │
│  Point d'entrée pour l'utilisateur  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   OrchestratorAgent                 │
│  • Analyse la question              │
│  • Route vers l'agent approprié     │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────────┐  ┌──────────────┐
│ Time Series  │  │Transformation│
│    Agent     │  │    Agent     │
└──────────────┘  └──────────────┘
```

## 📁 Structure des fichiers

```
csv_agent_project/
├── agents/
│   ├── __init__.py
│   ├── orchestrator_agent.py      # Agent principal qui route
│   ├── time_series_agent.py       # Agent spécialisé séries temporelles
│   └── transformation_agent.py    # Agent spécialisé transformation
├── csv_agent.py                   # Interface utilisateur (utilise orchestrateur)
├── csv_tools.py                   # Outils partagés (CSVTools)
└── ...
```

## 🤖 Agents spécialisés

### 1. OrchestratorAgent
**Rôle** : Agent principal qui route les questions vers les agents spécialisés

**Fonctionnalités** :
- Analyse la question de l'utilisateur
- Détecte le type de question (time series, transformation, etc.)
- Route vers l'agent approprié
- Gère l'initialisation de tous les agents

**Méthode de routing** :
- Utilise des mots-clés pour détecter le type de question
- Exemples :
  - "tendance", "croissance", "moyenne mobile" → Time Series Agent
  - "structure", "statistiques", "colonnes" → Transformation Agent

### 2. TimeSeriesAgent
**Rôle** : Expert en analyse de séries temporelles

**Outils disponibles** (4 outils pour commencer) :
1. `detect_time_columns` - Détecte les colonnes temporelles
2. `calculate_trend` - Calcule la tendance (croissance/décroissance)
3. `calculate_moving_average` - Calcule la moyenne mobile
4. `aggregate_by_period` - Agrège par période (jour, semaine, mois, etc.)

**Prompt spécialisé** : Optimisé pour les questions temporelles

**Exemples de questions** :
- "Quelle est la tendance des ventes ?"
- "Calcule la moyenne mobile sur 7 jours"
- "Agrège les ventes par mois"

### 3. TransformationAgent
**Rôle** : Expert en transformation et manipulation de données

**Outils disponibles** (4 outils pour commencer) :
1. `get_csv_info` - Informations générales sur le fichier
2. `get_head` - Affiche les premières lignes
3. `get_statistics` - Statistiques descriptives
4. `count_missing_values` - Compte les valeurs manquantes

**Prompt spécialisé** : Optimisé pour les questions de structure et statistiques

**Exemples de questions** :
- "Quelle est la structure du fichier ?"
- "Affiche les 10 premières lignes"
- "Quelles sont les statistiques de la colonne X ?"

## 🔄 Flux de travail

1. **Utilisateur pose une question** → `CSVAgent.query()`
2. **CSVAgent délègue** → `OrchestratorAgent.query()`
3. **Orchestrator analyse** → Détecte le type de question
4. **Routing** → Envoie à l'agent spécialisé approprié
5. **Agent spécialisé traite** → Utilise ses outils avec son prompt optimisé
6. **Réponse** → Retourne la réponse à l'utilisateur

## 📝 Exemple de routing

```python
Question: "Quelle est la tendance des ventes sur 6 mois ?"

Orchestrator détecte:
- Mots-clés: "tendance", "6 mois"
- Score time_series: 2
- Score transformation: 0
→ Route vers TimeSeriesAgent

TimeSeriesAgent:
- Utilise calculate_trend
- Retourne l'analyse de tendance
```

## 🚀 Utilisation

L'interface reste **identique** pour l'utilisateur :

```python
from csv_agent import CSVAgent

# Création de l'agent (utilise automatiquement l'orchestrateur)
agent = CSVAgent("data.csv", verbose=True)

# Pose une question (routing automatique)
response = agent.query("Quelle est la tendance des ventes ?")
print(response)
```

## 🔧 Ajout de nouveaux agents

Pour ajouter un nouvel agent spécialisé :

1. **Créer le fichier** `agents/nouvel_agent.py`
2. **Implémenter la classe** avec `query()` et `_create_tools()`
3. **Ajouter dans orchestrator** :
   - Importer l'agent
   - Initialiser dans `__init__()`
   - Ajouter la logique de routing dans `_detect_agent_type()`
   - Ajouter le cas dans `query()`

## 📊 Avantages de cette architecture

### Avant (mono-agent)
- ❌ 1 agent avec 25+ outils
- ❌ Prompt très long
- ❌ Confusion entre outils similaires
- ❌ Difficile à maintenir

### Après (multi-agents)
- ✅ 3 agents spécialisés avec 4 outils chacun
- ✅ Prompts courts et ciblés
- ✅ Chaque agent voit uniquement ses outils
- ✅ Facile à étendre et maintenir

## 🎯 Prochaines étapes

Pour améliorer le système :

1. **Ajouter plus d'outils** aux agents existants
2. **Créer de nouveaux agents** (ex: StatisticsAgent, ReportAgent)
3. **Améliorer le routing** (utiliser un LLM léger pour le routing)
4. **Ajouter la collaboration** entre agents si nécessaire

## 📌 Notes importantes

- Les agents partagent la même instance de `CSVTools` (même DataFrame)
- Chaque agent a son propre `AgentExecutor` et prompt
- Le routing est basé sur des mots-clés (simple mais efficace)
- L'interface `CSVAgent` reste compatible avec le code existant

