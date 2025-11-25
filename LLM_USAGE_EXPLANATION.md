# 📚 Explication : Quand les LLM sont utilisés dans le système

## 🎯 Vue d'ensemble

Le système utilise des **LLM (Large Language Models)** à **3 niveaux différents** pour traiter les questions des utilisateurs. Voici une explication claire de chaque utilisation.

---

## 🔄 Flux complet avec les LLM

```
Question Utilisateur
    ↓
┌─────────────────────────────────────┐
│ 1. OrchestratorAgent (LLM #1)      │
│    → Routing intelligent            │
│    → Décide quel agent utiliser     │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────────┐  ┌──────────────┐
│ 2. TimeSeries│  │ 2. Transform │
│    Agent     │  │    Agent     │
│  (LLM #2)    │  │  (LLM #2)    │
│              │  │              │
│ Pattern ReAct│  │ Pattern ReAct│
│ + Outils     │  │ + Outils     │
└──────────────┘  └──────────────┘
```

---

## 1️⃣ LLM de l'OrchestratorAgent (Routing)

### **Quand est-il utilisé ?**
- **À chaque question** de l'utilisateur
- **Avant** que la question soit traitée par un agent spécialisé
- **Une seule fois** par question

### **Rôle :**
Décider quel agent spécialisé doit traiter la question

### **Comment ça fonctionne :**
```python
# Dans orchestrator_agent.py
def _detect_agent_type(self, question: str) -> str:
    # Le LLM analyse la question
    routing_prompt = f"""Analyse cette question et détermine quel agent utiliser...
    Question: "{question}"
    Agents: time_series ou transformation"""
    
    response = self.routing_llm.invoke(routing_prompt)
    # Retourne "time_series" ou "transformation"
```

### **Exemple :**
```
Question: "Quelle est la tendance des ventes ?"
    ↓
LLM Orchestrateur analyse → "time_series"
    ↓
Route vers TimeSeriesAgent
```

### **Caractéristiques :**
- ✅ **Rapide** : max_output_tokens=50 (très court)
- ✅ **Déterministe** : temperature=0
- ✅ **Simple** : Juste choisir entre 2 options

---

## 2️⃣ LLM des Agents Spécialisés (Traitement)

### **Quand sont-ils utilisés ?**
- **Après** le routing par l'orchestrateur
- **Pour chaque question** routée vers l'agent
- **Plusieurs fois** si nécessaire (pattern ReAct)

### **Rôle :**
Traiter la question en utilisant le pattern ReAct (Reasoning + Acting)

### **Agents concernés :**
1. **TimeSeriesAgent** - Pour les questions temporelles
2. **TransformationAgent** - Pour les questions générales/stats

### **Comment ça fonctionne (Pattern ReAct) :**

```
Question: "Calcule la tendance des ventes"
    ↓
┌─────────────────────────────────────┐
│ LLM TimeSeriesAgent                 │
│                                     │
│ Thought: "Je dois calculer la      │
│          tendance de la colonne    │
│          ventes"                    │
│                                     │
│ Action: calculate_trend             │
│ Action Input: ventes                │
│                                     │
│ Observation: [résultat de l'outil] │
│                                     │
│ Thought: "J'ai la réponse"          │
│ Final Answer: "La tendance est..."  │
└─────────────────────────────────────┘
```

### **Exemple concret :**

```python
# Dans time_series_agent.py
def query(self, question: str) -> str:
    # Le LLM utilise le pattern ReAct
    response = self.agent_executor.invoke({"input": question})
    # Le LLM peut appeler plusieurs outils en boucle
    # Thought → Action → Observation → Thought → Action → ...
    # Jusqu'à trouver la réponse finale
```

### **Caractéristiques :**
- ✅ **Itératif** : Peut faire plusieurs actions (max 3-4)
- ✅ **Avec outils** : Accès à des outils spécialisés
- ✅ **Contextuel** : Comprend le contexte de la question
- ✅ **Prompt spécialisé** : Chaque agent a son propre prompt optimisé

---

## 📊 Résumé : Utilisation des LLM

| Étape | LLM Utilisé | Quand | Rôle | Nombre d'appels |
|-------|-------------|-------|------|-----------------|
| **1. Routing** | OrchestratorAgent | À chaque question | Choisir l'agent | **1 fois** |
| **2. Traitement** | TimeSeriesAgent OU TransformationAgent | Après routing | Traiter la question | **1-4 fois** (ReAct) |

---

## 🔢 Nombre total d'appels LLM par question

### **Cas simple (1 action) :**
```
Question → Orchestrator LLM (1) → Agent LLM (1) = **2 appels LLM**
```

### **Cas complexe (3 actions) :**
```
Question → Orchestrator LLM (1) → Agent LLM (3) = **4 appels LLM**
```

---

## 💡 Pourquoi cette architecture ?

### **Avantages :**
1. **Routing intelligent** : Le LLM comprend le contexte, pas juste des mots-clés
2. **Spécialisation** : Chaque agent a un prompt optimisé pour son domaine
3. **Efficacité** : Les prompts sont courts et ciblés
4. **Flexibilité** : Le routing s'adapte aux questions ambiguës

### **Exemple de routing intelligent :**
```
Question: "Les ventes augmentent-elles ?"
    ↓
LLM Orchestrateur comprend que c'est une question sur tendance
    ↓
Route vers TimeSeriesAgent (même sans mot-clé explicite)
```

---

## 🎯 Points clés à retenir

1. **3 LLM au total** dans le système :
   - 1 pour le routing (OrchestratorAgent)
   - 2 pour le traitement (TimeSeriesAgent, TransformationAgent)

2. **Chaque question** déclenche :
   - 1 appel LLM pour le routing
   - 1-4 appels LLM pour le traitement (selon complexité)

3. **Les outils** (CSVTools) ne sont **PAS** des LLM :
   - Ce sont des fonctions Python qui exécutent du code
   - Exemple : `calculate_trend()` fait une régression linéaire

4. **Le pattern ReAct** permet au LLM de :
   - Raisonner (Thought)
   - Agir (Action avec outil)
   - Observer (Observation)
   - Répéter jusqu'à la réponse finale

---

## 🔍 Exemple complet de flux

```
Utilisateur: "Quelle est la tendance des ventes sur 6 mois ?"
    ↓
┌─────────────────────────────────────┐
│ LLM Orchestrateur (Appel #1)        │
│ Analyse: "tendance" + "6 mois"      │
│ Décision: time_series                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ LLM TimeSeriesAgent (Appel #2)      │
│ Thought: "Je dois calculer tendance"│
│ Action: calculate_trend              │
│ Input: ventes                        │
│ Observation: "Tendance: +5.2%"      │
│ Thought: "J'ai la réponse"           │
│ Final Answer: "La tendance est..."   │
└──────────────────────────────────────┘
    ↓
Réponse à l'utilisateur
```

**Total : 2 appels LLM** (1 routing + 1 traitement)

---

## 📝 Code de référence

- **OrchestratorAgent LLM** : `agents/orchestrator_agent.py` ligne 46-51
- **TimeSeriesAgent LLM** : `agents/time_series_agent.py` ligne 40-45
- **TransformationAgent LLM** : `agents/transformation_agent.py` ligne 40-45

---

## ❓ Questions fréquentes

**Q: Pourquoi ne pas utiliser un seul LLM avec tous les outils ?**
R: Les prompts seraient trop longs, l'agent serait confus, et les performances se dégraderaient.

**Q: Le routing pourrait-il être fait sans LLM ?**
R: Oui, mais le LLM comprend mieux le contexte. Ex: "Les ventes augmentent ?" → comprend que c'est une tendance.

**Q: Combien ça coûte en tokens ?**
R: Routing ~100 tokens, Traitement ~500-2000 tokens selon complexité. Total ~600-2100 tokens par question.

