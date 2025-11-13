# Architecture de l'Agent CSV

## Vue d'ensemble

L'agent CSV est un système d'analyse de données basé sur l'IA qui permet d'interroger des fichiers CSV/Excel en langage naturel. Il utilise le pattern ReAct (Reasoning + Acting) avec le modèle Gemini de Google.

## Schéma d'architecture

```mermaid
graph TB
    subgraph "Interface Utilisateur"
        UI1[Streamlit App<br/>app.py]
        UI2[API REST<br/>sandbox_api.py]
        UI3[CLI<br/>csv_agent.py]
    end
    
    subgraph "Couche API/Service"
        API[FastAPI Server<br/>sandbox_api.py]
        API_STORE[(Session Store<br/>agents_store)]
    end
    
    subgraph "Agent Principal"
        AGENT[CSVAgent<br/>csv_agent.py]
        EXEC[AgentExecutor<br/>LangChain]
        PROMPT[Prompt Template<br/>ReAct Pattern]
    end
    
    subgraph "Outils d'Analyse"
        TOOLS[CSVTools<br/>csv_tools.py]
        T1[get_csv_info]
        T2[get_head]
        T3[get_statistics]
        T4[count_missing_values]
        T5[get_correlation]
        T6[python_code_executor]
    end
    
    subgraph "Exécution de Code"
        SANDBOX[Sandbox API<br/>sandbox_api.py]
        LOCAL[Exécution Locale<br/>Python exec]
    end
    
    subgraph "Modèle IA"
        LLM[Google Gemini<br/>gemini-2.0-flash]
    end
    
    subgraph "Données"
        CSV[(Fichier CSV/Excel)]
        DF[(DataFrame<br/>Pandas)]
        PLOTS[(Graphiques<br/>Plotly/Matplotlib)]
    end
    
    subgraph "Configuration"
        CONFIG[Config<br/>config.py]
        ENV[(Variables<br/>d'environnement)]
    end
    
    %% Flux utilisateur
    UI1 -->|Upload CSV| API
    UI2 -->|Upload CSV| API
    UI3 -->|Fichier local| AGENT
    
    %% Flux API
    API -->|Crée| AGENT
    API -->|Stocke| API_STORE
    API -->|Query| AGENT
    
    %% Flux Agent
    AGENT -->|Initialise| TOOLS
    AGENT -->|Utilise| EXEC
    EXEC -->|Prompt| PROMPT
    EXEC -->|Appelle| LLM
    EXEC -->|Exécute| TOOLS
    PROMPT -->|Instructions| LLM
    
    %% Flux Outils
    TOOLS -->|Contient| T1
    TOOLS -->|Contient| T2
    TOOLS -->|Contient| T3
    TOOLS -->|Contient| T4
    TOOLS -->|Contient| T5
    TOOLS -->|Contient| T6
    
    %% Flux Exécution
    T6 -->|Option 1| SANDBOX
    T6 -->|Option 2| LOCAL
    SANDBOX -->|Exécute| DF
    LOCAL -->|Exécute| DF
    
    %% Flux Données
    CSV -->|Charge| DF
    TOOLS -->|Lit| DF
    T6 -->|Génère| PLOTS
    DF -->|Analyse| TOOLS
    
    %% Flux Configuration
    CONFIG -->|Lit| ENV
    AGENT -->|Utilise| CONFIG
    LLM -->|API Key| ENV
    
    %% Retours
    LLM -->|Réponse| EXEC
    EXEC -->|Résultat| AGENT
    AGENT -->|Réponse| UI1
    AGENT -->|Réponse| API
    PLOTS -->|Affiche| UI1
    
    style AGENT fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style LLM fill:#FF6B6B,stroke:#C92A2A,stroke-width:3px,color:#fff
    style TOOLS fill:#51CF66,stroke:#2F9E44,stroke-width:2px,color:#fff
    style EXEC fill:#FFD43B,stroke:#F59F00,stroke-width:2px,color:#000
    style DF fill:#845EF7,stroke:#5F3DC4,stroke-width:2px,color:#fff
```

## Composants principaux

### 1. **Interface Utilisateur** (Couche Présentation)
- **Streamlit App** (`app.py`) : Interface web interactive avec chat
- **API REST** (`sandbox_api.py`) : Endpoints FastAPI pour intégration
- **CLI** (`csv_agent.py`) : Mode ligne de commande interactif

### 2. **Agent Principal** (Couche Logique Métier)
- **CSVAgent** : Classe principale orchestrant l'analyse
- **AgentExecutor** : Exécuteur LangChain gérant le cycle ReAct
- **Prompt Template** : Instructions détaillées pour l'agent (expert en séries temporelles)

### 3. **Outils d'Analyse** (Couche Fonctionnelle)
- **CSVTools** : Collection d'outils LangChain
  - `get_csv_info` : Informations sur la structure
  - `get_head` : Aperçu des données
  - `get_statistics` : Statistiques descriptives
  - `count_missing_values` : Détection de valeurs manquantes
  - `get_correlation` : Analyse de corrélations
  - `python_code_executor` : Exécution de code Python personnalisé

### 4. **Exécution de Code** (Couche Sécurité)
- **Sandbox API** : Exécution sécurisée via API externe (optionnel)
- **Exécution Locale** : Exécution directe avec environnement isolé

### 5. **Modèle IA** (Couche Intelligence)
- **Google Gemini 2.0 Flash** : Modèle de langage pour compréhension et raisonnement

### 6. **Données** (Couche Persistance)
- **Fichiers CSV/Excel** : Données sources
- **DataFrame Pandas** : Représentation en mémoire
- **Graphiques** : Visualisations Plotly/Matplotlib

## Flux de traitement

### Cycle ReAct (Reasoning + Acting)

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant A as Agent
    participant L as LLM (Gemini)
    participant T as Outils
    
    U->>A: Question en langage naturel
    A->>L: Question + Contexte
    L->>A: Thought: Réflexion
    L->>A: Action: Outil à utiliser
    A->>T: Exécution de l'outil
    T->>A: Observation: Résultat
    A->>L: Thought + Observation
    L->>A: Nouvelle Action (si nécessaire)
    A->>T: Exécution
    T->>A: Observation
    A->>L: Final Thought
    L->>A: Final Answer
    A->>U: Réponse finale
```

## Technologies utilisées

| Composant | Technologie |
|-----------|-------------|
| **Framework IA** | LangChain |
| **Modèle LLM** | Google Gemini 2.0 Flash |
| **Interface Web** | Streamlit |
| **API REST** | FastAPI |
| **Analyse de données** | Pandas, NumPy |
| **Visualisation** | Plotly, Matplotlib |
| **Configuration** | python-dotenv |

## Points clés de l'architecture

1. **Pattern ReAct** : L'agent raisonne avant d'agir, permettant une analyse itérative
2. **Outils modulaires** : Chaque fonction d'analyse est un outil LangChain indépendant
3. **Exécution flexible** : Support de l'exécution locale et sandbox pour la sécurité
4. **Multi-interface** : Support CLI, Web (Streamlit) et API REST
5. **Gestion de sessions** : L'API REST maintient des sessions pour plusieurs utilisateurs
6. **Spécialisation temporelle** : L'agent est spécialement optimisé pour les séries temporelles

## Configuration

Les paramètres sont centralisés dans `config.py` :
- Modèle : `gemini-2.0-flash`
- Max itérations : 5
- Timeout : 30s
- Délai entre appels LLM : 1.5s (pour éviter les erreurs 429)

## Sécurité

- Exécution de code dans un environnement isolé
- Support optionnel d'une API sandbox externe
- Validation des types de fichiers (CSV/Excel uniquement)
- Gestion des erreurs et timeouts

