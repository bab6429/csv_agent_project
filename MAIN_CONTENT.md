# Contenu Principal du Projet - CSV Agent Multi-Agents

## 🎯 À Quoi Sert Ce Projet ?

**CSV Agent** est un système d'analyse de données intelligent qui permet d'interroger vos fichiers CSV et Excel en **langage naturel**, comme si vous parliez à un analyste de données. Au lieu d'écrire du code Python ou des formules Excel complexes, vous posez simplement vos questions en français : *"Quelle est la tendance des ventes ce mois-ci ?"*, *"Montre-moi un graphique des prix dans le temps"*, ou *"Y a-t-il des corrélations entre mes colonnes ?"*. Le système comprend votre intention, analyse vos données, et vous fournit des réponses claires accompagnées de visualisations interactives.

## 🤖 Les Différents Types d'Agents

Le projet utilise une **architecture multi-agents** où chaque agent est un expert spécialisé dans un domaine particulier. Voici les quatre agents principaux :

### 1. **OrchestratorAgent** - Le Chef d'Orchestre
C'est le point d'entrée de toutes vos questions. Il joue le rôle de coordinateur intelligent : il analyse votre demande, comprend ce dont vous avez besoin, et décide automatiquement quel(s) agent(s) spécialisé(s) doit intervenir et dans quel ordre. Par exemple, si vous demandez *"Trace une courbe des ventes et analyse la tendance"*, il planifiera une séquence de 3 étapes : d'abord identifier les bonnes colonnes, ensuite créer le graphique, puis générer une analyse textuelle.

### 2. **TransformationAgent** - L'Expert en Structure de Données
Cet agent est votre allié pour comprendre et explorer la structure de vos données. Il répond aux questions sur la composition de votre fichier : combien de lignes et colonnes, quels sont les types de données, y a-t-il des valeurs manquantes, quelles sont les statistiques (moyenne, médiane, écart-type), et comment les colonnes sont corrélées entre elles. C'est l'agent parfait pour une première exploration ou pour obtenir un aperçu rapide de vos données.

### 3. **TimeSeriesAgent** - Le Spécialiste Temporel
Dès que vos données contiennent une dimension temporelle (dates, heures, timestamps), cet agent entre en jeu. Il détecte automatiquement les colonnes de temps, calcule les tendances (croissance ou décroissance), génère des moyennes mobiles pour lisser les variations, agrège vos données par période (jour, semaine, mois, année), et peut même détecter des anomalies dans vos séries temporelles. Idéal pour analyser des ventes, des métriques de performance, ou tout phénomène évoluant dans le temps.

### 4. **DataVizAgent** - Le Créateur de Visualisations
Cet agent transforme vos données en graphiques interactifs et professionnels. Il peut créer des courbes (pour les évolutions temporelles), des nuages de points (pour les relations entre variables), des histogrammes (pour les distributions), des graphiques en barres (pour les comparaisons), et des heatmaps de corrélation (pour visualiser les relations entre toutes vos colonnes). Tous les graphiques sont générés avec Plotly, ce qui les rend interactifs : vous pouvez zoomer, survoler les points pour voir les valeurs exactes, et exporter les images.

### 5. **PlotCommentaryAgent** - L'Analyste Visuel
Une fois qu'un graphique est créé, cet agent l'analyse et génère un commentaire textuel intelligent. Il identifie les tendances principales, les valeurs extrêmes, les patterns intéressants, et explique ce que le graphique révèle par rapport à votre question initiale. C'est comme avoir un analyste qui regarde le graphique avec vous et vous explique ce qu'il en pense.

## 💬 Comment Interagir avec le Système ?

Vous avez **deux façons principales** d'utiliser CSV Agent :

### Interface Web (Streamlit)
L'interface la plus conviviale : vous uploadez votre fichier CSV ou Excel via un simple glisser-déposer, puis vous chattez avec l'agent dans une interface de messagerie. Vous tapez vos questions en français, et les réponses apparaissent instantanément avec les graphiques affichés directement dans le navigateur. Parfait pour une utilisation interactive et exploratoire.

### Interface Programmation (Python)
Pour les développeurs ou les cas d'usage automatisés, vous pouvez intégrer CSV Agent directement dans votre code Python. Vous créez une instance de l'agent avec votre fichier, puis vous appelez la méthode `query()` avec vos questions. Les réponses sont retournées sous forme de texte, et les graphiques sont sauvegardés dans un dossier. Idéal pour des pipelines d'analyse automatisés ou des notebooks Jupyter.

## 📥 Entrées Acceptées

Le système accepte :
- **Fichiers** : CSV (avec différents séparateurs : virgule, point-virgule, tabulation) et Excel (.xlsx, .xls)
- **Encodages** : Détection automatique (UTF-8, Latin-1, etc.)
- **Questions** : Texte libre en français, en langage naturel
- **Types de données** : Numériques (entiers, décimaux), texte, dates/heures, booléens

## 📤 Sorties Produites

Le système génère :
- **Réponses textuelles** : Analyses, statistiques, explications en français
- **Graphiques interactifs** : Fichiers HTML (Plotly) ou images PNG (Matplotlib)
- **Données structurées** : Tableaux de statistiques, matrices de corrélation
- **Insights** : Tendances, anomalies, patterns détectés automatiquement

## 🎯 Exemples de Cas d'Usage

### Cas 1 : Exploration Initiale
**Situation** : Vous venez de recevoir un nouveau fichier de données et vous ne savez pas ce qu'il contient.
**Questions** : *"Quelle est la structure de ce fichier ?"*, *"Montre-moi les 10 premières lignes"*, *"Y a-t-il des valeurs manquantes ?"*
**Résultat** : Vous obtenez un aperçu complet de vos données en quelques secondes sans écrire une ligne de code.

### Cas 2 : Analyse de Ventes
**Situation** : Vous gérez un e-commerce et voulez comprendre l'évolution de vos ventes.
**Questions** : *"Quelle est la tendance des ventes sur les 6 derniers mois ?"*, *"Calcule la moyenne mobile sur 7 jours"*, *"Agrège les ventes par semaine et trace une courbe"*
**Résultat** : Graphiques de tendance avec analyse automatique identifiant les périodes de croissance, les pics, et les creux.

### Cas 3 : Analyse Financière
**Situation** : Vous analysez des données boursières ou financières.
**Questions** : *"Trace l'évolution du prix de l'action Apple"*, *"Y a-t-il une corrélation entre le volume et le prix ?"*, *"Détecte les anomalies dans les variations de prix"*
**Résultat** : Visualisations professionnelles avec heatmap de corrélation et détection automatique des mouvements inhabituels.

### Cas 4 : Reporting Automatisé
**Situation** : Vous devez générer des rapports hebdomadaires sur des KPIs.
**Questions** : *"Quelles sont les statistiques de la colonne 'Chiffre d'affaires' ?"*, *"Compare les performances par région"*, *"Montre un histogramme de la distribution des âges clients"*
**Résultat** : Statistiques détaillées et graphiques prêts à être intégrés dans vos présentations.

### Cas 5 : Analyse de Corrélations
**Situation** : Vous cherchez à comprendre quelles variables influencent vos résultats.
**Questions** : *"Montre-moi la matrice de corrélation de toutes les colonnes numériques"*, *"Quelles colonnes sont les plus corrélées avec les ventes ?"*
**Résultat** : Heatmap colorée montrant visuellement les relations entre toutes vos variables, avec analyse textuelle des corrélations fortes.

### Cas 6 : Analyse Temporelle Complexe
**Situation** : Vous avez des données avec des colonnes date et heure séparées.
**Questions** : *"Fusionne les colonnes date et heure en une seule colonne temporelle"*, *"Agrège les données par mois et calcule la moyenne"*, *"Y a-t-il une saisonnalité ?"*
**Résultat** : Le système prépare automatiquement vos données temporelles et génère des analyses de tendance et de saisonnalité.

## 🌟 Pourquoi Utiliser CSV Agent ?

- **Gain de temps** : Plus besoin d'écrire du code pour des analyses courantes
- **Accessibilité** : Utilisable par des non-programmeurs grâce au langage naturel
- **Intelligence** : Comprend le contexte et choisit automatiquement les bonnes méthodes
- **Visualisations professionnelles** : Graphiques interactifs de qualité sans effort
- **Flexibilité** : S'adapte à différents types de données et questions
- **Extensible** : Architecture modulaire permettant d'ajouter facilement de nouveaux agents

## 🚀 En Résumé

CSV Agent transforme l'analyse de données d'une tâche technique en une conversation naturelle. Que vous soyez analyste de données cherchant à gagner du temps, manager ayant besoin d'insights rapides, ou développeur voulant automatiser des analyses, ce système s'adapte à vos besoins. Il combine la puissance de l'intelligence artificielle (modèles de langage) avec des outils d'analyse de données éprouvés (Pandas, Plotly) pour vous offrir une expérience d'analyse intuitive et efficace.
