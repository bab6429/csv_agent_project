"""
Outils personnalisés pour l'analyse de fichiers CSV
"""
import pandas as pd
import numpy as np
import os
import uuid
import json
import matplotlib
matplotlib.use("Agg")  # backend non interactif pour génération de fichiers
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field


class CSVTools:
    """Classe contenant les outils pour analyser un fichier CSV"""
    
    def __init__(self, csv_path: str):
        """
        Initialise les outils avec un fichier CSV
        
        Args:
            csv_path: Chemin vers le fichier CSV
        """
        self.csv_path = csv_path
        # Détecte automatiquement le type de fichier et charge avec pandas
        lower_path = csv_path.lower()
        if lower_path.endswith('.csv'):
            self.df = pd.read_csv(csv_path)
            print(f"✅ CSV chargé : {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        elif lower_path.endswith('.xlsx') or lower_path.endswith('.xls'):
            # Nécessite openpyxl (xlsx) ou xlrd (xls anciennes versions). openpyxl suffit généralement.
            self.df = pd.read_excel(csv_path)
            print(f"✅ Excel chargé : {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        else:
            raise ValueError("Format de fichier non supporté. Utilisez .csv, .xlsx ou .xls")
    
    def get_info(self, query: str = "") -> str:
        """
        Retourne les informations générales sur le DataFrame
        (colonnes, types, dimensions)
        """
        info_str = f"📊 Informations sur le fichier CSV:\n\n"
        info_str += f"- Nombre de lignes : {len(self.df)}\n"
        info_str += f"- Nombre de colonnes : {len(self.df.columns)}\n"
        info_str += f"\nColonnes et types :\n"
        
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            info_str += f"  • {col}: {dtype}\n"
        
        return info_str
    
    def get_head(self, n: str = "5") -> str:
        """
        Retourne les n premières lignes du DataFrame
        
        Args:
            n: Nombre de lignes à afficher (par défaut 5)
        """
        try:
            n_int = int(n)
        except:
            n_int = 5
        
        head_df = self.df.head(n_int)
        return f"📋 Les {n_int} premières lignes :\n\n{head_df.to_string()}"
    
    def get_statistics(self, column: str = "") -> str:
        """
        Calcule les statistiques descriptives
        
        Args:
            column: Nom de la colonne (optionnel). Si vide, calcule pour toutes les colonnes numériques
        """
        if column and column in self.df.columns:
            stats = self.df[column].describe()
            result = f"📊 Statistiques pour '{column}' :\n\n{stats.to_string()}"
            
            # Ajoute des infos supplémentaires
            if self.df[column].dtype in ['int64', 'float64']:
                result += f"\n\nValeurs manquantes : {self.df[column].isna().sum()}"
                result += f"\nValeurs uniques : {self.df[column].nunique()}"
            
            return result
        else:
            # Statistiques pour toutes les colonnes numériques
            stats = self.df.describe()
            return f"📊 Statistiques descriptives :\n\n{stats.to_string()}"
    
    def count_missing(self, query: str = "") -> str:
        """
        Compte les valeurs manquantes dans chaque colonne
        """
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        result = "🔍 Valeurs manquantes :\n\n"
        
        for col in self.df.columns:
            if missing[col] > 0:
                result += f"  • {col}: {missing[col]} ({missing_pct[col]}%)\n"
        
        if missing.sum() == 0:
            result += "  ✅ Aucune valeur manquante !"
        
        total_missing = missing.sum()
        result += f"\nTotal : {total_missing} valeurs manquantes"
        
        return result
    
    def query_data(self, query: str) -> str:
        """
        Exécute une requête sur les données (filtrage, agrégation, etc.)
        
        Args:
            query: Description de la requête en langage naturel
        """
        # Cette fonction sera utilisée par l'agent pour formuler des requêtes plus complexes
        return f"Pour exécuter des requêtes complexes, utilisez l'outil 'python_code_executor' avec du code pandas."
    
    def get_correlation(self, col1: str, col2: str = "") -> str:
        """
        Calcule la corrélation entre deux colonnes ou affiche la matrice de corrélation
        
        Args:
            col1: Première colonne (ou vide pour matrice complète)
            col2: Deuxième colonne (optionnel)
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if not col1 or col1 not in self.df.columns:
            # Matrice de corrélation complète
            corr_matrix = self.df[numeric_cols].corr()
            return f"📈 Matrice de corrélation :\n\n{corr_matrix.to_string()}"
        
        if col2 and col2 in self.df.columns:
            # Corrélation entre deux colonnes spécifiques
            if col1 in numeric_cols and col2 in numeric_cols:
                corr = self.df[col1].corr(self.df[col2])
                return f"📈 Corrélation entre '{col1}' et '{col2}' : {corr:.4f}"
            else:
                return "❌ Les deux colonnes doivent être numériques pour calculer la corrélation"
        
        # Corrélations de col1 avec toutes les autres
        if col1 in numeric_cols:
            corr = self.df[numeric_cols].corr()[col1].sort_values(ascending=False)
            return f"📈 Corrélations de '{col1}' avec les autres colonnes :\n\n{corr.to_string()}"
        else:
            return f"❌ La colonne '{col1}' doit être numérique"
    
    def execute_python_code(self, code: str) -> str:
        """
        Exécute du code Python pour des analyses personnalisées
        ATTENTION : Utiliser avec précaution en production
        
        Args:
            code: Code Python à exécuter (utilisant 'df' comme DataFrame)
        """
        try:
            # Si une API sandbox est configurée, on l'utilise en priorité
            sandbox_url = os.getenv("SANDBOX_API_URL")
            if sandbox_url:
                try:
                    import requests
                    payload = {
                        "code": code,
                        "data_path": self.csv_path,
                        "file_type": None,
                    }
                    resp = requests.post(
                        sandbox_url.rstrip('/') + "/execute",
                        json=payload,
                        timeout=30,
                    )
                    if resp.ok:
                        data = resp.json()
                        text = data.get("result_text") or ""
                        img_b64 = data.get("image_b64")
                        if img_b64:
                            return f"{text}\nPLOT_B64::{img_b64}"
                        return text
                except Exception:
                    # Si l'API échoue, on retombe sur l'exécution locale
                    pass

            # Créer un environnement d'exécution sécurisé
            local_vars = {
                'df': self.df.copy(),
                'pd': pd,
                'np': np,
                'plt': plt,
                'go': go,
                'px': px,
                'plotly': __import__('plotly'),
            }
            
            # Exécuter le code
            plt.close('all')  # nettoie d'éventuelles figures précédentes
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Vérifier si une figure Plotly a été créée (priorité à Plotly pour l'affichage dynamique)
            plotly_fig = None
            if 'fig' in local_vars:
                try:
                    from plotly.graph_objects import Figure
                    if isinstance(local_vars['fig'], Figure):
                        plotly_fig = local_vars['fig']
                        # Vérifier que la figure n'est pas vide
                        if plotly_fig.data and len(plotly_fig.data) > 0:
                            # Vérifier que les données ne sont pas toutes vides/None
                            has_data = False
                            for trace in plotly_fig.data:
                                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
                                    has_data = True
                                    break
                                if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                                    has_data = True
                                    break
                            
                            if not has_data:
                                # La figure est vide, on ne la retourne pas
                                plotly_fig = None
                except Exception as e:
                    plotly_fig = None
            
            if plotly_fig is not None:
                # Sérialiser la figure Plotly en JSON pour transmission
                try:
                    plotly_json = plotly_fig.to_json()
                    return f"📈 Graphique interactif généré\nPLOTLY_JSON::{plotly_json}"
                except Exception as e:
                    # Si la sérialisation échoue, on continue avec matplotlib
                    pass
            
            # Si aucune figure Plotly, vérifier matplotlib (rétrocompatibilité)
            fig = None
            if 'fig' in local_vars:
                try:
                    from matplotlib.figure import Figure
                    if isinstance(local_vars['fig'], Figure):
                        fig = local_vars['fig']
                except Exception:
                    fig = None
            if fig is None:
                # tente de récupérer la figure courante si existante
                fig = plt.gcf() if plt.get_fignums() else None

            if fig is not None and plt.get_fignums():
                try:
                    os.makedirs('plots', exist_ok=True)
                    filename = f"plots/plot_{uuid.uuid4().hex}.png"
                    fig.savefig(filename, bbox_inches='tight')
                    return f"📈 Graphique généré\nPLOT::{filename}"
                except Exception as e:
                    # continue l'extraction des autres résultats si l'enregistrement échoue
                    pass

            # Récupérer le résultat (cherche 'result' dans les variables)
            if 'result' in local_vars:
                result = local_vars['result']
                if isinstance(result, pd.DataFrame):
                    return f"✅ Résultat :\n\n{result.to_string()}"
                elif isinstance(result, pd.Series):
                    return f"✅ Résultat :\n\n{result.to_string()}"
                else:
                    return f"✅ Résultat : {result}"
            else:
                return "✅ Code exécuté avec succès (aucun résultat retourné)"
                
        except Exception as e:
            return f"❌ Erreur lors de l'exécution : {str(e)}"
    
    def get_tools(self) -> list:
        """
        Retourne la liste des outils LangChain pour l'agent
        """
        tools = [
            Tool(
                name="get_csv_info",
                func=self.get_info,
                description="Obtient les informations générales sur le fichier CSV (nombre de lignes, colonnes, types de données). Utilise cet outil pour comprendre la structure du fichier."
            ),
            Tool(
                name="get_head",
                func=self.get_head,
                description="Affiche les n premières lignes du DataFrame. Input: nombre de lignes (ex: '5', '10'). Utile pour voir un aperçu des données."
            ),
            Tool(
                name="get_statistics",
                func=self.get_statistics,
                description="Calcule les statistiques descriptives (moyenne, médiane, écart-type, min, max, quartiles). Input: nom de la colonne (ou vide pour toutes les colonnes numériques)."
            ),
            Tool(
                name="count_missing_values",
                func=self.count_missing,
                description="Compte les valeurs manquantes dans chaque colonne du DataFrame. Utile pour évaluer la qualité des données."
            ),
            Tool(
                name="get_correlation",
                func=self.get_correlation,
                description="Calcule la corrélation entre colonnes numériques. Input: 'col1,col2' pour deux colonnes spécifiques, ou vide pour la matrice complète."
            ),
            Tool(
                name="python_code_executor",
                func=self.execute_python_code,
                description="""Exécute du code Python personnalisé pour des analyses avancées et des graphiques.
                Contexte:
                - DataFrame: df (copie des données)
                - Librairies: pd (pandas), np (numpy), go (plotly.graph_objects), px (plotly.express)
                IMPORTANT POUR LES GRAPHIQUES :
                - OBLIGATOIRE: Utilise UNIQUEMENT Plotly pour créer des graphiques (px ou go)
                - N'utilise JAMAIS matplotlib pour les graphiques
                - Plotly permet un affichage interactif dynamique (zoom, pan, hover)
                Résultats attendus:
                - Pour renvoyer une valeur/tableau: assigne à 'result'
                  ex: result = df[df['prix'] > 100].shape[0]
                - Pour tracer un graphique interactif: utilise Plotly OBLIGATOIREMENT
                  ex histogramme:
                      import plotly.express as px
                      # Vérifier que la colonne existe et filtrer les valeurs manquantes
                      if 'age' in df.columns:
                          df_clean = df[df['age'].notna()]
                          if len(df_clean) > 0:
                              fig = px.histogram(df_clean, x='age', nbins=20, title='Répartition des âges')
                              fig.update_xaxes(title_text='Âge')
                              fig.update_yaxes(title_text='Fréquence')
                              result = 'graph_ok'
                          else:
                              result = 'Aucune donnée disponible'
                      else:
                          result = 'Colonne introuvable. Colonnes: ' + str(list(df.columns))
                  ex ligne/courbe:
                      import plotly.express as px
                      fig = px.line(df, x='date', y='valeur', title='Évolution dans le temps')
                      result = 'graph_ok'
                  ex scatter:
                      import plotly.express as px
                      fig = px.scatter(df, x='x', y='y', title='Nuage de points')
                      result = 'graph_ok'
                  ex barres:
                      import plotly.express as px
                      data = df.groupby('cat')['val'].sum().reset_index()
                      fig = px.bar(data, x='cat', y='val', title='Valeurs par catégorie')
                      result = 'graph_ok'
                  Pour plus de contrôle, utilise plotly.graph_objects (go):
                      import plotly.graph_objects as go
                      fig = go.Figure()
                      fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines'))
                      fig.update_layout(title='Mon graphique')
                      result = 'graph_ok'
                Les figures Plotly sont automatiquement détectées et affichées de manière interactive dans l'interface.
                IMPORTANT: Assigne toujours la figure à 'fig' et assigne result = 'graph_ok' à la fin."""
            ),
        ]
        
        return tools

