"""
Outils personnalisés pour l'analyse de fichiers CSV
"""
import sys
import warnings
import pandas as pd
import numpy as np
from langchain.tools import Tool
from scipy import stats
from datetime import datetime

# Configuration de l'encodage UTF-8 pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback pour les anciennes versions de Python
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


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
    
    # ==================== OUTILS POUR SÉRIES TEMPORELLES ====================
    
    def detect_time_columns(self, query: str = "") -> str:
        """
        Détecte automatiquement les colonnes contenant des dates/timestamps
        
        Returns:
            Liste des colonnes temporelles détectées
        """
        time_columns = []
        
        for col in self.df.columns:
            # Vérifier si le type est déjà datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                time_columns.append(col)
                continue
            
            # Vérifier si le nom de la colonne suggère une date
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'jour', 'mois', 'année', 'year', 'month', 'day']):
                # Essayer de convertir en datetime
                try:
                    # Supprimer le warning de format inference
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
                        pd.to_datetime(self.df[col].dropna().head(10), errors='coerce')
                    time_columns.append(col)
                except:
                    pass
        
        if time_columns:
            result = f"📅 Colonnes temporelles détectées :\n\n"
            for col in time_columns:
                dtype = self.df[col].dtype
                sample = self.df[col].dropna().head(3).tolist()
                result += f"  • {col} (type: {dtype})\n"
                result += f"    Exemples: {sample}\n"
            return result
        else:
            return "❌ Aucune colonne temporelle détectée. Vérifiez que vos colonnes de dates sont au bon format."
    
    def calculate_trend(self, input_str: str = "") -> str:
        """
        Calcule la tendance (croissance/décroissance) d'une série temporelle
        
        Args:
            input_str: Format "column" ou "column,time_column"
        """
        parts = [p.strip() for p in input_str.split(',')] if input_str else []
        column = parts[0] if len(parts) > 0 else ""
        time_column = parts[1] if len(parts) > 1 else ""
        
        if not column:
            return "❌ Spécifiez au moins le nom de la colonne. Format: 'column' ou 'column,time_column'"
        
        if column not in self.df.columns:
            return f"❌ Colonne '{column}' introuvable"
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"❌ La colonne '{column}' doit être numérique"
        
        # Détecter la colonne temporelle si non fournie
        if not time_column:
            time_cols = [col for col in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[col])]
            if not time_cols:
                # Essayer de détecter automatiquement
                for col in self.df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            self.df[col] = pd.to_datetime(self.df[col])
                            time_cols = [col]
                            break
                        except:
                            pass
            
            if not time_cols:
                return "❌ Aucune colonne temporelle trouvée. Spécifiez-la avec le paramètre time_column."
            time_column = time_cols[0]
        
        # Préparer les données
        df_clean = self.df[[time_column, column]].dropna()
        if len(df_clean) < 2:
            return "❌ Pas assez de données pour calculer une tendance"
        
        # Convertir en datetime si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df_clean[time_column]):
            try:
                df_clean[time_column] = pd.to_datetime(df_clean[time_column])
            except:
                return f"❌ Impossible de convertir '{time_column}' en datetime"
        
        # Trier par date
        df_clean = df_clean.sort_values(time_column)
        
        # Calculer la tendance avec régression linéaire
        x = np.arange(len(df_clean))
        y = df_clean[column].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Interprétation
        if slope > 0:
            direction = "📈 Croissance"
        elif slope < 0:
            direction = "📉 Décroissance"
        else:
            direction = "➡️ Stable"
        
        result = f"📊 Analyse de tendance pour '{column}' :\n\n"
        result += f"{direction}\n"
        result += f"Pente (tendance) : {slope:.4f}\n"
        result += f"Coefficient de corrélation (R²) : {r_value**2:.4f}\n"
        result += f"P-valeur : {p_value:.4f}\n"
        
        if p_value < 0.05:
            result += "✅ Tendance statistiquement significative (p < 0.05)\n"
        else:
            result += "⚠️ Tendance non significative statistiquement (p >= 0.05)\n"
        
        # Calculer la variation totale
        first_val = df_clean[column].iloc[0]
        last_val = df_clean[column].iloc[-1]
        variation = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
        
        result += f"\nValeur initiale : {first_val:.2f}\n"
        result += f"Valeur finale : {last_val:.2f}\n"
        result += f"Variation totale : {variation:+.2f}%"
        
        return result
    
    def calculate_moving_average(self, input_str: str = "") -> str:
        """
        Calcule la moyenne mobile d'une série temporelle
        
        Args:
            input_str: Format "column,window" ou "column,window,time_column"
        """
        parts = [p.strip() for p in input_str.split(',')] if input_str else []
        column = parts[0] if len(parts) > 0 else ""
        window = parts[1] if len(parts) > 1 else "7"
        time_column = parts[2] if len(parts) > 2 else ""
        
        if not column:
            return "❌ Spécifiez au moins le nom de la colonne. Format: 'column,window' ou 'column,window,time_column'"
        
        if column not in self.df.columns:
            return f"❌ Colonne '{column}' introuvable"
        
        try:
            window_int = int(window)
        except:
            window_int = 7
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"❌ La colonne '{column}' doit être numérique"
        
        # Détecter la colonne temporelle si nécessaire
        if time_column and time_column in self.df.columns:
            df_work = self.df[[time_column, column]].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_work[time_column]):
                try:
                    df_work[time_column] = pd.to_datetime(df_work[time_column])
                except:
                    pass
            df_work = df_work.sort_values(time_column)
            series = df_work[column]
        else:
            series = self.df[column].sort_index()
        
        # Calculer la moyenne mobile
        ma = series.rolling(window=window_int, min_periods=1).mean()
        
        result = f"📈 Moyenne mobile ({window_int} périodes) pour '{column}' :\n\n"
        result += f"Valeurs calculées : {len(ma.dropna())} / {len(ma)}\n\n"
        result += "Dernières valeurs :\n"
        result += pd.DataFrame({
            'Valeur originale': series.tail(10),
            f'MA({window_int})': ma.tail(10)
        }).to_string()
        
        return result
    
    def aggregate_by_period(self, input_str: str = "") -> str:
        """
        Agrège les données par période (jour, semaine, mois, année)
        
        Args:
            input_str: Format "column,period,time_column,agg_func" (period et time_column optionnels)
        """
        parts = [p.strip() for p in input_str.split(',')] if input_str else []
        column = parts[0] if len(parts) > 0 else ""
        period = parts[1] if len(parts) > 1 else "M"
        time_column = parts[2] if len(parts) > 2 else ""
        agg_func = parts[3] if len(parts) > 3 else "mean"
        
        if not column:
            return "❌ Spécifiez au moins le nom de la colonne. Format: 'column,period,time_column,agg_func'"
        
        if column not in self.df.columns:
            return f"❌ Colonne '{column}' introuvable"
        
        # Détecter la colonne temporelle
        if not time_column:
            time_cols = [col for col in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[col])]
            if not time_cols:
                for col in self.df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            self.df[col] = pd.to_datetime(self.df[col])
                            time_cols = [col]
                            break
                        except:
                            pass
            
            if not time_cols:
                return "❌ Aucune colonne temporelle trouvée"
            time_column = time_cols[0]
        
        # Préparer les données
        df_work = self.df[[time_column, column]].copy()
        if not pd.api.types.is_datetime64_any_dtype(df_work[time_column]):
            try:
                df_work[time_column] = pd.to_datetime(df_work[time_column])
            except:
                return f"❌ Impossible de convertir '{time_column}' en datetime"
        
        df_work = df_work.dropna().sort_values(time_column)
        df_work = df_work.set_index(time_column)
        
        # Mapper les fonctions d'agrégation
        agg_map = {
            'mean': 'mean',
            'sum': 'sum',
            'min': 'min',
            'max': 'max',
            'count': 'count',
            'moyenne': 'mean',
            'somme': 'sum',
            'minimum': 'min',
            'maximum': 'max',
            'compte': 'count'
        }
        agg_func_clean = agg_map.get(agg_func.lower(), 'mean')
        
        # Mapper les périodes
        period_map = {
            'jour': 'D', 'day': 'D', 'd': 'D',
            'semaine': 'W', 'week': 'W', 'w': 'W',
            'mois': 'M', 'month': 'M', 'm': 'M',
            'trimestre': 'Q', 'quarter': 'Q', 'q': 'Q',
            'année': 'Y', 'year': 'Y', 'y': 'Y'
        }
        period_clean = period_map.get(period.lower(), period.upper())
        
        # Agréger
        aggregated = df_work[column].resample(period_clean).agg(agg_func_clean)
        
        period_names = {
            'D': 'jour',
            'W': 'semaine',
            'M': 'mois',
            'Q': 'trimestre',
            'Y': 'année'
        }
        period_name = period_names.get(period_clean, period_clean)
        
        result = f"📊 Agrégation par {period_name} ({agg_func_clean}) pour '{column}' :\n\n"
        result += aggregated.to_string()
        result += f"\n\nNombre de périodes : {len(aggregated)}"
        
        return result
    
    def detect_anomalies(self, input_str: str = "") -> str:
        """
        Détecte les anomalies dans une série temporelle
        
        Args:
            input_str: Format "column,method,threshold" (method et threshold optionnels)
        """
        parts = [p.strip() for p in input_str.split(',')] if input_str else []
        column = parts[0] if len(parts) > 0 else ""
        method = parts[1] if len(parts) > 1 else "iqr"
        threshold = parts[2] if len(parts) > 2 else "3"
        
        if not column:
            return "❌ Spécifiez au moins le nom de la colonne. Format: 'column,method,threshold'"
        
        if column not in self.df.columns:
            return f"❌ Colonne '{column}' introuvable"
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"❌ La colonne '{column}' doit être numérique"
        
        series = self.df[column].dropna()
        
        if method.lower() in ['iqr', 'interquartile']:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = series[(series < lower_bound) | (series > upper_bound)]
            method_name = "IQR (Interquartile Range)"
        else:  # zscore
            try:
                threshold_float = float(threshold)
            except:
                threshold_float = 3.0
            
            z_scores = np.abs(stats.zscore(series))
            anomalies = series[z_scores > threshold_float]
            method_name = f"Z-score (seuil: {threshold_float})"
        
        result = f"🔍 Détection d'anomalies ({method_name}) pour '{column}' :\n\n"
        result += f"Nombre d'anomalies détectées : {len(anomalies)} / {len(series)} ({len(anomalies)/len(series)*100:.2f}%)\n\n"
        
        if len(anomalies) > 0:
            result += "Anomalies détectées :\n"
            result += anomalies.to_string()
        else:
            result += "✅ Aucune anomalie détectée"
        
        return result
    
    def calculate_growth_rate(self, input_str: str = "") -> str:
        """
        Calcule le taux de croissance entre périodes
        
        Args:
            input_str: Format "column,time_column,period" (time_column et period optionnels)
        """
        parts = [p.strip() for p in input_str.split(',')] if input_str else []
        column = parts[0] if len(parts) > 0 else ""
        time_column = parts[1] if len(parts) > 1 else ""
        period = parts[2] if len(parts) > 2 else "1"
        
        if not column:
            return "❌ Spécifiez au moins le nom de la colonne. Format: 'column,time_column,period'"
        
        if column not in self.df.columns:
            return f"❌ Colonne '{column}' introuvable"
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"❌ La colonne '{column}' doit être numérique"
        
        try:
            period_int = int(period)
        except:
            period_int = 1
        
        # Détecter la colonne temporelle si nécessaire
        if time_column and time_column in self.df.columns:
            df_work = self.df[[time_column, column]].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_work[time_column]):
                try:
                    df_work[time_column] = pd.to_datetime(df_work[time_column])
                except:
                    pass
            df_work = df_work.sort_values(time_column)
            series = df_work[column].reset_index(drop=True)
        else:
            series = self.df[column].sort_index().reset_index(drop=True)
        
        # Calculer le taux de croissance
        growth_rate = series.pct_change(periods=period_int) * 100
        
        result = f"📈 Taux de croissance ({period_int} période(s)) pour '{column}' :\n\n"
        result += f"Valeurs calculées : {len(growth_rate.dropna())} / {len(growth_rate)}\n\n"
        result += "Dernières valeurs :\n"
        
        df_result = pd.DataFrame({
            'Valeur': series.tail(15),
            f'Croissance (%)': growth_rate.tail(15)
        })
        df_result = df_result[df_result[f'Croissance (%)'].notna()]
        result += df_result.to_string()
        
        # Statistiques
        growth_clean = growth_rate.dropna()
        if len(growth_clean) > 0:
            result += f"\n\n📊 Statistiques de croissance :\n"
            result += f"Moyenne : {growth_clean.mean():.2f}%\n"
            result += f"Médiane : {growth_clean.median():.2f}%\n"
            result += f"Min : {growth_clean.min():.2f}%\n"
            result += f"Max : {growth_clean.max():.2f}%"
        
        return result
    
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
            # Outils pour séries temporelles
            Tool(
                name="detect_time_columns",
                func=self.detect_time_columns,
                description="Détecte automatiquement les colonnes contenant des dates/timestamps dans le fichier CSV. Utile pour identifier les colonnes temporelles avant d'effectuer des analyses de séries temporelles. Input: vide."
            ),
            Tool(
                name="calculate_trend",
                func=self.calculate_trend,
                description="Calcule la tendance (croissance/décroissance) d'une série temporelle. Détecte automatiquement la colonne temporelle si non spécifiée. Input: 'column' ou 'column,time_column'. Exemple: 'ventes' ou 'ventes,date'."
            ),
            Tool(
                name="calculate_moving_average",
                func=self.calculate_moving_average,
                description="Calcule la moyenne mobile d'une série temporelle pour lisser les données. Input: 'column,window' ou 'column,window,time_column'. Exemple: 'ventes,7' pour une moyenne mobile sur 7 périodes."
            ),
            Tool(
                name="aggregate_by_period",
                func=self.aggregate_by_period,
                description="Agrège les données par période (jour=D, semaine=W, mois=M, trimestre=Q, année=Y). Input: 'column,period,time_column,agg_func'. Exemple: 'ventes,M,date,sum' pour sommer les ventes par mois. agg_func peut être: mean, sum, min, max, count."
            ),
            Tool(
                name="detect_anomalies",
                func=self.detect_anomalies,
                description="Détecte les anomalies/outliers dans une série temporelle. Input: 'column,method,threshold'. method peut être 'iqr' ou 'zscore'. Exemple: 'ventes,iqr' ou 'ventes,zscore,3'."
            ),
            Tool(
                name="calculate_growth_rate",
                func=self.calculate_growth_rate,
                description="Calcule le taux de croissance en pourcentage entre périodes. Input: 'column,time_column,period'. Exemple: 'ventes,date,1' pour la croissance période par période, ou 'ventes,date,12' pour la croissance sur 12 périodes."
            ),
        ]
        
        return tools

