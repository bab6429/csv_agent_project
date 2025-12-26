"""
Agent spécialisé dans la visualisation et le tracé de courbes.
Les tools sont déterministes : aucun code libre n'est exécuté.

Le tracé est enregistré dans un registre in-memory et renvoyé via un plot_id
pour affichage côté Streamlit (sans base64 dans le texte).
"""
import json
import time
from typing import Optional, Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from callbacks import LLMIterationCounter
from config import Config
from csv_tools import CSVTools
from llm_factory import get_llm
from plot_registry import store_plot


class DataVizAgent:
    """Agent dédié aux visualisations (courbes, heatmaps, distributions)."""

    # Marqueurs pour que l'UI récupère le plot_id + summary JSON
    PLOT_ID_START = "PLOT_ID_START"
    PLOT_ID_END = "PLOT_ID_END"
    PLOT_SUMMARY_START = "PLOT_SUMMARY_START"
    PLOT_SUMMARY_END = "PLOT_SUMMARY_END"

    def __init__(self, csv_tools: CSVTools, api_key: Optional[str] = None, verbose: bool = True, llm_counter: Optional[dict] = None):
        self.csv_tools = csv_tools
        self.verbose = verbose
        self.last_llm_call_time = 0
        self.callbacks = [LLMIterationCounter(llm_counter)] if llm_counter is not None else None
        # Mapping case-insensible des colonnes
        self.col_lookup = {c.lower(): c for c in self.csv_tools.df.columns}

        self.llm = get_llm(
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_output_tokens=1024,
            max_retries=2,
            api_key=api_key,
            verbose=verbose
        )

        self.tools = self._create_tools()
        self.prompt = self._create_prompt()

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        max_iterations = self._get_max_iterations()

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=(
                "Format invalide. Réponds STRICTEMENT en suivant le format ReAct. "
                "Si tu choisis une Action, n'inclus jamais de Final Answer."
            ),
            max_iterations=max_iterations,
            max_execution_time=Config.MAX_EXECUTION_TIME,
            return_intermediate_steps=True,
            early_stopping_method="force",
            callbacks=self.callbacks,
        )

    def _get_max_iterations(self) -> Optional[int]:
        llm_type = type(self.llm).__name__
        if "GoogleGenerativeAI" in llm_type or "Gemini" in llm_type:
            if self.verbose:
                print(f"🔒 Limite d'itérations Gemini activée: {Config.MAX_ITERATIONS_GEMINI}")
            return Config.MAX_ITERATIONS_GEMINI
        return Config.MAX_ITERATIONS

    def _create_tools(self) -> list:
        return [
            Tool(
                name="get_csv_info",
                func=self.csv_tools.get_info,
                description="Affiche les colonnes, types, dimensions du fichier. Input: vide."
            ),
            Tool(
                name="detect_time_columns",
                func=self.csv_tools.detect_time_columns,
                description="Détecte les colonnes temporelles. Input: vide."
            ),
            Tool(
                name="plot_line",
                func=self.plot_line,
                description="Trace une courbe (x,y). PRÉFÉRÉ pour séries temporelles (ex: volume par jour). Input: 'x_col,y_col[,hue_col]'. Utilise des colonnes réelles.",
                return_direct=True,
            ),
            Tool(
                name="plot_scatter",
                func=self.plot_scatter,
                description="Trace un nuage de points. Input: 'x_col,y_col[,hue_col]'. Colonnes réelles uniquement.",
                return_direct=True,
            ),
            Tool(
                name="plot_bar",
                func=self.plot_bar,
                description="Barres agrégées par catégorie ou temps (ex: volume par jour, ventes par mois). Input: 'category_col,value_col[,agg_func]'. agg_func=sum|mean|min|max|count.",
                return_direct=True,
            ),
            Tool(
                name="plot_hist",
                func=self.plot_hist,
                description="Histogramme de DISTRIBUTION statistique (fréquence des valeurs). NE PAS utiliser pour séries temporelles. Input: 'column[,bins]'.",
                return_direct=True,
            ),
            Tool(
                name="plot_corr_heatmap",
                func=self.plot_corr_heatmap,
                description="Heatmap de corrélation des colonnes numériques. Input: vide.",
                return_direct=True,
            ),
        ]

    def _create_prompt(self):
        template = """Tu es un expert data viz. Tu disposes d'outils de tracé sûrs (pas de code libre).

Outils disponibles:
{tools}

FORMAT STRICT (TRÈS IMPORTANT):
- Tu dois répondre soit avec une Action, soit avec un Final Answer, JAMAIS les deux dans le même message.
- Dans cette application, si la question implique un graphique, tu DOIS utiliser un outil de tracé.
- Comme les outils de tracé sont configurés en return_direct, tu NE DOIS PAS écrire "Final Answer" après avoir choisi une Action.

Format attendu quand tu utilises un outil:
Question: ...
Thought: bref
Action: choix dans [{tool_names}]
Action Input: ...

Règles:
- Réponds en français.
- Utilise des noms de colonnes RÉELS, ne rien inventer.
- IMPORTANT: Pour tracer des valeurs en fonction du temps ou d'une catégorie (ex: 'volume par jour'), utilise 'plot_line' ou 'plot_bar', JAMAIS 'plot_hist'.
- 'plot_hist' est UNIQUEMENT pour montrer la distribution statistique d'une colonne (combien de fois chaque valeur apparaît).
- 1 action suffit généralement.
- Les tools de tracé renvoient directement un plot_id et un summary JSON. Ne ré-écris pas les marqueurs.
- Si tu ne connais pas les colonnes ou si l'utilisateur ne les nomme pas, commence par 'get_csv_info'. Pour les axes temps, utilise 'detect_time_columns' avant de tracer.
- IMPORTANT : Quand l'utilisateur parle de "temps" ou "time", il fait référence à la notion de temps qui passe (Date + Heure), pas forcément à une colonne nommée "Time". Cherche toujours à utiliser une colonne DateTime complète (souvent créée par fusion) plutôt que juste l'heure ou la date seule.
- Évite de demander à l'utilisateur les noms de colonnes : découvre-les avec les outils.
- Si un contexte dataset est fourni (colonnes, types, colonnes temporelles), exploite-le pour choisir x/y par défaut sans reposer des questions.
- IMPORTANT: Si l'utilisateur demande un graphique sur une période ou une catégorie précise, assure-toi que les données ont été filtrées par un agent précédent (Transformation ou TimeSeries). Sinon, tu peux utiliser 'reset_filter' si tu as besoin de revenir au dataset complet.
- Ne filtre PAS les données toi-même, utilise les outils de tracé sur le dataset tel quel (il est déjà filtré si nécessaire).

Question: {input}
Thought: {agent_scratchpad}"""
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        )

    # ===================== TOOLS ===================== #
    def _resolve_col(self, col: Optional[str]) -> Optional[str]:
        if not col:
            return None
        if col in self.csv_tools.df.columns:
            return col
        col_low = col.lower()
        if col_low in self.col_lookup:
            return self.col_lookup[col_low]
        return None

    def _validate_columns(self, cols):
        resolved = []
        missing = []
        for c in cols:
            rc = self._resolve_col(c)
            if c and rc is None:
                missing.append(c)
            resolved.append(rc)
        if missing:
            return None, f"❌ Colonnes introuvables: {', '.join(missing)}"
        return resolved, ""

    def _pack_result(self, kind: str, fig: Any, summary: Dict[str, Any], description: str) -> str:
        plot_id = store_plot(kind=kind, figure=fig, summary=summary)
        summary_json = json.dumps(summary, ensure_ascii=False)
        return (
            f"{description}\n\n"
            f"{self.PLOT_ID_START}\n{plot_id}\n{self.PLOT_ID_END}\n"
            f"{self.PLOT_SUMMARY_START}\n{summary_json}\n{self.PLOT_SUMMARY_END}"
        )

    def _maybe_combine_datetime(self, df: pd.DataFrame, x_col: str) -> Tuple[pd.DataFrame, str]:
        """
        Si le dataset a Date + Time et que x_col est Date ou Time, on combine en une colonne datetime.
        """
        cols_lower = {c.lower(): c for c in df.columns}
        if "date" in cols_lower and "time" in cols_lower:
            date_col = cols_lower["date"]
            time_col = cols_lower["time"]
            if x_col in [date_col, time_col]:
                work = df.copy()
                dt = pd.to_datetime(
                    work[date_col].astype(str) + " " + work[time_col].astype(str),
                    errors="coerce"
                )
                # Fallback si échec (format européen)
                if dt.isna().all():
                    dt = pd.to_datetime(
                        work[date_col].astype(str) + " " + work[time_col].astype(str),
                        errors="coerce",
                        dayfirst=True,
                    )
                work["__datetime__"] = dt
                return work.dropna(subset=["__datetime__"]), "__datetime__"
        return df, x_col

    def _summary_line(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        y = pd.to_numeric(df[y_col], errors="coerce").dropna()
        if y.empty:
            return {"kind": "line", "y_col": y_col, "count": 0}
        ymin = float(y.min())
        ymax = float(y.max())
        ymean = float(y.mean())
        ystd = float(y.std(ddof=1)) if len(y) > 1 else 0.0
        # tendance approximative sur l'index
        yvals = y.values
        x = np.arange(len(yvals))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, yvals) if len(yvals) > 1 else (0, 0, 0, 1, 0)
        return {
            "kind": "line",
            "x_col": x_col,
            "y_col": y_col,
            "count": int(len(yvals)),
            "min": ymin,
            "max": ymax,
            "mean": ymean,
            "std": float(ystd),
            "trend_slope_per_step": float(slope),
            "trend_r2": float(r_value**2),
            "trend_p_value": float(p_value),
            "trend_p_value": float(p_value),
            "data_sample_head": df[[x_col, y_col]].head(5).astype(str).to_dict(orient="records"),
            "data_sample_tail": df[[x_col, y_col]].tail(5).astype(str).to_dict(orient="records"),
            "x_col_eff": x_col,
            "x_col_type": str(df[x_col].dtype),
            "y_col_type": str(df[y_col].dtype)
        }

    def _summary_corr(self, corr: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        # extraire top corr hors diagonale
        cols = list(corr.columns)
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
        pairs_sorted_pos = sorted(pairs, key=lambda t: t[2], reverse=True)[:top_k]
        pairs_sorted_neg = sorted(pairs, key=lambda t: t[2])[:top_k]
        return {
            "kind": "corr_heatmap",
            "n_cols": int(len(cols)),
            "top_positive": [{"a": a, "b": b, "corr": v} for a, b, v in pairs_sorted_pos],
            "top_negative": [{"a": a, "b": b, "corr": v} for a, b, v in pairs_sorted_neg],
        }

    def plot_line(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 2:
            # Fallback : deviner une colonne temps + une colonne numérique
            time_cols = [c for c in self.csv_tools.df.columns if pd.api.types.is_datetime64_any_dtype(self.csv_tools.df[c])]
            num_cols = list(self.csv_tools.df.select_dtypes(include="number").columns)
            if time_cols and num_cols:
                x_col, y_col = time_cols[0], num_cols[0]
                hue_col = None
            else:
                return "❌ Fournis au moins x_col,y_col. Format: 'x_col,y_col[,hue_col]'. Utilise d'abord get_csv_info/detect_time_columns si besoin."
        else:
            x_col, y_col = parts[0], parts[1]
            hue_col = parts[2] if len(parts) > 2 else None

        (x_col_res, y_col_res, hue_col_res), err = self._validate_columns([x_col, y_col, hue_col])
        if err:
            return err

        x_col = x_col_res or x_col
        y_col = y_col_res or y_col
        hue_col = hue_col_res or hue_col

        df = self.csv_tools.df.dropna(subset=[y_col]).copy()
        df, x_col_eff = self._maybe_combine_datetime(df, x_col)
        df = df.dropna(subset=[x_col_eff])

        # S'assurer que y est numérique
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.dropna(subset=[y_col])

        # Trier par x pour éviter les segments incohérents
        try:
            df = df.sort_values(by=x_col_eff).reset_index(drop=True)
        except Exception:
            pass

        # Éviter de créer des milliers de courbes si hue a trop de valeurs (ex: Time)
        if hue_col and hue_col in df.columns:
            try:
                if df[hue_col].nunique(dropna=True) > 50:
                    hue_col = None
            except Exception:
                hue_col = None

        fig = px.line(
            df,
            x=x_col_eff,
            y=y_col,
            color=hue_col,
            title=f"{y_col} en fonction de {x_col_eff}",
        )
        summary = self._summary_line(df, x_col_eff, y_col)
        # Verrouiller les axes pour éviter des rendus aberrants (ex: y interprété comme index)
        try:
            if summary.get("count", 0) and "min" in summary and "max" in summary:
                fig.update_yaxes(range=[summary["min"] - 1, summary["max"] + 1], title=y_col)
            fig.update_xaxes(title="Temps")
            fig.update_layout(title=f"{y_col} en fonction du temps")
        except Exception:
            pass
        return self._pack_result("line", fig, summary, f"📈 Courbe {y_col} vs {x_col_eff}")

    def plot_scatter(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 2:
            return "❌ Fournis au moins x_col,y_col. Format: 'x_col,y_col[,hue_col]'."
        x_col, y_col = parts[0], parts[1]
        hue_col = parts[2] if len(parts) > 2 else None

        (x_col_res, y_col_res, hue_col_res), err = self._validate_columns([x_col, y_col, hue_col])
        if err:
            return err
        x_col = x_col_res or x_col
        y_col = y_col_res or y_col
        hue_col = hue_col_res or hue_col

        df = self.csv_tools.df.dropna(subset=[x_col, y_col]).copy()
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, title=f"{y_col} vs {x_col}")
        corr = float(pd.to_numeric(df[x_col], errors="coerce").corr(pd.to_numeric(df[y_col], errors="coerce")))
        summary = {"kind": "scatter", "x_col": x_col, "y_col": y_col, "count": int(len(df)), "corr": corr}
        return self._pack_result("scatter", fig, summary, "📊 Nuage de points")

    def plot_bar(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 2:
            return "❌ Format: 'category_col,value_col[,agg_func]'."
        cat_col, val_col = parts[0], parts[1]
        agg_func = parts[2].lower() if len(parts) > 2 else "sum"

        (cat_col_res, val_col_res), err = self._validate_columns([cat_col, val_col])
        if err:
            return err
        cat_col = cat_col_res or cat_col
        val_col = val_col_res or val_col
        if agg_func not in ["sum", "mean", "min", "max", "count"]:
            agg_func = "sum"

        df = self.csv_tools.df.dropna(subset=[cat_col, val_col])
        grouped = df.groupby(cat_col)[val_col]
        agg_map = {
            "sum": grouped.sum(),
            "mean": grouped.mean(),
            "min": grouped.min(),
            "max": grouped.max(),
            "count": grouped.count(),
        }
        series = agg_map[agg_func]
        
        # Détecter si cat_col est temporel ou numérique pour le tri
        is_time = pd.api.types.is_datetime64_any_dtype(df[cat_col]) or "date" in cat_col.lower()
        if is_time:
            # Trier par date (index)
            series = series.sort_index()
        else:
            # Trier par valeur décroissante (comportement par défaut pour catégories)
            series = series.sort_values(ascending=False)
            
        bar_df = series.reset_index()
        bar_df.columns = [cat_col, val_col]
        fig = px.bar(bar_df, x=cat_col, y=val_col, title=f"{val_col} par {cat_col} ({agg_func})")
        top = bar_df.head(5).to_dict(orient="records")
        summary = {"kind": "bar", "category_col": cat_col, "value_col": val_col, "agg": agg_func, "top_5": top}
        return self._pack_result("bar", fig, summary, f"📊 Barres ({agg_func})")

    def _format_number(self, num: float) -> str:
        """Formate un nombre de manière lisible (K, M, B)."""
        abs_num = abs(num)
        if abs_num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif abs_num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs_num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.1f}"
    
    def plot_hist(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 1:
            return "❌ Format: 'column[,bins]'."
        col = parts[0]
        bins = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30

        (col_res,), err = self._validate_columns([col])
        if err:
            return err
        col = col_res or col
        if not pd.api.types.is_numeric_dtype(self.csv_tools.df[col]):
            return f"❌ La colonne '{col}' doit être numérique."

        series = pd.to_numeric(self.csv_tools.df[col], errors="coerce").dropna()
        
        # Calculer l'histogramme manuellement avec numpy
        counts, bin_edges = np.histogram(series, bins=bins)
        
        # Créer des labels lisibles pour les bins (plages)
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            start = self._format_number(bin_edges[i])
            end = self._format_number(bin_edges[i + 1])
            bin_labels.append(f"{start}-{end}")
        
        # Créer une figure Plotly en barres avec des labels textuels
        fig = go.Figure(data=[go.Bar(
            x=bin_labels,
            y=counts,
            text=counts,
            textposition='auto'
        )])
        fig.update_layout(
            title=f"Distribution de {col}",
            xaxis_title=f"Plages de {col}",
            yaxis_title="Fréquence",
            xaxis={'tickangle': -45}  # Rotation des labels pour meilleure lisibilité
        )
        
        q = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        summary = {
            "kind": "hist", 
            "col": col, 
            "count": int(series.shape[0]), 
            "mean": float(series.mean()), 
            "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0, 
            "quantiles": {str(k): float(v) for k, v in q.items()},
            "bins": int(bins),
            "bin_edges": [float(x) for x in bin_edges[:10]],  # Limiter pour le JSON
            "bin_labels": bin_labels[:10]  # Ajouter les labels pour référence
        }
        return self._pack_result("hist", fig, summary, f"📊 Histogramme de {col}")

    def plot_corr_heatmap(self, input_str: str = "") -> str:
        numeric_df = self.csv_tools.df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return "❌ Pas assez de colonnes numériques pour une matrice de corrélation."
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Matrice de corrélation")
        summary = self._summary_corr(corr, top_k=5)
        return self._pack_result("corr_heatmap", fig, summary, "📈 Heatmap de corrélation")

    # ===================== PUBLIC ===================== #
    def query(self, question: str) -> str:
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        if time_since_last_call < Config.LLM_REQUEST_DELAY:
            delay_needed = Config.LLM_REQUEST_DELAY - time_since_last_call
            time.sleep(delay_needed)
        self.last_llm_call_time = time.time()

        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.agent_executor.invoke({"input": question})
                return response.get("output", "")
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str or "resource exhausted" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if self.verbose:
                            print(f"⚠️ Erreur 429. Nouvelle tentative dans {delay}s...")
                        time.sleep(delay)
                        self.last_llm_call_time = time.time()
                        continue
                    else:
                        return "❌ Erreur : Limite de taux API atteinte. Veuillez patienter."
                else:
                    return f"❌ Erreur : {error_str}"
        return "❌ Erreur : Échec après plusieurs tentatives"

