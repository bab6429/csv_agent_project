"""
Agent spécialisé dans la visualisation et le tracé de courbes.
Les outils sont déterministes : aucun code libre n'est exécuté.
"""
import io
import base64
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from callbacks import LLMIterationCounter
from config import Config
from csv_tools import CSVTools
from llm_factory import get_llm


def _fig_to_base64(fig) -> str:
    """Convertit une figure matplotlib en base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


class DataVizAgent:
    """Agent dédié aux visualisations (courbes, heatmaps, distributions)."""

    MARKER_START = "__PLOT_BASE64_START__"
    MARKER_END = "__PLOT_BASE64_END__"

    def __init__(self, csv_tools: CSVTools, api_key: Optional[str] = None, verbose: bool = True, llm_counter: Optional[dict] = None):
        self.csv_tools = csv_tools
        self.verbose = verbose
        self.last_llm_call_time = 0
        self.callbacks = [LLMIterationCounter(llm_counter)] if llm_counter is not None else None

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
            handle_parsing_errors=True,
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
                name="plot_line",
                func=self.plot_line,
                description="Trace une courbe (x,y). Input: 'x_col,y_col[,hue_col]'. Utilise des colonnes réelles."
            ),
            Tool(
                name="plot_scatter",
                func=self.plot_scatter,
                description="Trace un nuage de points. Input: 'x_col,y_col[,hue_col]'. Colonnes réelles uniquement."
            ),
            Tool(
                name="plot_bar",
                func=self.plot_bar,
                description="Barres agrégées par catégorie. Input: 'category_col,value_col[,agg_func]'. agg_func=sum|mean|min|max|count."
            ),
            Tool(
                name="plot_hist",
                func=self.plot_hist,
                description="Histogramme d'une colonne numérique. Input: 'column[,bins]'."
            ),
            Tool(
                name="plot_corr_heatmap",
                func=self.plot_corr_heatmap,
                description="Heatmap de corrélation des colonnes numériques. Input: vide."
            ),
        ]

    def _create_prompt(self):
        template = """Tu es un expert data viz. Tu disposes d'outils de tracé sûrs (pas de code libre).

Outils disponibles:
{tools}

Format:
Question: ...
Thought: bref
Action: choix dans [{tool_names}]
Action Input: ...
Observation: résultat
Thought: réponse finale prête
Final Answer: description courte du graphique + inclure le payload base64 si fourni dans l'Observation.

Règles:
- Réponds en français.
- Utilise des noms de colonnes RÉELS, ne rien inventer.
- 1 action suffit généralement.
- Si l'Observation contient un payload base64 entre {marker_start} et {marker_end}, garde-le dans la réponse finale pour affichage.

Question: {input}
Thought: {agent_scratchpad}"""
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            partial_variables={
                "marker_start": self.MARKER_START,
                "marker_end": self.MARKER_END,
            },
        )

    # ===================== TOOLS ===================== #
    def _validate_columns(self, cols):
        missing = [c for c in cols if c and c not in self.csv_tools.df.columns]
        if missing:
            return f"❌ Colonnes introuvables: {', '.join(missing)}"
        return ""

    def _payload(self, description: str, fig) -> str:
        b64 = _fig_to_base64(fig)
        return f"{description}\n\n{self.MARKER_START}\n{b64}\n{self.MARKER_END}"

    def plot_line(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 2:
            return "❌ Fournis au moins x_col,y_col. Format: 'x_col,y_col[,hue_col]'."
        x_col, y_col = parts[0], parts[1]
        hue_col = parts[2] if len(parts) > 2 else None

        err = self._validate_columns([x_col, y_col, hue_col])
        if err:
            return err

        df = self.csv_tools.df.dropna(subset=[x_col, y_col])
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        ax.set_title(f"{y_col} en fonction de {x_col}")
        fig.autofmt_xdate()
        return self._payload(f"📈 Courbe {y_col} vs {x_col}", fig)

    def plot_scatter(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 2:
            return "❌ Fournis au moins x_col,y_col. Format: 'x_col,y_col[,hue_col]'."
        x_col, y_col = parts[0], parts[1]
        hue_col = parts[2] if len(parts) > 2 else None

        err = self._validate_columns([x_col, y_col, hue_col])
        if err:
            return err
        df = self.csv_tools.df.dropna(subset=[x_col, y_col])
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        ax.set_title(f"{y_col} vs {x_col}")
        return self._payload("📊 Nuage de points", fig)

    def plot_bar(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 2:
            return "❌ Format: 'category_col,value_col[,agg_func]'."
        cat_col, val_col = parts[0], parts[1]
        agg_func = parts[2].lower() if len(parts) > 2 else "sum"

        err = self._validate_columns([cat_col, val_col])
        if err:
            return err
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
        series = agg_map[agg_func].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        series.plot(kind="bar", ax=ax)
        ax.set_title(f"{val_col} par {cat_col} ({agg_func})")
        ax.set_ylabel(val_col)
        return self._payload(f"📊 Barres ({agg_func})", fig)

    def plot_hist(self, input_str: str = "") -> str:
        parts = [p.strip() for p in input_str.split(",") if p.strip()] if input_str else []
        if len(parts) < 1:
            return "❌ Format: 'column[,bins]'."
        col = parts[0]
        bins = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30

        err = self._validate_columns([col])
        if err:
            return err
        if not pd.api.types.is_numeric_dtype(self.csv_tools.df[col]):
            return f"❌ La colonne '{col}' doit être numérique."

        df = self.csv_tools.df[col].dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df, bins=bins, ax=ax, kde=False)
        ax.set_title(f"Distribution de {col}")
        return self._payload(f"📊 Histogramme de {col}", fig)

    def plot_corr_heatmap(self, input_str: str = "") -> str:
        numeric_df = self.csv_tools.df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return "❌ Pas assez de colonnes numériques pour une matrice de corrélation."
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Matrice de corrélation")
        return self._payload("📈 Heatmap de corrélation", fig)

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

