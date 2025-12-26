"""
Agent orchestrateur qui route les questions vers les agents spécialisés
Utilise un LLM pour un routing intelligent
"""
import os
import time
import json
from typing import Optional, List, Dict, Any
import pandas as pd
from csv_tools import CSVTools
from .time_series_agent import TimeSeriesAgent
from .transformation_agent import TransformationAgent
from .data_viz_agent import DataVizAgent
from .plot_commentary_agent import PlotCommentaryAgent
from config import Config
from llm_factory import get_llm


class OrchestratorAgent:
    """
    Agent principal qui orchestre les agents spécialisés
    Route les questions vers l'agent approprié selon le type de question
    Utilise un LLM pour un routing intelligent et contextuel
    """
    
    def __init__(self, csv_path: str, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialise l'agent orchestrateur
        
        Args:
            csv_path: Chemin vers le fichier CSV
            api_key: Clé API Google (optionnel)
            verbose: Si True, affiche les étapes de raisonnement
        """
        self.csv_path = csv_path
        self.verbose = verbose
        self.last_llm_call_time = 0
        self.llm_counter = {"count": 0}
        
        # Initialisation du LLM pour le routing (léger, rapide)
        # Utilise Ollama si disponible, sinon fallback vers Gemini
        print("🤖 Initialisation du LLM de routing...")
        try:
            self.routing_llm = get_llm(
                model_name=Config.MODEL_NAME,
                temperature=0,  # Déterministe pour le routing/plan
                max_output_tokens=1000,  # Plus de marge pour le JSON de plan
                max_retries=2,
                api_key=api_key,
                verbose=verbose
            )
        except ValueError as e:
            # Si ni Ollama ni Gemini ne sont disponibles, on essaie quand même
            # avec Gemini en forçant la clé API
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            raise ValueError(
                f"Impossible d'initialiser un LLM. {str(e)}\n"
                "Solutions:\n"
                "1. Installez et démarrez Ollama (recommandé pour usage local)\n"
                "2. Définissez GOOGLE_API_KEY dans .env ou passez-la en paramètre"
            )
        
        # Initialisation des outils CSV (partagés par tous les agents)
        print("🔧 Initialisation des outils d'analyse...")
        self.csv_tools = CSVTools(csv_path)
        
        # Initialisation des agents spécialisés
        print("🤖 Initialisation des agents spécialisés...")
        self.time_series_agent = TimeSeriesAgent(
            csv_tools=self.csv_tools,
            api_key=api_key,
            verbose=verbose,
            llm_counter=self.llm_counter,
        )
        
        self.transformation_agent = TransformationAgent(
            csv_tools=self.csv_tools,
            api_key=api_key,
            verbose=verbose,
            llm_counter=self.llm_counter,
        )

        self.data_viz_agent = DataVizAgent(
            csv_tools=self.csv_tools,
            api_key=api_key,
            verbose=verbose,
            llm_counter=self.llm_counter,
        )

        self.plot_commentary_agent = PlotCommentaryAgent(
            api_key=api_key,
            verbose=verbose,
            llm_counter=self.llm_counter,
        )
        
        print("✅ Orchestrateur prêt !\n")
    
    def _plan_agents(self, question: str) -> List[Dict[str, Any]]:
        """
        Planifie 1 à 3 étapes avec les agents disponibles.
        Retour: liste de dicts {agent, instruction}
        """
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        if time_since_last_call < Config.LLM_REQUEST_DELAY:
            time.sleep(Config.LLM_REQUEST_DELAY - time_since_last_call)
        self.last_llm_call_time = time.time()

        agents_desc = (
            "Agents disponibles:\n"
            "- transformation: structure, stats, valeurs manquantes, corrélations, aperçu.\n"
            "- time_series: préparation (fusion date/heure), tendances, moyennes mobiles, agrégations temporelles, anomalies.\n"
            "- visualization: tracés (courbe, scatter, bar, hist, heatmap corr), avec colonnes réelles.\n"
            "- plot_commentary: commente un graphique à partir du résumé JSON produit par visualization.\n"
        )
        prompt = (
            "Tu es un planificateur. Propose un plan de 1 à 3 étapes pour répondre à la question.\n"
            f"{agents_desc}\n"
            "Règles de planification:\n"
            "- IMPORTANT: Si l'utilisateur demande un sous-ensemble de données (ex: une plage de dates, une catégorie, un mois précis), la PREMIÈRE étape DOIT être d'utiliser un outil de filtrage ('filter_data' ou 'filter_by_date') via l'agent 'transformation' ou 'time_series'.\n"
            "- Simplement afficher les données avec 'get_head' n'est PAS suffisant pour que les étapes suivantes (comme la visualisation) en profitent.\n"
            "- N'ajoute une étape 'visualization' QUE si l'utilisateur demande EXPLICITEMENT un graphique/tracé/courbe/plot/heatmap/histogramme.\n"
            "- L'agent 'visualization' utilisera automatiquement les données filtrées par les étapes précédentes.\n"
            "- Ajoute TOUJOURS une étape 'plot_commentary' à la fin pour fournir une petite analyse (5-8 lignes) basée sur les résultats précédents.\n"
            "Formate en JSON strict: {\"steps\": [{\"agent\": \"...\", \"instruction\": \"...\"}, ...]}\n"
            "- agent ∈ {transformation, time_series, visualization, plot_commentary}\n"
            "- instruction: consigne concise en français.\n"
            "- Pas de texte hors JSON.\n"
            f"Question: {question}"
        )
        try:
            self.llm_counter["count"] += 1
            resp = self.routing_llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            if self.verbose:
                print(f"📜 Plan LLM (brut): {content!r}")

            # Nettoyage: retirer fences ```json ... ``` et extraire le premier objet JSON
            cleaned = (content or "").strip()
            if cleaned.startswith("```"):
                # enlève la première ligne ```json / ``` et la dernière ```
                cleaned = cleaned.strip("`").strip()
            # Extraire le premier {...} si du texte s'est glissé
            if "{" in cleaned and "}" in cleaned:
                cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]

            plan = json.loads(cleaned)
            steps = plan.get("steps", [])
            if not isinstance(steps, list) or not steps:
                raise ValueError("steps manquant")
            valid = []
            for step in steps[:3]:
                agent = step.get("agent", "").strip().lower()
                instr = step.get("instruction", "").strip()
                if agent in ["transformation", "time_series", "visualization", "plot_commentary"] and instr:
                    valid.append({"agent": agent, "instruction": instr})
            if not valid:
                raise ValueError("steps invalides")
            return valid
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Planification LLM échouée ({e}), fallback transformation.")
            return [{"agent": "transformation", "instruction": "Réponds à la question de l'utilisateur."}]

    def _synthesize_response(self, question: str, full_context: str) -> str:
        """
        Utilise le LLM pour synthétiser la réponse finale à partir de tout le contexte.
        Filtre le bavardage inutile et ne garde que la valeur ajoutée.
        """
        prompt = (
            "Tu es l'orchestrateur final d'un système multi-agents d'analyse de données.\n"
            "Ta tâche est de produire une réponse PROPRE, CONCISE et PROFESSIONNELLE à l'utilisateur.\n\n"
            "RÈGLES DE SYNTHÈSE :\n"
            "1. Supprime tout le 'bavardage' interne des agents (ex: 'Je vais maintenant...', 'Étape 1 terminée', 'Vous pouvez utiliser...').\n"
            "2. Garde UNIQUEMENT la réponse finale à la question, les statistiques importantes et les tableaux de données s'ils sont pertinents.\n"
            "3. IMPORTANT : Garde les marqueurs de graphiques (PLOT_ID_START/END et PLOT_SUMMARY_START/END) EXACTEMENT tels quels, sans les modifier. Ils sont cruciaux pour l'affichage.\n"
            "4. Si une analyse (commentaire) est présente, fusionne-la intelligemment avec la réponse.\n"
            "5. Réponds TOUJOURS en français.\n"
            "6. Ne mentionne pas les noms techniques des agents (ex: 'L'agent transformation dit...'). Présente les faits directement.\n\n"
            f"Question de l'utilisateur : {question}\n\n"
            f"Contenu brut des agents :\n{full_context}\n\n"
            "Réponse synthétisée :"
        )
        
        try:
            self.llm_counter["count"] += 1
            resp = self.routing_llm.invoke(prompt)
            final_text = resp.content if hasattr(resp, "content") else str(resp)
            return final_text.strip()
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Synthèse échouée ({e}), retour au mode concaténation.")
            return None

    def query(self, question: str) -> str:
        """
        Exécute 1 à 3 agents en séquence selon un plan LLM.
        Le texte produit par chaque agent est passé en contexte au suivant.
        """
        # Réinitialiser les filtres au début de chaque nouvelle question
        self.csv_tools.reset_filter()
        
        steps = self._plan_agents(question)
        context_text = ""
        last_answer = ""
        viz_answer = ""
        commentary_answer = ""

        agent_map = {
            "transformation": self.transformation_agent,
            "time_series": self.time_series_agent,
            "visualization": self.data_viz_agent,
            "plot_commentary": self.plot_commentary_agent,
        }

        for idx, step in enumerate(steps, start=1):
            agent_name = step["agent"]
            instruction = step["instruction"]
            agent = agent_map.get(agent_name)
            if agent is None:
                continue

            composed_question = (
                f"Contexte des étapes précédentes:\n{context_text}\n\n"
                f"Instruction: {instruction}\n\n"
                f"Question utilisateur: {question}"
            )
            if self.verbose:
                print(f"➡️ Étape {idx}: {agent_name} avec instruction '{instruction}'")
            if agent_name == "plot_commentary":
                # On attend que le contexte contienne PLOT_SUMMARY (JSON) produit par visualization ou des stats
                analysis_prompt = (
                    "Tu es un analyste data. On te fournit le contexte des étapes précédentes.\n"
                    "Donne une analyse courte (5-8 lignes max) des résultats : tendances, extrêmes, relations, et ce que ça implique pour la question.\n"
                    "Si un PLOT_SUMMARY est présent, base-toi dessus. Sinon, base-toi sur les statistiques et données textuelles fournies.\n\n"
                    f"{composed_question}"
                )
                answer = agent.query(analysis_prompt)
                commentary_answer = answer
            else:
                answer = agent.query(composed_question)
                if agent_name == "visualization":
                    viz_answer = answer

            context_text += f"\n\n[Étape {idx} - {agent_name}]:\n{answer}"
            last_answer = answer

        if not last_answer:
            last_answer = self.transformation_agent.query(question)

        # Tentative de synthèse intelligente
        synthesized = self._synthesize_response(question, context_text)
        if synthesized:
            return synthesized

        # Fallback : construction manuelle si la synthèse échoue
        previous_outputs = []
        for idx, step in enumerate(steps, start=1):
            if step["agent"] in ["visualization", "plot_commentary"]:
                continue
            marker = f"[Étape {idx} - {step['agent']}]:"
            if marker in context_text:
                start_idx = context_text.find(marker) + len(marker)
                next_marker = f"[Étape {idx + 1} -"
                if next_marker in context_text:
                    end_idx = context_text.find(next_marker)
                    agent_output = context_text[start_idx:end_idx].strip()
                else:
                    agent_output = context_text[start_idx:].strip()
                if agent_output:
                    previous_outputs.append(agent_output)
        
        final_response = ""
        if previous_outputs:
            final_response = "\n\n".join(previous_outputs) + "\n\n"
        
        if viz_answer:
            final_response += viz_answer
        elif not final_response and last_answer and last_answer != commentary_answer:
            final_response = last_answer

        if commentary_answer:
            final_response += f"\n\n📝 Analyse:\n{commentary_answer}"
        
        return final_response.strip() if final_response else last_answer
    
    def get_dataframe(self):
        """Retourne le DataFrame pandas pour un accès direct si nécessaire"""
        return self.csv_tools.df

    def get_llm_iterations(self) -> int:
        """Retourne le nombre d'appels LLM effectués depuis le chargement de l'agent"""
        return int(self.llm_counter.get("count", 0))

