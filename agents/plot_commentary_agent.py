"""
Agent spécialisé pour commenter un graphique à partir d'un résumé JSON.
Pas d'outils : uniquement du texte LLM à partir d'inputs contrôlés.
"""
import time
from typing import Optional

from callbacks import LLMIterationCounter
from config import Config
from llm_factory import get_llm


class PlotCommentaryAgent:
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True, llm_counter: Optional[dict] = None):
        self.verbose = verbose
        self.last_llm_call_time = 0
        self.callbacks = [LLMIterationCounter(llm_counter)] if llm_counter is not None else None

        self.llm = get_llm(
            model_name=Config.MODEL_NAME,
            temperature=0,
            max_output_tokens=512,
            max_retries=2,
            api_key=api_key,
            verbose=verbose,
        )

    def query(self, prompt: str) -> str:
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        if time_since_last_call < Config.LLM_REQUEST_DELAY:
            time.sleep(Config.LLM_REQUEST_DELAY - time_since_last_call)
        self.last_llm_call_time = time.time()

        # Pas de tools -> simple invoke
        try:
            if self.callbacks:
                resp = self.llm.invoke(prompt, config={"callbacks": self.callbacks})
            else:
                resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            return f"❌ Erreur : {e}"



