from langchain.callbacks.base import BaseCallbackHandler


class LLMIterationCounter(BaseCallbackHandler):
    """Callback pour compter les appels LLM."""

    def __init__(self, counter_ref: dict):
        self.counter_ref = counter_ref

    def on_llm_start(self, *args, **kwargs):
        try:
            self.counter_ref["count"] = self.counter_ref.get("count", 0) + 1
        except Exception:
            # Ne pas bloquer l'exécution en cas de souci
            pass

