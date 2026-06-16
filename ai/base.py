# ════════════════════════════════════════════════════════
#  DataBridge AI — AI Engine Base (Strategy Pattern)
# ════════════════════════════════════════════════════════
import abc


class AIEngineStrategy(abc.ABC):

    @abc.abstractmethod
    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        ...

    @abc.abstractmethod
    def get_engine_type(self) -> str:
        ...
<<<<<<< HEAD

    def test_connection(self) -> tuple[bool, str]:
        """Verify the engine is reachable/configured. Returns (ok, message)."""
        return True, "OK"
=======
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
