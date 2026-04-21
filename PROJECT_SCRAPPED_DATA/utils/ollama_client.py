"""
MIDAN Data Pipeline -- Ollama LLM Client
Local LLM integration for insight extraction ONLY.

STRICT USE:
- Narrative parsing from Failory / case studies
- Failure reasoning extraction

DO NOT:
- Replace rule-based classification engine
- Infer business models or categories
- Use for generating data (no hallucination risk)
- Use as primary extraction method
"""

import json
import requests
from utils.logger import setup_logger
from config.settings import LOG_FILE, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = setup_logger("ollama", LOG_FILE)


class OllamaClient:
    """
    Client for local Ollama LLM -- used as a SUPPLEMENT to rule-based extraction.
    Only activated when rule-based extraction produces ambiguous results.
    """

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = model or OLLAMA_MODEL
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.model in model_names:
                    logger.info(f"Ollama available with model: {self.model}")
                    return True
                else:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {model_names}. Run: ollama pull {self.model}"
                    )
                    return False
            return False
        except Exception:
            logger.info("Ollama not available -- using rule-based extraction only")
            return False

    def extract_failure_reasoning(self, text: str, startup_name: str) -> dict:
        """
        Use LLM to extract structured failure reasoning from narrative text.

        Returns:
            dict with keys: failure_reasons, funding_stage, market_context
            Returns empty dict if LLM is unavailable or fails.
        """
        if not self.available:
            return {}

        prompt = f"""Analyze this startup failure case study for "{startup_name}".
Extract ONLY factual information that is explicitly stated in the text.
Do NOT infer, assume, or generate information not present.

Text:
{text[:2000]}

Respond in JSON format with these fields:
- failure_reasons: list of 1-3 specific reasons the startup failed (from the text)
- funding_stage: what funding stage they were at when they failed (e.g., "Seed", "Series A")
- market_context: brief description of the market they operated in

If information is not in the text, use empty string or empty list.
Respond with ONLY the JSON object, no explanation."""

        result = self._query(prompt)
        if result:
            return self._parse_json_response(result)
        return {}


    def _query(self, prompt: str, temperature: float = 0.2) -> str | None:
        """Send a query to Ollama and return the response text."""
        if not self.available:
            return None

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 500,
                    },
                },
                timeout=30,
            )

            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                logger.warning(f"Ollama returned status {resp.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("Ollama query timed out")
            return None
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> dict:
        """Parse a JSON response from the LLM, handling common formatting issues."""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        logger.warning(f"Failed to parse LLM JSON response: {text[:100]}...")
        return {}


# Singleton instance
_ollama_client = None


def get_ollama_client() -> OllamaClient:
    """Get or create the singleton Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
