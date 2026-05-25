# backend/app/services/gemini.py
import json
import asyncio
import re
import logging

from app.config import settings
from app.services.plastic_info import get_plastic_info

logger = logging.getLogger(__name__)

# Lazy-initialised Gemini client singleton
_client = None


def _get_client():
    """
    Return a Gemini client, or None if no API key is configured.
    Lazy-initialised so startup doesn't fail if google-genai isn't installed.
    """
    global _client
    if _client is None and settings.gemini_api_key:
        try:
            from google import genai
            _client = genai.Client(api_key=settings.gemini_api_key)
        except ImportError:
            logger.warning("google-genai package not installed. Gemini disabled.")
    return _client


def _build_prompt(plastic_type: str, confidence: float, is_uncertain: bool) -> str:
    """Build a structured prompt requesting JSON output from Gemini."""
    info = get_plastic_info(plastic_type)  # imported at top

    if is_uncertain:
        confidence_note = (
            f"Note: identification confidence is {confidence:.0%}, which is below the "
            "reliable threshold. Provide general advice for this plastic type."
        )
    else:
        confidence_note = f"Identification confidence: {confidence:.0%}."

    return f"""You are a plastic waste and recycling expert helping a user responsibly dispose of plastic.

The user's item has been identified as: {plastic_type} ({info['full_name']}, Resin Code #{info['resin_code']}).
{confidence_note}

Respond ONLY with a valid JSON object — no markdown, no code fences, no extra text:
{{
  "recycling_tips": ["tip1", "tip2", "tip3"],
  "reuse_ideas": ["idea1", "idea2", "idea3"],
  "eco_alternatives": ["alt1", "alt2"],
  "environmental_note": "One concise sentence about the environmental impact of this plastic."
}}

Be specific and actionable. Each item must be one sentence or less."""


def _parse_gemini_response(text: str) -> dict | None:
    """
    Extract the JSON object from Gemini's response text.
    Handles cases where the model wraps output in markdown code fences.
    Returns None if parsing fails entirely.
    """
    # Direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Extract first {...} block (handles markdown fences or preamble text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _static_fallback(plastic_type: str) -> dict:
    """
    Build a suggestions dict from the static knowledge base.
    Schema is identical to what Gemini returns — frontend is unaware of the source.
    """
    info = get_plastic_info(plastic_type)
    return {
        "recycling_tips": info["recycling_tips"],
        "reuse_ideas": info["reuse_ideas"],
        "eco_alternatives": info["eco_alternatives"],
        "environmental_note": (
            f"{info['full_name']} takes approximately {info['decomposition_years']} "
            f"years to decompose. Recyclability: {info['recyclability']}."
        ),
    }


async def get_ai_suggestions(
    plastic_type: str,
    confidence: float,
    is_uncertain: bool,
) -> tuple[dict, str]:
    """
    Return (suggestions_dict, source) where source is 'ai' or 'static'.

    This function NEVER raises. All failure paths return the static fallback
    so the /predict endpoint always completes successfully.

    Why asyncio.wait_for + run_in_executor?
    The google-genai SDK is synchronous. Calling it directly in an async
    FastAPI route would block the entire event loop, preventing other requests
    from being served during the Gemini round-trip. run_in_executor offloads
    the blocking call to a thread pool, and wait_for enforces the timeout.
    """
    client = _get_client()

    if client is None:
        return _static_fallback(plastic_type), "static"

    prompt = _build_prompt(plastic_type, confidence, is_uncertain)

    try:
        # get_running_loop() — get_event_loop() is deprecated in Python 3.10+
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                ),
            ),
            timeout=settings.gemini_timeout_seconds,
        )

        parsed = _parse_gemini_response(response.text)
        if parsed is None:
            logger.warning("Gemini response could not be parsed as JSON — using static fallback")
            return _static_fallback(plastic_type), "static"

        return parsed, "ai"

    except asyncio.TimeoutError:
        logger.warning(f"Gemini timed out after {settings.gemini_timeout_seconds}s — using static fallback")
        return _static_fallback(plastic_type), "static"

    except Exception as e:
        logger.warning(f"Gemini error ({type(e).__name__}: {e}) — using static fallback")
        return _static_fallback(plastic_type), "static"
