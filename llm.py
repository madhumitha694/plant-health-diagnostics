"""Google Gemini explanation for the classifier's existing result."""
import json
from typing import Any

from google import genai


def generate_explanation(result: dict[str, Any], api_key: str) -> str:
    if not api_key.strip():
        raise ValueError("Enter a Gemini API key to request an explanation.")

    diagnosis = {
        "predicted_label": result["primary_label"],
        "confidence": result["primary_confidence"],
        "top_predictions": result["top_k"],
        "classifier_warnings": result["warnings"],
        "existing_recommendations": result["recommendations"],
    }
    prompt = (
        "Explain this plant-health classifier result for a home gardener. "
        "Treat the predicted label as a fallible estimate, not a confirmed diagnosis. "
        "Do not change or overstate the classifier result. Use only the supplied "
        "information, distinguish possibilities from facts, and give cautious, "
        "practical next steps. Do not invent fertilizer rates or claim that a photo "
        "confirms a soil nutrient level. Recommend soil/tissue testing or an "
        "agronomist where appropriate. The uploaded image is not included.\n\n"
        f"Classifier result:\n{json.dumps(diagnosis, ensure_ascii=False)}"
    )
    client = genai.Client(api_key=api_key.strip())
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        store=False,
    )
    explanation = (interaction.output_text or "").strip()
    if not explanation:
        raise RuntimeError("Gemini returned an empty explanation.")
    return explanation
