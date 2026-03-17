"""Generate images via the Google Gemini API."""

import logging
from io import BytesIO
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config import GEMINI_API_KEY, GEMINI_MODEL, IMAGE_FORMAT

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def generate_image(prompt: str, output_path: Path) -> bool:
    """Call Gemini to generate an image from *prompt* and save it to *output_path*.

    Returns True on success, False on failure (logged).
    """
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        # Extract image data from response
        if not response.candidates or not response.candidates[0].content.parts:
            logger.error("No image returned for %s", output_path.name)
            return False

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, format=IMAGE_FORMAT)
                logger.info("Saved %s", output_path)
                return True

        logger.error("Response contained no image data for %s", output_path.name)
        return False

    except Exception:
        logger.exception("Failed to generate image for %s", output_path.name)
        return False
