"""Image generation — dispatches to Gemini API or ComfyUI backend."""

import logging
from pathlib import Path

import config
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_client = None


# Gemini backend

def _get_gemini_client():
    global _client
    if _client is None:
        from google import genai

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _generate_gemini(
    positive_prompt: str,
    negative_prompt: str,
    output_path: Path,
    player_id: int,
    seed: int | None = None,
) -> bool:
    from google.genai import types

    client = _get_gemini_client()

    # Gemini has no negative prompt API — embed constraints in the prompt
    anti_logo = (
        "\n\nIMPORTANT: The clothing must be completely plain and unbranded. "
        "Do NOT add any logos, brand names, text, letters, symbols, swooshes, "
        "stripes, or sponsor markings anywhere on the clothing or image. "
        "All garments should be simple solid colors with no embroidery or prints."
    )
    full_prompt = positive_prompt + anti_logo

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="512",
                ),
            ),
        )

        if not response.parts:
            logger.error("Gemini returned empty response for %s", output_path.name)
            return False

        for part in response.parts:
            if part.text is not None:
                logger.info("Model response for %s: %s", output_path.name, part.text)
            elif image := part.as_image():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                logger.info("Saved %s", output_path)
                return True

        logger.error("Response contained no image data for %s", output_path.name)
        return False

    except Exception:
        logger.exception("Gemini generation failed for %s", output_path.name)
        return False


# Public API

def generate_image(
    positive_prompt: str,
    negative_prompt: str,
    output_path: Path,
    player_id: int,
    seed: int | None = None,
    gender: str = "",
) -> bool:
    """Generate an image and save it to *output_path*.

    Dispatches to Gemini or ComfyUI based on the ``BACKEND`` config value.
    Returns True on success, False on failure.
    """
    if config.BACKEND == "comfyui":
        from comfyui_client import generate_comfyui

        return generate_comfyui(
            positive_prompt, negative_prompt, output_path, player_id, seed,
            gender=gender,
        )

    return _generate_gemini(
        positive_prompt, negative_prompt, output_path, player_id, seed
    )
