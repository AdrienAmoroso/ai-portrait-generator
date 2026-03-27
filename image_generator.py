"""Image generation — dispatches to Gemini API or ComfyUI backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import config
from config import GEMINI_API_KEY, GEMINI_MODEL

if TYPE_CHECKING:
    from PIL.Image import Image as _PILImage

logger = logging.getLogger(__name__)

_client = None
_face_detector = None


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


def _get_face_detector():
    """Lazy-load and cache the MediaPipe face detector."""
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        from config import PROJECT_DIR

        model_path = PROJECT_DIR / "blaze_face_short_range.tflite"
        if not model_path.exists():
            return None

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            min_detection_confidence=0.5,
        )
        _face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    return _face_detector


def _blur_faces(img: _PILImage) -> _PILImage:
    """Blur detected faces in a PIL image to remove identity signals.

    Uses MediaPipe Face Detection (Tasks API). Returns the image unchanged
    if no faces are found or if detection fails.
    """
    import numpy as np
    from PIL import ImageFilter

    try:
        import mediapipe as mp
    except ImportError:
        logger.debug("mediapipe not available — skipping face blur")
        return img

    try:
        detector = _get_face_detector()
        if detector is None:
            logger.debug("Face detection model not found — skipping face blur")
            return img

        img_rgb = img.convert("RGB")
        np_img = np.array(img_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
        result = detector.detect(mp_image)
    except Exception:
        logger.debug("Face detection failed — skipping face blur", exc_info=True)
        return img

    if not result.detections:
        logger.debug("No faces detected in pose image — skipping blur")
        return img

    w, h = img.size
    blurred = img.copy()
    for det in result.detections:
        bb = det.bounding_box
        x = max(bb.origin_x, 0)
        y = max(bb.origin_y, 0)
        bw = min(bb.width, w - x)
        bh = min(bb.height, h - y)
        face_region = blurred.crop((x, y, x + bw, y + bh))
        face_region = face_region.filter(ImageFilter.GaussianBlur(radius=30))
        blurred.paste(face_region, (x, y))

    logger.debug("Blurred %d face(s) in pose image", len(result.detections))
    return blurred


def _log_response_diagnostics(response, label: str) -> None:
    """Log finish_reason and safety_ratings when a Gemini response is empty."""
    try:
        if not response.candidates:
            logger.warning("  [%s] No candidates in response", label)
            return
        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason") and candidate.finish_reason:
            logger.warning(
                "  [%s] finish_reason: %s", label, candidate.finish_reason
            )
        if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                logger.warning(
                    "  [%s] safety: %s = %s", label, rating.category, rating.probability
                )
    except Exception:
        logger.debug("Could not extract response diagnostics for %s", label)


def _try_generate_gemini(
    client,
    contents: list,
    output_path: Path,
    label: str,
) -> bool:
    """Send a single generate_content request and save the image if present.

    Returns True on success, False on failure.
    """
    from google.genai import types

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="512",
                ),
            ),
        )

        if not response.parts:
            logger.error("Gemini returned empty response for %s", label)
            _log_response_diagnostics(response, label)
            return False

        for part in response.parts:
            if part.text is not None:
                logger.info("Model response for %s: %s", label, part.text)
            elif image := part.as_image():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                logger.info("Saved %s", output_path)
                return True

        logger.error("Response contained no image data for %s", label)
        _log_response_diagnostics(response, label)
        return False

    except Exception:
        logger.exception("Gemini generation failed for %s", label)
        return False


def _generate_gemini(
    positive_prompt: str,
    negative_prompt: str,
    output_path: Path,
    player_id: int,
    seed: int | None = None,
    gender: str = "",
) -> bool:
    from PIL import Image as PILImage

    from comfyui_client import pick_pose

    client = _get_gemini_client()

    # Gemini has no negative prompt API — embed constraints in the prompt
    anti_logo = (
        "\n\nIMPORTANT: The clothing must be completely plain and unbranded. "
        "Do NOT add any logos, brand names, text, letters, symbols, swooshes, "
        "stripes, or sponsor markings anywhere on the clothing or image. "
        "All garments should be simple solid colors with no embroidery or prints."
    )
    full_prompt = positive_prompt + anti_logo

    # Build multimodal contents: optional pose reference + text prompt
    contents: list = []
    used_pose = False
    pose_path = pick_pose(player_id, gender) if config.USE_POSE_REFERENCES else None
    if pose_path is not None:
        pose_img = PILImage.open(pose_path)
        if config.BLUR_POSE_FACES:
            pose_img = _blur_faces(pose_img)
        contents.append(pose_img)
        contents.append(
            full_prompt
            + "\n\nUse this reference image ONLY as a guide for the body pose, "
            "head angle, and framing composition. Generate a completely new and "
            "different person — do not copy the face, clothing, or appearance "
            "from the reference."
        )
        used_pose = True
        logger.info("Using pose reference: %s", pose_path.name)
    else:
        contents.append(full_prompt)

    label = output_path.name

    # First attempt — with pose (if available)
    ok = _try_generate_gemini(client, contents, output_path, label)
    if ok:
        return True

    # Fallback — retry without pose if pose was used
    if used_pose:
        logger.warning(
            "Retrying WITHOUT pose reference for %s (pose may have caused block)",
            label,
        )
        ok = _try_generate_gemini(client, [full_prompt], output_path, label)
        if ok:
            return True

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
        positive_prompt, negative_prompt, output_path, player_id, seed,
        gender=gender,
    )
