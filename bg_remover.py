"""Background removal via rembg with birefnet-portrait model."""

import logging
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded session
_session = None


def _get_session():
    """Lazy-load the rembg birefnet-portrait session."""
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session(model_name="birefnet-portrait")
    return _session


def remove_background(input_path: Path, output_path: Path) -> bool:
    """Remove the background from an image and save as transparent PNG.

    Uses the rembg birefnet-portrait model, optimized for portrait images.

    Returns True on success, False on failure.
    """
    try:
        from rembg import remove

        session = _get_session()

        image = Image.open(input_path)
        if image.mode in ("P", "L", "LA"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = image.convert("RGBA")

        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        output_data = remove(buf.read(), session=session)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(output_data)

        logger.info("Background removed → %s", output_path)
        return True

    except ImportError:
        logger.error(
            "rembg is not installed. Install with: pip install rembg"
        )
        return False
    except Exception:
        logger.exception("Background removal failed for %s", input_path.name)
        return False
