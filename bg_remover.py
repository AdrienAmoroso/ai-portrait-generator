"""Background removal via U2-Net ONNX (rembg, no pymatting dependency)."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded session
_session = None


@contextmanager
def _suppress_native_stderr():
    """Temporarily redirect the native stderr fd to devnull."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


def _get_session():
    """Lazy-load the rembg ONNX session (U2-Net by default)."""
    global _session
    if _session is None:
        import onnxruntime as ort

        # Suppress ONNX Runtime native warnings (CUDA provider load errors)
        ort.set_default_logger_severity(4)  # FATAL only

        from rembg.sessions import U2netSession

        opts = ort.SessionOptions()
        with _suppress_native_stderr():
            _session = U2netSession("u2net", opts)
    return _session


def remove_background(input_path: Path, output_path: Path) -> bool:
    """Remove the background from an image and save as transparent PNG.

    Uses the rembg U2-Net ONNX model directly, bypassing the pymatting
    dependency which has compatibility issues with Python 3.14.

    Returns True on success, False on failure.
    """
    try:
        session = _get_session()

        img = Image.open(input_path).convert("RGB")

        # Run the model: get alpha mask
        masks = session.predict(img)
        # masks is a list of PIL images; take the first one
        mask = masks[0].convert("L").resize(img.size, Image.LANCZOS)

        # Apply mask as alpha channel
        result = img.convert("RGBA")
        result.putalpha(mask)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path), "PNG")
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
