"""Automated QA — face framing, OCR text detection, and CLIP object detection."""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image

from config import (
    EYELINE_TARGET,
    EYELINE_TOLERANCE,
    PROJECT_DIR,
    SHOULDERLINE_TARGET,
    SHOULDERLINE_TOLERANCE,
)

logger = logging.getLogger(__name__)

# Path to the MediaPipe face detection model
_FACE_MODEL_PATH = PROJECT_DIR / "blaze_face_short_range.tflite"

# Lazy-loaded modules (heavy imports)
_mp_face_detector = None
_easyocr_reader = None
_clip_classifier = None


@contextmanager
def _suppress_native_stderr():
    """Temporarily redirect the native stderr fd to devnull.

    This suppresses C++ library messages (TFLite, ONNX) that bypass
    Python’s sys.stderr and write directly to file descriptor 2.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


@dataclass
class QAResult:
    """Result of quality-assurance checks on a single image."""
    passed: bool = True
    reasons: list[str] = field(default_factory=list)
    eye_line_ratio: float | None = None
    shoulder_line_ratio: float | None = None
    detected_text: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)


def _get_face_detector():
    """Lazy-load MediaPipe face detector (Tasks API)."""
    global _mp_face_detector
    if _mp_face_detector is None:
        # Suppress MediaPipe/absl INFO and WARNING logs
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass

        import mediapipe as mp

        if not _FACE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {_FACE_MODEL_PATH}. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/latest/"
                "blaze_face_short_range.tflite"
            )

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(_FACE_MODEL_PATH)
            ),
            min_detection_confidence=0.5,
        )
        with _suppress_native_stderr():
            _mp_face_detector = mp.tasks.vision.FaceDetector.create_from_options(
                options
            )
    return _mp_face_detector


def _get_ocr_reader():
    """Lazy-load EasyOCR reader (uses GPU if available)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        with _suppress_native_stderr():
            _easyocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _easyocr_reader


def _get_clip_classifier():
    """Lazy-load CLIP zero-shot image classifier."""
    global _clip_classifier
    if _clip_classifier is None:
        import torch
        from transformers import pipeline

        # Suppress noisy HuggingFace / httpx download logs
        for name in ("httpx", "huggingface_hub", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)

        device = 0 if torch.cuda.is_available() else -1
        _clip_classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=device,
        )
        logger.info("CLIP loaded on %s", "GPU" if device == 0 else "CPU")
    return _clip_classifier


def _check_face_framing(image: Image.Image, result: QAResult) -> None:
    """Detect faces and verify eye-line / shoulder-line positions."""
    import mediapipe as mp

    detector = _get_face_detector()
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        result.fail("no_face_detected")
        return

    if len(detection_result.detections) > 1:
        result.fail(
            f"multiple_faces_detected ({len(detection_result.detections)})"
        )

    # Use the highest-confidence detection
    best = max(
        detection_result.detections,
        key=lambda d: d.categories[0].score,
    )
    bbox = best.bounding_box

    # Convert pixel bbox to relative coordinates
    rel_ymin = bbox.origin_y / h
    rel_height = bbox.height / h

    # Eye-line estimate: top of bbox + ~30% of bbox height
    eye_y = rel_ymin + rel_height * 0.30
    result.eye_line_ratio = eye_y

    # Shoulder-line estimate: bottom of face bbox + some margin
    shoulder_y = rel_ymin + rel_height * 1.2
    result.shoulder_line_ratio = min(shoulder_y, 1.0)

    # Check eye-line tolerance
    if abs(eye_y - EYELINE_TARGET) > EYELINE_TOLERANCE:
        result.fail(
            f"eyeline_out_of_range (actual={eye_y:.3f}, "
            f"target={EYELINE_TARGET}±{EYELINE_TOLERANCE})"
        )

    # Check shoulder-line tolerance
    if abs(shoulder_y - SHOULDERLINE_TARGET) > SHOULDERLINE_TOLERANCE:
        result.fail(
            f"shoulderline_out_of_range (actual={shoulder_y:.3f}, "
            f"target={SHOULDERLINE_TARGET}±{SHOULDERLINE_TOLERANCE})"
        )


def _check_text_logos(image: Image.Image, result: QAResult) -> None:
    """Use OCR to detect unwanted text / brand logos in the image."""
    try:
        reader = _get_ocr_reader()
        img_array = np.array(image)
        detections = reader.readtext(img_array, detail=1)

        # Filter by confidence
        texts = [
            text for (_, text, conf) in detections
            if conf > 0.4 and len(text.strip()) > 1
        ]
        if texts:
            result.detected_text = texts
            result.fail(f"text_detected: {', '.join(texts)}")
    except Exception:
        logger.warning("OCR check failed, skipping text detection")


def _check_unwanted_objects(image: Image.Image, result: QAResult) -> None:
    """Use CLIP zero-shot classification to detect unwanted objects."""
    try:
        classifier = _get_clip_classifier()

        candidate_labels = [
            "a portrait of a person",
            "a tennis ball",
            "a tennis racket",
            "a brand logo on clothing",
            "a Nike swoosh logo",
            "an Adidas logo",
            "text or writing on clothing",
            "a sportswear brand emblem",
        ]

        _UNWANTED = {
            "a tennis ball", "a tennis racket",
            "a brand logo on clothing", "a Nike swoosh logo",
            "an Adidas logo", "text or writing on clothing",
            "a sportswear brand emblem",
        }

        results = classifier(image, candidate_labels=candidate_labels)
        for item in results:
            if item["label"] in _UNWANTED:
                if item["score"] > 0.10:
                    result.fail(
                        f"unwanted_object: {item['label']} "
                        f"(score={item['score']:.3f})"
                    )
    except ImportError:
        logger.debug(
            "transformers not installed, skipping CLIP object detection"
        )
    except Exception:
        logger.warning("CLIP check failed, skipping object detection")


def run_qa(image_path: Path, checks: str = "all") -> QAResult:
    """Run QA checks on a generated image.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    checks : str
        Comma-separated list of checks to run, or "all".
        Available: face, ocr, clip

    Returns
    -------
    QAResult with pass/fail status and details.
    """
    result = QAResult()

    if not image_path.exists():
        result.fail("file_not_found")
        return result

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        result.fail("invalid_image_file")
        return result

    enabled = set(checks.split(",")) if checks != "all" else {"face", "ocr", "clip"}

    if "face" in enabled:
        _check_face_framing(image, result)

    if "ocr" in enabled:
        _check_text_logos(image, result)

    if "clip" in enabled:
        _check_unwanted_objects(image, result)

    return result
