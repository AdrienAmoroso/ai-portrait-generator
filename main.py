"""Pipeline orchestrator — prompt build, generation, QA, and background removal."""

import os

# Suppress native C++ warnings before any heavy imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # TensorFlow Lite
os.environ.setdefault("GLOG_minloglevel", "3")           # MediaPipe / absl

import argparse
import csv
import logging
import time
from pathlib import Path

from config import (
    BACKEND,
    BG_REMOVAL_ENABLED,
    DEFAULT_INPUT_FILE,
    DELAY_BETWEEN_REQUESTS,
    FINAL_DIR,
    IMAGE_FORMAT,
    OUTPUT_DIR,
    QA_ENABLED,
    QA_MAX_RETRIES,
    RAW_DIR,
)
from excel_reader import read_prompts
from image_generator import generate_image
from prompt_builder import build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI player portraits – full pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the Excel or CSV file containing prompts.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of IDs to generate (default: all).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["gemini", "comfyui"],
        help="Override the generation backend (default: from config/env).",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA checks even if QA_ENABLED is True.",
    )
    parser.add_argument(
        "--skip-bg",
        action="store_true",
        help="Skip background removal even if BG_REMOVAL_ENABLED is True.",
    )
    parser.add_argument(
        "--qa-checks",
        type=str,
        default="all",
        help="Comma-separated QA checks to run: face,ocr,clip or 'all'.",
    )
    parser.add_argument(
        "--skip-poses",
        action="store_true",
        help="Skip pose references even if images are available.",
    )
    return parser.parse_args()


def _run_qa(image_path: Path, checks: str) -> "QAResult":  # noqa: F821
    """Run QA on an image, handling import gracefully."""
    from qa_checker import run_qa
    return run_qa(image_path, checks=checks)


def _run_bg_removal(raw_path: Path, final_path: Path) -> bool:
    """Remove background, handling import gracefully."""
    from bg_remover import remove_background
    return remove_background(raw_path, final_path)


def main() -> None:
    args = parse_args()

    # Allow CLI override of backend
    if args.backend:
        import config as _cfg
        _cfg.BACKEND = args.backend

    # Allow CLI override of pose references
    if args.skip_poses:
        import config as _cfg
        _cfg.USE_POSE_REFERENCES = False

    # Re-read after potential override
    from config import BACKEND as active_backend

    do_qa = QA_ENABLED and not args.skip_qa
    do_bg = BG_REMOVAL_ENABLED and not args.skip_bg

    # Read prompts
    rows = read_prompts(args.input)
    logger.info("Loaded %d prompts from %s", len(rows), args.input.name)

    # Filter by IDs if requested
    if args.ids:
        id_set = {int(x.strip()) for x in args.ids.split(",")}
        rows = [r for r in rows if r.id in id_set]
        logger.info("Filtered to %d prompts matching --ids", len(rows))

    logger.info(
        "Backend: %s | QA: %s | BG removal: %s", active_backend, do_qa, do_bg
    )

    # Ensure output directories exist
    raw_dir = RAW_DIR if do_bg else OUTPUT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    if do_bg:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"generated": 0, "skipped": 0, "failed": 0, "qa_failed": 0, "bg_done": 0}

    # QA report rows for CSV
    qa_rows: list[dict] = []

    for i, row in enumerate(rows, 1):
        ext = IMAGE_FORMAT.lower()
        final_path = (FINAL_DIR if do_bg else OUTPUT_DIR) / f"{row.id}.{ext}"
        raw_path = raw_dir / f"{row.id}.{ext}"

        # Skip if final output already exists
        if final_path.exists():
            logger.info("[%d/%d] Skipping ID %s (already exists)", i, len(rows), row.id)
            stats["skipped"] += 1
            continue

        # Build structured prompt
        built = build_prompt(row.id, row.prompt, row.details)
        logger.info("[%d/%d] Generating image for ID %s …", i, len(rows), row.id)

        # Generation + QA retry loop
        generated = False
        attempt = 0
        max_attempts = QA_MAX_RETRIES if do_qa else 1

        while attempt < max_attempts:
            attempt += 1
            seed = None if attempt == 1 else attempt * 1000 + row.id

            ok = generate_image(
                built.positive, built.negative, raw_path, row.id,
                seed=seed, gender=row.gender,
            )
            if not ok:
                if attempt < max_attempts:
                    logger.warning(
                        "  Generation failed (attempt %d/%d), retrying…",
                        attempt, max_attempts,
                    )
                    continue
                break

            # Run QA if enabled
            if do_qa:
                qa_result = _run_qa(raw_path, args.qa_checks)
                qa_rows.append({
                    "id": row.id,
                    "attempt": attempt,
                    "passed": qa_result.passed,
                    "reasons": "; ".join(qa_result.reasons),
                    "eye_line": qa_result.eye_line_ratio,
                    "shoulder_line": qa_result.shoulder_line_ratio,
                    "detected_text": ", ".join(qa_result.detected_text),
                })

                if not qa_result.passed:
                    logger.warning(
                        "  QA failed (attempt %d/%d): %s",
                        attempt, max_attempts, ", ".join(qa_result.reasons),
                    )
                    if attempt < max_attempts:
                        # Delete and retry
                        raw_path.unlink(missing_ok=True)
                        continue
                    else:
                        stats["qa_failed"] += 1
                        logger.warning(
                            "  QA still failing after %d attempts, keeping last result",
                            max_attempts,
                        )

            generated = True
            break

        if generated:
            stats["generated"] += 1
            # Background removal
            if do_bg:
                bg_ok = _run_bg_removal(raw_path, final_path)
                if bg_ok:
                    stats["bg_done"] += 1
                else:
                    logger.warning("  BG removal failed, raw file kept at %s", raw_path)
        else:
            stats["failed"] += 1

        # Rate-limit between API calls (skip delay after last item)
        if i < len(rows):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Write QA report if applicable
    if qa_rows:
        report_path = OUTPUT_DIR / "qa_report.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id", "attempt", "passed", "reasons",
                    "eye_line", "shoulder_line", "detected_text",
                ],
            )
            writer.writeheader()
            writer.writerows(qa_rows)
        logger.info("QA report written to %s", report_path)

    logger.info(
        "Done — %d generated, %d skipped, %d failed, %d QA-failed, %d BG-removed "
        "(out of %d)",
        stats["generated"],
        stats["skipped"],
        stats["failed"],
        stats["qa_failed"],
        stats["bg_done"],
        len(rows),
    )


if __name__ == "__main__":
    main()
