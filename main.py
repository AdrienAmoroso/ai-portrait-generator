"""Main orchestrator – reads prompts from Excel and generates images."""

import argparse
import logging
import time
from pathlib import Path

from config import DEFAULT_EXCEL_FILE, DELAY_BETWEEN_REQUESTS, IMAGE_FORMAT, OUTPUT_DIR
from excel_reader import read_prompts
from image_generator import generate_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI images from Excel prompts via Gemini API."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_EXCEL_FILE,
        help="Path to the Excel file containing prompts.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of IDs to generate (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Read prompts
    rows = read_prompts(args.excel)
    logger.info("Loaded %d prompts from %s", len(rows), args.excel.name)

    # Filter by IDs if requested
    if args.ids:
        id_set = {int(x.strip()) for x in args.ids.split(",")}
        rows = [r for r in rows if r.id in id_set]
        logger.info("Filtered to %d prompts matching --ids", len(rows))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        ext = IMAGE_FORMAT.lower()
        output_path = OUTPUT_DIR / f"{row.id}.{ext}"

        if output_path.exists():
            logger.info("[%d/%d] Skipping ID %s (already exists)", i, len(rows), row.id)
            skipped += 1
            continue

        logger.info("[%d/%d] Generating image for ID %s …", i, len(rows), row.id)
        ok = generate_image(row.full_prompt, output_path)

        if ok:
            success += 1
        else:
            failed += 1

        # Rate-limit between API calls (skip delay after last item)
        if i < len(rows):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    logger.info(
        "Done — %d generated, %d skipped, %d failed (out of %d)",
        success,
        skipped,
        failed,
        len(rows),
    )


if __name__ == "__main__":
    main()
