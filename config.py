"""Centralized configuration for the image generation pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL_FILE = PROJECT_DIR / "TM26_PromptsTest_PlayerPics.xlsx"
OUTPUT_DIR = PROJECT_DIR / "output"

# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-image-preview"

# --- Image generation ---
IMAGE_FORMAT = "PNG"
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "2"))

# --- Excel columns (1-indexed) ---
COL_ID = 1
COL_GENDER = 2
COL_PROMPT = 3
COL_DETAILS = 4
HEADER_ROW = 1
