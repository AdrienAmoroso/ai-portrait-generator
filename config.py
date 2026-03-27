"""Centralized configuration — all settings and environment variable overrides."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = PROJECT_DIR / "TM26_TestGenPicsPlayers_Prompts_2.xlsx"
OUTPUT_DIR = PROJECT_DIR / "output"
RAW_DIR = OUTPUT_DIR / "raw"
FINAL_DIR = OUTPUT_DIR / "final"
POSES_DIR = PROJECT_DIR / "poses"

# Backend selection
BACKEND = os.getenv("BACKEND", "comfyui")

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-image-preview"

# ComfyUI
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_CHECKPOINT = os.getenv(
    "COMFYUI_CHECKPOINT", "sd_xl_base_1.0.safetensors"
)
COMFYUI_CONTROLNET_MODEL = os.getenv(
    "COMFYUI_CONTROLNET_MODEL", "control-lora-openposeXL2-rank256.safetensors"
)
CONTROLNET_STRENGTH = float(os.getenv("CONTROLNET_STRENGTH", "0.7"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1024"))
GENERATION_STEPS = int(os.getenv("GENERATION_STEPS", "25"))
CFG_SCALE = float(os.getenv("CFG_SCALE", "7.0"))

# Image generation
IMAGE_FORMAT = "PNG"
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "2"))

# QA thresholds
QA_ENABLED = os.getenv("QA_ENABLED", "true").lower() == "true"
QA_MAX_RETRIES = int(os.getenv("QA_MAX_RETRIES", "3"))
EYELINE_TARGET = 0.375  # 37.5% from top of frame
EYELINE_TOLERANCE = 0.10  # ±10%
SHOULDERLINE_TARGET = 0.75  # 75% from top
SHOULDERLINE_TOLERANCE = 0.20  # ±20%

# Background removal
BG_REMOVAL_ENABLED = os.getenv("BG_REMOVAL_ENABLED", "true").lower() == "true"

# Pose references
USE_POSE_REFERENCES = os.getenv("USE_POSE_REFERENCES", "true").lower() == "true"
BLUR_POSE_FACES = os.getenv("BLUR_POSE_FACES", "true").lower() == "true"

# Input file columns (1-indexed)
COL_ID = 1
COL_GENDER = 2
COL_PROMPT = 3
COL_DETAILS = 4
HEADER_ROW = 1
CSV_DELIMITER = ";"
CSV_ENCODING = "latin-1"

# Global prompt & negative prompt
GLOBAL_PROMPT = (
    "Realistic portrait photograph of a professional athlete, studio quality, "
    "head-and-shoulders composition, shallow depth of field, plain white background, "
    "shot on 85mm portrait lens, high resolution, natural skin texture with subtle "
    "imperfections, professional sports portrait photography style, "
    "plain unbranded clothing with no visible logos or text, "
    "no brand markings, no sponsor logos, no embroidered symbols on clothing"
)

NEGATIVE_PROMPT = (
    "brand logo, text, letters, brand name, sponsor, Nike, Adidas, Wilson, Head, "
    "watermark, signature, tennis ball, tennis racket, tennis court, tennis net, "
    "equipment, hands, fingers, holding anything, ball, racquet, "
    "deformed, distorted, disfigured, blurry, low quality, low resolution, "
    "cartoon, illustration, painting, drawing, anime, CGI, 3D render, "
    "duplicate, multiple people, background people, crowd, spectators"
)
