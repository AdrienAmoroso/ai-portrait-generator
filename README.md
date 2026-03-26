# GenerationImageIA

Automated AI image generation pipeline for realistic player portraits. Reads prompts from an Excel/CSV spreadsheet, generates images, runs QA checks, removes backgrounds, and exports transparent PNGs.

Supports two backends:
- **Google Gemini API** - cloud-based generation via `gemini-3.1-flash-image-preview`
- **ComfyUI** - local SDXL + ControlNet OpenPose for full control over generation

## Pipeline

```
Excel/CSV → Prompt Builder → Image Generation → QA Checks → Background Removal → Final PNG
```

1. **Data loading** - reads player rows from `.xlsx` or `.csv`
2. **Prompt building** - sanitizes text (removes brands, fixes clothing terms), injects per-player variety (lighting, expression)
3. **Generation** - dispatches to Gemini or ComfyUI
4. **QA checks** - face framing (MediaPipe), text/logo detection (EasyOCR), unwanted object detection (CLIP)
5. **Retry** - failed QA triggers regeneration with a different seed (configurable max retries)
6. **Background removal** - removes background via U2-Net ONNX model (rembg)
7. **Reporting** - QA results exported to `output/qa_report.csv`

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**For GPU acceleration** (recommended for QA checks):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key and preferred backend. See [.env.example](.env.example) for all available variables.

### 3. Download MediaPipe model

Download [blaze_face_short_range.tflite](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite) and place it in the project root.

### 4. Place your input file

Put your Excel (`.xlsx`) or CSV file in the project root. The expected columns are:

| Column | Content |
|--------|---------|
| A | Player ID (integer) |
| B | Gender (`MAN`, `WOMAN`, `MALE`, `FEMALE`) |
| C | Image generation prompt |
| D | Additional facial details (optional) |

## Usage

```bash
# Generate all images (uses backend from .env)
python main.py

# Use a specific input file
python main.py --input path/to/file.xlsx

# Generate only specific IDs
python main.py --ids 93,104,356

# Use Gemini backend, skip QA and BG removal
python main.py --backend gemini --skip-qa --skip-bg

# Run only specific QA checks
python main.py --qa-checks face,ocr
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--input` | Path to Excel or CSV file (default: from `config.py`) |
| `--ids` | Comma-separated player IDs to generate |
| `--backend` | Override backend: `gemini` or `comfyui` |
| `--skip-qa` | Skip all QA checks |
| `--skip-bg` | Skip background removal |
| `--qa-checks` | Comma-separated checks: `face`, `ocr`, `clip`, or `all` |

Generated images are saved to `output/`. Already-generated images are skipped on re-run.

## Project structure

| File | Responsibility |
|------|---------------|
| `config.py` | Centralized settings, paths, prompts, thresholds |
| `excel_reader.py` | Read and parse Excel/CSV input files |
| `prompt_builder.py` | Prompt sanitization, brand removal, variety injection |
| `image_generator.py` | Backend dispatcher (Gemini / ComfyUI) |
| `comfyui_client.py` | ComfyUI API wrapper with ControlNet workflow |
| `qa_checker.py` | Face framing, OCR, and CLIP QA checks |
| `bg_remover.py` | Background removal via U2-Net ONNX |
| `main.py` | Pipeline orchestrator and CLI entry point |

## Configuration

All settings are in `config.py` and can be overridden via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required for Gemini)* | Google AI Studio API key |
| `BACKEND` | `comfyui` | `gemini` or `comfyui` |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI server address |
| `COMFYUI_CHECKPOINT` | `sd_xl_base_1.0.safetensors` | SDXL checkpoint file |
| `COMFYUI_CONTROLNET_MODEL` | `control-lora-openposeXL2-rank256.safetensors` | ControlNet model file |
| `CONTROLNET_STRENGTH` | `0.7` | ControlNet influence strength |
| `QA_ENABLED` | `true` | Enable QA checks |
| `QA_MAX_RETRIES` | `3` | Max regeneration attempts on QA failure |
| `BG_REMOVAL_ENABLED` | `true` | Enable background removal |
| `DELAY_BETWEEN_REQUESTS` | `2` | Seconds between API calls |

## Pose references

Place pose images in `poses/men/` and `poses/women/` for gender-specific ControlNet guidance. Poses are deterministically assigned per player ID. Falls back to `poses/` if gender-specific folders are empty.

## Output

```
output/
├── raw/           # Raw generated images (before BG removal)
├── final/         # Final transparent PNGs
└── qa_report.csv  # QA check results per image
```
