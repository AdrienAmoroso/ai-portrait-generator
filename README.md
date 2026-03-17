# GenerationImageIA

Automated AI image generation pipeline using **Google Gemini API**. Reads prompts from an Excel spreadsheet and generates one image per row.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and paste your [Google AI Studio](https://aistudio.google.com/apikey) API key.

3. **Place your Excel file** in the project root (default: `TM26_PromptsTest_PlayerPics.xlsx`).

## Usage

```bash
# Generate all images
python main.py

# Use a different Excel file
python main.py --excel path/to/file.xlsx

# Generate only specific IDs
python main.py --ids 93,104,356
```

Generated images are saved to the `output/` folder as `{ID}.png`. Already-generated images are skipped on re-run.

## Project Structure

| File | Responsibility |
|---|---|
| `config.py` | Paths, API settings, column mapping |
| `excel_reader.py` | Read and parse the Excel spreadsheet |
| `image_generator.py` | Gemini API interaction |
| `main.py` | Orchestrator (CLI, loop, logging) |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `DELAY_BETWEEN_REQUESTS` | `2` (seconds) | Rate-limit delay between API calls |
