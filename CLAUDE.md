# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image multi-label annotation tool for anime/illustration datasets. A Flask web app that supports local VLM auto-labeling (Qwen3.5-4B), remote API auto-labeling (OpenAI/Anthropic), and manual correction. The entire repo lives in `F:/datasetpic/` — images and code share the same directory.

## Commands

```bash
# Manual-only mode (no model)
python annotate.py

# Local model auto-labeling (recommended)
python annotate.py --local-model F:/qwen3_5

# With options
python annotate.py --local-model F:/qwen3_5 --dtype float16 --port 8080

# Remote API
python annotate.py --api-key YOUR_KEY --api-type openai --base-url https://api.xxx.com
python annotate.py --api-key YOUR_KEY --api-type anthropic

# CLI batch labeling (no web UI)
python local_vlm.py --model F:/qwen3_5 --image-dir . --batch-size 50

# Test VLM inference
python test_vlm.py
```

Dependencies: `pip install flask pillow` (base), `pip install torch transformers>=5.0.0` (local model).

## Architecture

### Backend (`annotate.py`)

Flask server serving a single-page web UI. Key areas:

- **Label config**: `label_config.json` defines all annotation categories. Each category has `labels` (array of strings) and `multi` (boolean for single vs multi-select). `load_label_config()` falls back to `DEFAULT_LABEL_CONFIG` hardcoded in `annotate.py`.
- **Annotation storage**: `annotations.json` — flat dict keyed by filename. Each entry has `labels`, `custom_tags`, `description`, `review`, `review_history`, `auto_labeled`, `verified`. Read/written on every request (no database).
- **Auto-labeling backends**: Three interchangeable backends selected by `app_config["api_type"]`:
  - `"local"` → `local_vlm.py:LocalVLM` loaded into `app_config["local_vlm"]`
  - `"openai"` → `auto_label_with_openai()` (urllib, no SDK)
  - `"anthropic"` → `auto_label_with_anthropic()` (urllib, no SDK)
- **API routes** (all under `/api/`):
  - `/api/images` — image list with annotation status
  - `/api/annotation/<filename>` — GET/POST/DELETE per-image annotations
  - `/api/auto-label/<filename>` — single image auto-label
  - `/api/auto-label-batch` — threaded batch auto-label
  - `/api/generate-semi-free-description/<filename>` — VLM-generated natural language description
  - `/api/generate-review/<filename>` — content moderation review generation with multi-turn history
  - `/api/label-config` — GET/POST for tag config
  - `/api/role-names` — serves `role_name.json`
  - `/api/export?format=json|csv` — export annotations
  - `/api/image/<filename>` DELETE — deletes image file + annotation

### Local VLM (`local_vlm.py`)

`LocalVLM` class wraps a HuggingFace `AutoModelForImageTextToText` model:

- Uses `AutoModelForImageTextToText` (not `AutoModelForCausalLM`) — critical for image inputs
- `label_image(path)` → structured label dict by building a prompt from `label_config.json` and parsing JSON from model output
- `generate_text(path, custom_prompt)` → free-form text generation for descriptions/reviews
- `parse_model_output()` handles Qwen3.5 thinking mode (`<thinkvi>` tags), markdown code blocks, and nested JSON extraction
- `normalize_labels()` validates model output against `label_config.json`, filtering invalid labels
- Batch CLI via `batch_label()` — processes images sequentially, saves after each

### Frontend (`templates/index.html`)

Single HTML file with inline CSS/JS. Dark theme. Communicates with Flask API via fetch. Keyboard shortcuts for workflow (arrow keys, Ctrl+S, Ctrl+Enter).

### Config Files

- `label_config.json` — category definitions (20+ categories covering gender, hair, eyes, clothing, pose, background, content moderation, etc.)
- `role_name.json` — character name dictionaries per game franchise (Genshin Impact, Blue Archive, Honkai Star Rail, Zenless Zone Zero, Wuthering Waves, Gakuen Idolmaster)
- `annotations.json` — runtime annotation data (auto-generated)

### `fix_role.py`

Debugging/diagnostic script that checks the HTML template for function definitions related to role name handling. Not part of the main application flow.

## Key Design Decisions

- No database — annotations stored as a single JSON file, read/written on every request
- Remote API calls use raw `urllib` (no SDK dependencies)
- Batch auto-labeling runs in a background thread with progress polling
- Images and code coexist in the same directory — `IMAGE_DIR = BASE_DIR`
- The label config in `label_config.json` has diverged from `DEFAULT_LABEL_CONFIG` in `annotate.py` — the JSON file has more categories (20+ vs 10)
