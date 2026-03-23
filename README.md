# Content Cleaner — Manga NSFW Censor

Automatically detects and censors explicit content in manga/anime PDF files using AI models trained on drawn art. Rebuilds a clean PDF at original image quality.

---

## How it works

| Step | What happens |
|------|-------------|
| **1. Extract** | Each page is extracted at native resolution (no recompression) |
| **2. WD14 gate** | SmilingWolf's WD14 tagger (trained on 100M+ Danbooru images) scores the page for explicit tags — pages scoring below `tile-cls-gate` are skipped entirely |
| **3. Tile scan** | Explicit-flagged pages are scanned with a 256 px sliding window; each tile is re-scored by WD14 |
| **4. NudeNet** | Secondary photorealistic fallback (bounding-box detector) also runs on every page |
| **5. Merge & censor** | Overlapping tile hits are merged into clean rectangles, then drawn as black boxes |
| **6. Text restore** | EasyOCR finds text regions inside censor boxes and re-renders them as white boxes with black text, so speech-bubble dialogue is preserved without ever revealing original pixels |
| **7. Rebuild** | Censored pages are stitched back into a new PDF |

---

## Installation

### Prerequisites

- Python 3.10+
- `pip`

### 1. Clone / download

```bash
git clone <repo-url>
cd "Content Cleaner"
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS Python 3.13 SSL fix** — if you see `CERTIFICATE_VERIFY_FAILED` errors,
> run once:
> ```bash
> /Applications/Python\ 3.13/Install\ Certificates.command
> ```

Models are downloaded automatically on first run and cached locally:

| Model | Size | Cache |
|-------|------|-------|
| NudeNet ONNX | ~5 MB | `~/.NudeNet/` |
| WD14 SwinV2 ONNX | ~200 MB | `~/.cache/huggingface/hub/` |
| EasyOCR CRAFT + recognizer | ~50 MB | `~/.EasyOCR/model/` |

After the first run the tool is **fully offline**.

---

## Usage

### Graphical UI (recommended)

```bash
source .venv/bin/activate
python3 gui.py
```

The GUI exposes every flag below in a point-and-click interface with a live command preview and streaming log output.

### Command line

#### Single file

```bash
python3 censor.py manga.pdf
```

Output: `manga_censored.pdf` in the same folder.

#### Entire folder

```bash
python3 censor.py ./manga-folder/
```

Output: `./manga-folder_censored/` containing censored copies of every PDF.

#### Custom output path

```bash
python3 censor.py manga.pdf --output ./censored/manga.pdf
python3 censor.py ./manga-folder/ --output ./censored/
```

---

## All flags

### Thresholds

| Flag | Default | Description |
|------|---------|-------------|
| `--page-threshold` | `0.20` | WD14 score required to run tile scan on a page. Lower → more pages scanned |
| `--tile-threshold` | `0.40` | WD14 score for a tile to be censored. Lower → larger censor areas |
| `--nudenet-threshold` | `0.50` | Confidence cutoff for NudeNet bounding-box detections |
| `--tile-cls-gate` | `0.02` | Minimum WD14 page score to enable tile scanning at all. Pages below this are skipped with no tile scan |

### Tile scanner

| Flag | Default | Description |
|------|---------|-------------|
| `--tile-size` | `256` | Tile size in pixels. Smaller = more precise but slower |
| `--tile-stride` | `128` | Step between tiles. Smaller = more overlap, finer coverage |

### Models

| Flag | Default | Description |
|------|---------|-------------|
| `--no-wd14` | off | Disable WD14 entirely — NudeNet only |
| `--no-tile` | off | Disable tile scanner — WD14 used as page gate only |
| `--wd14-model` | `SmilingWolf/wd-v1-4-swinv2-tagger-v2` | HuggingFace repo for the WD14 ONNX model |
| `--yolo-weights` | — | Path to YOLOv8 `.pt` weights for an anime-specific NSFW detector |
| `--fallback-dpi` | `300` | DPI used to render pages when native image extraction fails |
| `--labels` | NudeNet defaults | Space-separated NudeNet class names to enable |

### Text restoration

| Flag | Default | Description |
|------|---------|-------------|
| `--no-restore-text` | off | Disable text restoration (it is **on** by default) |
| `--restore-text-lang` | `en` | EasyOCR language codes. Use `ja` for Japanese, `ja+en` for both |

Text restoration works by running EasyOCR on the original page, finding text regions that overlap with censor boxes, then drawing a white rectangle + re-rendered text on the censored image. **No original pixels are ever restored** — NSFW content cannot leak through.

### Debug

| Flag | Default | Description |
|------|---------|-------------|
| `--debug-pages` | — | Process only these pages and save annotated PNGs to `--debug-dir`. No output PDF is written. Supports ranges and comma lists: `41-43`, `7`, `1-3,7` |
| `--debug-dir` | `./debug` | Directory for debug PNG output |

---

## Tuning guide

**Too many false positives** (clean pages being censored):
- Raise `--page-threshold` (e.g. `0.30`)
- Raise `--tile-threshold` (e.g. `0.50`)

**Missing explicit content** (pages not being caught):
- Lower `--page-threshold` (e.g. `0.10`)
- Lower `--tile-threshold` (e.g. `0.30`)
- Lower `--tile-cls-gate` (e.g. `0.01`)

**Debug workflow:**
```bash
# 1. Check specific pages with annotated output
python3 censor.py manga.pdf --debug-pages 41-43

# 2. Inspect ./debug/debug_page_041.png — see WD14 scores + tile boxes
# 3. Adjust thresholds, re-debug
# 4. Full run when satisfied
python3 censor.py manga.pdf
```

---

## Project structure

```
Content Cleaner/
├── censor.py          # Core CLI tool
├── gui.py             # Tkinter GUI wrapper
├── requirements.txt   # Python dependencies
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyMuPDF` | PDF page extraction and rebuilding |
| `Pillow` | Image manipulation |
| `numpy` | Array processing for WD14 |
| `nudenet` | Photorealistic NSFW bounding-box detector |
| `onnxruntime` | ONNX inference for NudeNet and WD14 |
| `huggingface_hub` | WD14 model download |
| `easyocr` | Neural text detection and OCR for text restoration |
