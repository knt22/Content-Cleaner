#!/usr/bin/env python3
"""
censor.py — Censor NSFW regions in manga/anime PDF files.

Detection approach
──────────────────
1. WD14 tagger (SmilingWolf/wd-v1-4-swinv2-tagger-v2, ~200 MB, downloads once)
   Trained on 100 M+ anime/manga images from Danbooru.  It knows what "nipples",
   "nude", "genitalia" look like in drawn art — and crucially it does NOT fire on
   "character in clothing", "action pose", or "mech suit".  Used as:
     • Page-level gate  — skip tile scan when no explicit tags score above 0.20
     • Tile scorer      — 256 px tiles, censor when any explicit tag ≥ 0.40

2. NudeNet (secondary, photorealistic fallback)

3. Box merging — overlapping tile boxes are merged into clean rectangles before
   being drawn, preventing the checkerboard effect.

Usage:
    python censor.py manga.pdf
    python censor.py manga.pdf --debug-pages 41-43
    python censor.py manga.pdf --page-threshold 0.15 --tile-threshold 0.35
    python censor.py ./folder/

Tuning:
    --page-threshold  lower → more pages get tile-scanned  (default 0.20)
    --tile-threshold  lower → more tiles get censored      (default 0.40)
    Both thresholds refer to WD14 explicit-tag scores (0–1).
"""

import argparse
import csv
import io
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


# ─── Detection data class ─────────────────────────────────────────────────────

@dataclass
class Detection:
    label:  str
    score:  float
    box:    tuple        # (x, y, w, h) in pixels
    source: str = ""     # which detector produced this


# ─── Detector base ────────────────────────────────────────────────────────────

class BaseDetector(ABC):
    name: str = "base"

    def load(self) -> None:
        pass

    @abstractmethod
    def detect(self, img_bytes: bytes) -> list[Detection]:
        pass


# ─── NudeNet (photorealistic fallback) ────────────────────────────────────────

NUDENET_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}


class NudeNetDetector(BaseDetector):
    name = "nudenet"

    def __init__(self, labels: set[str] = NUDENET_LABELS):
        self._labels = labels
        self._model  = None

    def load(self) -> None:
        from nudenet import NudeDetector
        print("  Loading NudeNet model...")
        self._model = NudeDetector()

    def detect(self, img_bytes: bytes) -> list[Detection]:
        raw = self._model.detect(img_bytes)
        return [
            Detection(d["class"], d["score"], tuple(d["box"]), source="nudenet")
            for d in raw if d["class"] in self._labels
        ]


# ─── WD14 Tagger ──────────────────────────────────────────────────────────────

class WD14Tagger:
    """
    SmilingWolf's WD14 anime image tagger, loaded via ONNX.

    Why this beats a generic photo-NSFW classifier for manga:
      • Trained on 100 M+ images from Danbooru — understands drawn art natively.
      • Returns specific tags: "nipples", "nude", "penis", "vagina", etc.
      • Does NOT fire on: cleavage, swimsuits, action poses, tight clothing.
      • Correctly ignores: mechs, battle scenes, fully-clothed characters.

    Only tags in EXPLICIT_TAGS are counted as censor-worthy.  Tags like
    "large_breasts" or "cleavage" are intentionally excluded — they describe
    suggestive but non-explicit content.
    """

    DEFAULT_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"

    # WD14 general tags (category 0) that indicate content to be censored.
    # Deliberately narrow — suggestive-but-clothed tags are excluded.
    EXPLICIT_TAGS = {
        "nipples", "nude", "pussy", "penis", "sex", "cum", "vagina",
        "naked", "topless", "bottomless", "uncensored", "areolae", "vulva",
        "anus", "nipple_slip", "completely_nude", "partially_nude",
        "pubic_hair", "nipple_focus", "phallus", "erection", "breast_out",
        "genitals", "spread_legs",
    }

    def __init__(self, repo: str = DEFAULT_REPO):
        self._repo            = repo
        self._model           = None
        self._input_name      = None
        self._tag_names:      list[str]       = []
        self._explicit_idx:   list[int]       = []
        self._rating_idx:     dict[str, int]  = {}

    def load(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        print(f"  Loading WD14 tagger: {self._repo}")
        model_path = hf_hub_download(self._repo, "model.onnx")
        tags_path  = hf_hub_download(self._repo, "selected_tags.csv")

        # CPU only — CoreML doesn't support SwinV2/ViT transformer architectures
        providers = ["CPUExecutionProvider"]

        self._model      = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._model.get_inputs()[0].name

        with open(tags_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx  = len(self._tag_names)
                name = row["name"]
                cat  = int(row.get("category", 0))
                self._tag_names.append(name)
                if cat == 9:
                    # rating:safe / rating:questionable / rating:explicit / …
                    self._rating_idx[name.replace("rating:", "").strip()] = idx
                elif cat == 0 and name in self.EXPLICIT_TAGS:
                    self._explicit_idx.append(idx)

        print(f"    {len(self._tag_names)} tags loaded  "
              f"({len(self._explicit_idx)} explicit, "
              f"{len(self._rating_idx)} rating)")

    # ── inference ──────────────────────────────────────────────────────────────

    def _preprocess(self, img_bytes: bytes) -> np.ndarray:
        """Resize to 448×448 with white padding, BGR float32."""
        img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        target = 448
        img.thumbnail((target, target), Image.LANCZOS)
        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        canvas.paste(img, ((target - img.width) // 2, (target - img.height) // 2))
        arr = np.array(canvas, dtype=np.float32)[:, :, ::-1]  # RGB → BGR
        return arr[np.newaxis]  # (1, 448, 448, 3)

    def _run(self, img_bytes: bytes) -> np.ndarray:
        arr = self._preprocess(img_bytes)
        return self._model.run(None, {self._input_name: arr})[0][0]  # (n_tags,)

    def score(self, img_bytes: bytes) -> float:
        """
        0–1 explicit score.
        = max(highest explicit-tag score,  rating:explicit × 0.7)
        The 0.7 weight prevents rating:explicit alone from over-triggering —
        it fires on 'mature' pages too, not just fully explicit ones.
        """
        out      = self._run(img_bytes)
        tag_max  = max((float(out[i]) for i in self._explicit_idx), default=0.0)
        r_exp    = float(out[self._rating_idx["explicit"]]) if "explicit" in self._rating_idx else 0.0
        return max(tag_max, r_exp * 0.7)

    def top_tags(self, img_bytes: bytes, min_score: float = 0.05) -> dict[str, float]:
        """All explicit tags + rating tags above min_score (for debug output)."""
        out    = self._run(img_bytes)
        result = {}
        for i in self._explicit_idx:
            s = float(out[i])
            if s >= min_score:
                result[self._tag_names[i]] = s
        for name, i in self._rating_idx.items():
            s = float(out[i])
            if s >= min_score:
                result[f"rating:{name}"] = s
        return dict(sorted(result.items(), key=lambda x: -x[1]))


# ─── YOLO detector (optional anime-specific weights) ──────────────────────────

class YOLODetector(BaseDetector):
    name = "yolo"

    def __init__(self, weights_path: str):
        self._weights = weights_path
        self._model   = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("  [YOLO] ultralytics not installed. Run: pip install ultralytics")
            raise
        print(f"  Loading YOLO model: {self._weights}")
        self._model = YOLO(self._weights)

    def detect(self, img_bytes: bytes) -> list[Detection]:
        img     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results = self._model(img, verbose=False)
        out     = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                out.append(Detection(
                    r.names[int(box.cls[0])], float(box.conf[0]),
                    (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    source="yolo",
                ))
        return out


# ─── Tile scanner ─────────────────────────────────────────────────────────────

class TileScanner:
    """
    Slides a window across pages already flagged as explicit and scores each
    tile with WD14.  Smaller tiles (256 px default) give much better precision
    than the previous 512 px approach — the censored area more closely matches
    the actual explicit region rather than blacking out half the page.
    """

    def __init__(
        self,
        tagger:         WD14Tagger,
        tile_size:      int   = 256,
        stride:         int   = 128,
        tile_threshold: float = 0.40,
    ):
        self._tagger    = tagger
        self._tile_size = tile_size
        self._stride    = stride
        self._threshold = tile_threshold

    def scan(self, img_bytes: bytes) -> list[Detection]:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        ts   = self._tile_size

        def positions(length: int) -> list[int]:
            pts = list(range(0, max(0, length - ts), self._stride))
            if length >= ts:
                pts.append(length - ts)
            elif not pts:
                pts = [0]
            return list(dict.fromkeys(pts))

        detections = []
        for y in positions(h):
            for x in positions(w):
                x2, y2 = min(x + ts, w), min(y + ts, h)
                tile   = img.crop((x, y, x2, y2))
                buf    = io.BytesIO()
                tile.save(buf, format="PNG")
                s = self._tagger.score(buf.getvalue())
                if s >= self._threshold:
                    detections.append(Detection(
                        "tile_explicit", s, (x, y, x2 - x, y2 - y), source="tile",
                    ))
        return detections


# ─── Box merging ──────────────────────────────────────────────────────────────

def merge_boxes(detections: list[Detection], gap: int = 8) -> list[Detection]:
    """
    Merge overlapping / adjacent bounding boxes into minimal covering boxes.
    'gap' px tolerance ensures touching tiles merge cleanly.
    Result: a handful of clean rectangles instead of a grid of tiles.
    """
    if len(detections) <= 1:
        return detections

    # Expand each box by gap to allow touching boxes to merge
    rects = []
    for d in detections:
        x, y, w, h = d.box
        rects.append([
            max(0, x - gap), max(0, y - gap),
            x + w + gap,     y + h + gap,
            d.score, d.source, d.label,
        ])

    changed = True
    while changed:
        changed = False
        merged  = [False] * len(rects)
        new_rects: list[list] = []
        for i, a in enumerate(rects):
            if merged[i]:
                continue
            a = a[:]
            for j in range(i + 1, len(rects)):
                if merged[j]:
                    continue
                b = rects[j]
                # overlap check
                if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                    a[0] = min(a[0], b[0])
                    a[1] = min(a[1], b[1])
                    a[2] = max(a[2], b[2])
                    a[3] = max(a[3], b[3])
                    a[4] = max(a[4], b[4])
                    merged[j] = True
                    changed   = True
            new_rects.append(a)
        rects = new_rects

    return [
        Detection(r[6], r[4], (r[0], r[1], r[2] - r[0], r[3] - r[1]), r[5])
        for r in rects
    ]


# ─── Text restoration ─────────────────────────────────────────────────────────

class TextRestorer:
    """
    Post-censoring step: uses EasyOCR's CRAFT-based text detector to find real
    text regions in the original image, then for each region that overlaps a
    censor box, draws a white rectangle + re-renders the OCR'd text on the
    censored image.

    Why EasyOCR over Tesseract:
      • CRAFT neural text detector distinguishes actual text from artwork.
      • Returns line-level boxes (not per-character fragments).
      • Handles manga bubble text, angled text, and mixed-size lettering.
      • Won't misread high-contrast art lines as letters.

    No original pixels are ever restored — zero NSFW leak risk.
    """

    # Max text-region height as a fraction of the image height.
    # Manga dialogue text is small; very tall "detections" are almost always art.
    MAX_TEXT_HEIGHT_FRAC = 0.12

    def __init__(self, lang: str = "en"):
        self._lang   = [l.strip() for l in lang.split("+")]
        self._reader = None   # lazy-loaded on first use

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError:
                return None
            # macOS Python 3.13 ships without system SSL certs configured.
            # Patch ssl to use certifi's bundle before EasyOCR downloads its models.
            import ssl
            try:
                import certifi
                ssl._create_default_https_context = lambda: ssl.create_default_context(
                    cafile=certifi.where()
                )
            except ImportError:
                pass
            print(f"  [TextRestorer] Loading EasyOCR (lang={self._lang})...")
            # gpu=False avoids CoreML/CUDA issues on Mac; still fast enough for manga pages
            self._reader = easyocr.Reader(self._lang, gpu=False, verbose=False)
        return self._reader

    @staticmethod
    def _bbox_to_xywh(pts) -> tuple[int, int, int, int]:
        """Convert EasyOCR's 4-point [[x,y]…] bbox to (x, y, w, h)."""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = int(min(xs)), int(min(ys))
        return x, y, int(max(xs)) - x, int(max(ys)) - y

    @staticmethod
    def _overlaps(ax: int, ay: int, aw: int, ah: int,
                  bx: int, by: int, bw: int, bh: int) -> bool:
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        for path in (
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    def restore(
        self,
        original_bytes: bytes,
        censored_bytes: bytes,
        censor_boxes:   list[tuple],  # [(x, y, w, h), …]
    ) -> bytes:
        """
        For every text region detected by EasyOCR that overlaps a censor box:
          1. Draw a white filled rectangle at the region's bounding box.
          2. Render the OCR'd text in black on top.
        No original image pixels are used.
        """
        reader = self._get_reader()
        if reader is None:
            print("  [TextRestorer] easyocr not installed — skipping.\n"
                  "  Install with:  pip install easyocr")
            return censored_bytes

        if not censor_boxes:
            return censored_bytes

        orig = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        img_h = orig.height
        max_text_h = int(img_h * self.MAX_TEXT_HEIGHT_FRAC)

        try:
            results = reader.readtext(np.array(orig))
            # results: [(bbox_pts, text, confidence), …]
        except Exception as exc:
            print(f"  [TextRestorer] OCR error: {exc}")
            return censored_bytes

        cens = Image.open(io.BytesIO(censored_bytes)).convert("RGB")
        draw = ImageDraw.Draw(cens)
        restored = 0

        for (pts, text, conf) in results:
            text = text.strip()
            if not text or conf < 0.3:
                continue

            wx, wy, ww, wh = self._bbox_to_xywh(pts)
            if ww <= 0 or wh <= 0:
                continue

            # Skip detections taller than max_text_h — these are almost always
            # high-contrast art regions mistaken for text, not speech-bubble dialogue.
            if wh > max_text_h:
                continue

            # Only act on regions that overlap at least one censor box
            if not any(
                self._overlaps(wx, wy, ww, wh, cx, cy, cw, ch)
                for cx, cy, cw, ch in censor_boxes
            ):
                continue

            # White background box, then black text
            draw.rectangle([wx, wy, wx + ww, wy + wh], fill=(255, 255, 255))
            font_size = max(8, int(wh * 0.80))
            font      = self._load_font(font_size)
            draw.text((wx + 2, wy + 2), text, fill=(0, 0, 0), font=font)
            restored += 1

        if restored:
            print(f"      [text-restore] restored {restored} text region(s)")

        out = io.BytesIO()
        cens.save(out, format="PNG")
        return out.getvalue()


# ─── Debug annotation ─────────────────────────────────────────────────────────

_COLORS = {
    "nudenet": (220, 50,  50),
    "yolo":    (50,  200, 50),
    "tile":    (50,  100, 220),
}


def save_debug_image(
    img_bytes:     bytes,
    detections:    list[Detection],
    eff_threshold: float,
    gated:         bool,
    page_num:      int,
    debug_dir:     Path,
) -> None:
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        color  = _COLORS.get(det.source, (255, 200, 0))
        x, y, w, h = det.box
        gated_tile = gated and det.source == "tile"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        txt = f"{'GATE ' if gated_tile else ''}{det.source}:{det.label} {det.score:.2f}"
        tx, ty = x + 2, max(0, y - 22)
        draw.text((tx + 1, ty + 1), txt, fill=(0, 0, 0), font=font)
        draw.text((tx,     ty),     txt, fill=color,     font=font)
        if not gated_tile and det.score >= eff_threshold:
            draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0), width=1)

    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / f"debug_page_{page_num:03d}.png"
    img.save(out)
    print(f"      → debug: {out}")


# ─── Image helpers ────────────────────────────────────────────────────────────

def extract_native_image(page: fitz.Page, doc: fitz.Document) -> tuple[bytes, str] | None:
    imgs = page.get_images(full=True)
    if len(imgs) != 1:
        return None
    img_dict = doc.extract_image(imgs[0][0])
    return img_dict["image"], img_dict["ext"]


def render_page(page: fitz.Page, dpi: int) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def apply_censors(
    img_bytes:  bytes,
    detections: list[Detection],
    threshold:  float,
    do_merge:   bool = True,
) -> tuple[bytes, int, list[tuple]]:
    """
    Filter detections by threshold, merge overlapping boxes, then draw.
    Returns (censored_image_bytes, n_regions_drawn, drawn_boxes).
    drawn_boxes is a list of (x, y, w, h) tuples for every rectangle drawn.
    """
    matching = [d for d in detections if d.score >= threshold]
    if not matching:
        return img_bytes, 0, []

    if do_merge:
        matching = merge_boxes(matching)

    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    drawn_boxes: list[tuple] = []
    for det in matching:
        x, y, w, h = det.box
        draw.rectangle([x, y, x + w, y + h], fill="black")
        drawn_boxes.append((x, y, w, h))

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue(), len(matching), drawn_boxes


# ─── Core processing ──────────────────────────────────────────────────────────

def process_pdf(
    input_path:        Path,
    output_path:       Optional[Path],
    detectors:         list[BaseDetector],
    tagger:            Optional[WD14Tagger],
    tile_scanner:      Optional[TileScanner],
    nudenet_threshold: float,
    page_threshold:    float,
    tile_cls_gate:     float,
    fallback_dpi:      int,
    debug_pages:       Optional[set[int]],
    debug_dir:         Path,
    debug_only:        bool,
    text_restorer:     Optional["TextRestorer"] = None,
) -> None:
    print(f"\n  Processing: {input_path.name}")

    doc          = fitz.open(str(input_path))
    total_pages  = len(doc)
    total_censored = 0
    page_data: list[tuple[bytes, fitz.Rect]] = []

    for page_num in range(total_pages):
        page         = doc[page_num]
        page_1idx    = page_num + 1
        is_debug     = debug_pages is not None and page_1idx in debug_pages

        if debug_only and not is_debug:
            continue

        result = extract_native_image(page, doc)
        if result:
            img_bytes, _ext = result
            source = "native"
        else:
            img_bytes = render_page(page, fallback_dpi)
            source    = f"rendered@{fallback_dpi}dpi"

        # ── NudeNet / YOLO bounding-box detectors ──────────────────────────────
        all_detections: list[Detection] = []
        for det in detectors:
            all_detections.extend(det.detect(img_bytes))

        # ── WD14 page-level gate ────────────────────────────────────────────────
        page_score  = 0.0
        gated_out   = False
        note        = ""

        if tagger is not None:
            page_score = tagger.score(img_bytes)

            if page_score >= tile_cls_gate:
                # Run tile scanner
                if tile_scanner is not None:
                    tile_hits = tile_scanner.scan(img_bytes)
                    all_detections.extend(tile_hits)
                    if tile_hits:
                        note += f" [tiles:{len(tile_hits)}]"
                note = f" [wd14={page_score:.2f}]{note}"
            else:
                gated_out = True
                note = f" [wd14={page_score:.2f}<gate]"
                # In debug mode, still run tile scanner to show potential hits
                if is_debug and tile_scanner is not None:
                    tile_hits = tile_scanner.scan(img_bytes)
                    all_detections.extend(tile_hits)
                    if tile_hits:
                        note += f" [tiles:{len(tile_hits)} GATED]"

        # ── Threshold: NudeNet uses nudenet_threshold, tiles use their own ──────
        # Combine: tile detections already filtered by tile_scanner's threshold;
        # we keep them as-is (score >= tile_threshold already).
        # NudeNet detections use nudenet_threshold.
        effective_detections = [
            d for d in all_detections
            if (d.source == "tile" or d.score >= nudenet_threshold)
            and not (gated_out and d.source == "tile")  # drop gated tile hits in real run
        ]

        # ── Debug output ──────────────────────────────────────────────────────
        if is_debug:
            tags = tagger.top_tags(img_bytes) if tagger else {}
            print(f"    Page {page_1idx:>3}  [{source}]{note}")
            print(f"      WD14 page score : {page_score:.4f}  (gate={tile_cls_gate})")
            if tags:
                print(f"      WD14 top tags  : " +
                      "  ".join(f"{k}={v:.2f}" for k, v in list(tags.items())[:8]))
            print(f"      Detections ({len(all_detections)} total, "
                  f"{len(effective_detections)} effective):")
            for d in sorted(all_detections, key=lambda x: -x.score):
                if gated_out and d.source == "tile":
                    mark = "  GATE  "
                elif d in effective_detections or (d.source == "tile"):
                    mark = "✓ CENSOR" if d.source == "tile" or d.score >= nudenet_threshold else "  skip  "
                else:
                    mark = "  skip  "
                print(f"        [{mark}] {d.source:<8} {d.label:<30} {d.score:.3f}  {d.box}")
            save_debug_image(img_bytes, all_detections, nudenet_threshold, gated_out, page_1idx, debug_dir)
            if debug_only:
                continue

        pre_censor_bytes = img_bytes
        img_bytes, n_censored, drawn_boxes = apply_censors(
            img_bytes, effective_detections, threshold=0.0
        )
        if n_censored and text_restorer is not None:
            img_bytes = text_restorer.restore(pre_censor_bytes, img_bytes, drawn_boxes)
        total_censored += n_censored
        page_data.append((img_bytes, page.rect))

        if not is_debug:
            status = f"censored {n_censored}" if n_censored else "clean"
            print(f"    Page {page_1idx:>3}/{total_pages}  [{source}]{note}  —  {status}")

    doc.close()

    if debug_only:
        print(f"\n  Debug images → {debug_dir}")
        return

    out_doc = fitz.open()
    for img_bytes, rect in page_data:
        new_page = out_doc.new_page(width=rect.width, height=rect.height)
        new_page.insert_image(new_page.rect, stream=img_bytes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(output_path))
    out_doc.close()
    print(f"  Saved → {output_path}  ({total_censored} total regions censored)")


# ─── Path helpers ─────────────────────────────────────────────────────────────

def collect_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in {path}")
            sys.exit(1)
        return pdfs
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    print(f"Error: '{path}' is not a PDF file or directory.")
    sys.exit(1)


def parse_page_range(s: str) -> set[int]:
    """'41-43' → {41,42,43}  |  '7,19' → {7,19}  |  '1-3,7' → {1,2,3,7}"""
    pages: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return pages


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Censor NSFW regions in manga/anime PDFs (WD14 + NudeNet).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
tuning:
  Lower --page-threshold → more pages get tile-scanned (catch more, risk more noise)
  Lower --tile-threshold → more tiles within a page get censored

debug workflow:
  1. python censor.py manga.pdf --debug-pages 41-43
       Saves annotated PNGs to ./debug/ and prints WD14 tag scores.
       Shows which tiles fired and why, without writing an output PDF.
  2. Adjust thresholds until the boxes look right.
  3. Full run: python censor.py manga.pdf

examples:
  python censor.py manga.pdf
  python censor.py manga.pdf --debug-pages 41-43
  python censor.py manga.pdf --page-threshold 0.15 --tile-threshold 0.35
  python censor.py manga.pdf --yolo-weights ./anime_nsfw.pt
  python censor.py ./folder/ --output ./censored/
        """,
    )

    parser.add_argument("input",  type=Path, help="PDF file or directory.")
    parser.add_argument("--output", "-o", type=Path, default=None)

    thr = parser.add_argument_group("thresholds")
    thr.add_argument("--page-threshold",   type=float, default=0.20,
                     help="WD14 score to run tile scan on a page.  Default: 0.20")
    thr.add_argument("--tile-threshold",   type=float, default=0.40,
                     help="WD14 score for a tile to be censored.  Default: 0.40")
    thr.add_argument("--nudenet-threshold", type=float, default=0.50,
                     help="Confidence for NudeNet detections.  Default: 0.50")
    thr.add_argument("--tile-cls-gate",    type=float, default=0.02,
                     help="Minimum WD14 page score to enable tile scan. Default: 0.02")

    mdl = parser.add_argument_group("models")
    mdl.add_argument("--no-tile",    action="store_true", help="Disable tile scanner.")
    mdl.add_argument("--no-wd14",    action="store_true", help="Disable WD14 (NudeNet only).")
    mdl.add_argument("--wd14-model", type=str,
                     default=WD14Tagger.DEFAULT_REPO,
                     help=f"HuggingFace repo for WD14 model.  Default: {WD14Tagger.DEFAULT_REPO}")
    mdl.add_argument("--yolo-weights", type=str, default=None, metavar="PATH",
                     help="YOLOv8 .pt weights for an anime-specific NSFW detector.")

    tile = parser.add_argument_group("tile scanner")
    tile.add_argument("--tile-size",   type=int, default=256,
                      help="Tile size in pixels.  Smaller = more precise.  Default: 256")
    tile.add_argument("--tile-stride", type=int, default=128,
                      help="Step between tiles.  Smaller = more overlap.  Default: 128")

    txt = parser.add_argument_group("text restoration")
    txt.add_argument("--no-restore-text", action="store_true",
                     help="Disable text restoration (on by default).  "
                          "Requires easyocr:  pip install easyocr")
    txt.add_argument("--restore-text-lang", type=str, default="en", metavar="LANG",
                     help="EasyOCR language code(s).  Default: en  "
                          "(use 'ja' for Japanese, 'ja+en' for both)")

    dbg = parser.add_argument_group("debug")
    dbg.add_argument("--debug-pages", type=str, default=None, metavar="RANGE",
                     help="Only process these pages and save annotated PNGs (no output PDF).  "
                          "Examples: 41-43  or  7  or  1-3,7")
    dbg.add_argument("--debug-dir", type=Path, default=Path("./debug"))

    parser.add_argument("--fallback-dpi", type=int, default=300)
    parser.add_argument("--labels",       nargs="+", default=None, metavar="LABEL")

    args = parser.parse_args()

    labels      = set(args.labels) if args.labels else NUDENET_LABELS
    debug_pages = parse_page_range(args.debug_pages) if args.debug_pages else None
    debug_only  = debug_pages is not None

    # Build stack
    detectors: list[BaseDetector] = [NudeNetDetector(labels)]
    if args.yolo_weights:
        detectors.append(YOLODetector(args.yolo_weights))

    tagger:       Optional[WD14Tagger]  = None
    tile_scanner: Optional[TileScanner] = None

    if not args.no_wd14:
        tagger = WD14Tagger(args.wd14_model)
        if not args.no_tile:
            tile_scanner = TileScanner(
                tagger,
                tile_size      = args.tile_size,
                stride         = args.tile_stride,
                tile_threshold = args.tile_threshold,
            )

    text_restorer: Optional[TextRestorer] = None
    if not args.no_restore_text:
        text_restorer = TextRestorer(lang=args.restore_text_lang)

    # Load all models upfront
    print("Loading models...")
    for det in detectors:
        det.load()
    if tagger:
        tagger.load()

    # Paths
    input_path = args.input.expanduser().resolve()
    pdfs       = collect_pdfs(input_path)

    if debug_only:
        output_paths = [None] * len(pdfs)
    elif len(pdfs) == 1:
        if args.output:
            output_paths = [args.output.expanduser().resolve()]
        else:
            p = pdfs[0]
            output_paths = [p.parent / f"{p.stem}_censored{p.suffix}"]
    else:
        out_dir = (args.output.expanduser().resolve() if args.output
                   else input_path.parent / f"{input_path.name}_censored")
        output_paths = [out_dir / p.name for p in pdfs]

    # Summary
    print(f"\nDetectors         : {', '.join(d.name for d in detectors)}"
          + (" + WD14 tile scanner" if tile_scanner else "")
          + (" [WD14 gate only]"    if tagger and not tile_scanner else ""))
    print(f"Page threshold    : {args.page_threshold}  (WD14 gate to run tiles)")
    print(f"Tile threshold    : {args.tile_threshold}  (WD14 score to censor a tile)")
    print(f"NudeNet threshold : {args.nudenet_threshold}")
    print(f"Tile size / stride: {args.tile_size} px / {args.tile_stride} px")
    print(f"Text restoration  : {'disabled' if args.no_restore_text else f'enabled (lang={args.restore_text_lang})'}")
    if debug_only:
        print(f"Debug pages       : {sorted(debug_pages)}")
    print(f"Files to process  : {len(pdfs)}")

    for pdf_path, out_path in zip(pdfs, output_paths):
        process_pdf(
            input_path        = pdf_path,
            output_path       = out_path,
            detectors         = detectors,
            tagger            = tagger,
            tile_scanner      = tile_scanner,
            nudenet_threshold = args.nudenet_threshold,
            page_threshold    = args.page_threshold,
            tile_cls_gate     = args.tile_cls_gate,
            fallback_dpi      = args.fallback_dpi,
            debug_pages       = debug_pages,
            debug_dir         = args.debug_dir,
            debug_only        = debug_only,
            text_restorer     = text_restorer,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
