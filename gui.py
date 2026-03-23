#!/usr/bin/env python3
"""
gui.py — Tkinter UI for censor.py

Wraps all CLI flags in a graphical interface. The underlying censor.py command
is built from the UI state and run as a subprocess — identical to typing it in
the terminal. Live output is streamed into the log panel.

Run with:
    python3 gui.py          (uses system python3)
    source .venv/bin/activate && python3 gui.py   (recommended)
"""

import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, font, scrolledtext, ttk

SCRIPT  = Path(__file__).parent / "censor.py"
PYTHON  = sys.executable         # same interpreter that launched the GUI


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _label(parent, text, **kw):
    return tk.Label(parent, text=text, anchor="w", **kw)


def _entry(parent, var, width=42, **kw):
    return tk.Entry(parent, textvariable=var, width=width, **kw)


def _browse_file(var):
    p = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
    if p:
        var.set(p)


def _browse_dir(var):
    p = filedialog.askdirectory()
    if p:
        var.set(p)


def _section(parent, title):
    """Labelled frame used as a visual section divider."""
    return ttk.LabelFrame(parent, text=f"  {title}  ", padding=(10, 6))


# ─── Main window ──────────────────────────────────────────────────────────────

class CensorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Content Cleaner — Manga Censor Tool")
        self.resizable(True, True)
        self.minsize(680, 700)

        self._proc:   subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

        self._build_vars()
        self._build_ui()
        # Register traces after UI is built so _cmd_text exists when they fire
        for v in vars(self).values():
            if isinstance(v, tk.Variable):
                v.trace_add("write", lambda *_: self._update_command_preview())
        self._update_command_preview()

    # ── Variable definitions ──────────────────────────────────────────────────

    def _build_vars(self):
        # I/O
        self.v_input          = tk.StringVar()
        self.v_output         = tk.StringVar()

        # Thresholds
        self.v_page_thr       = tk.DoubleVar(value=0.20)
        self.v_tile_thr       = tk.DoubleVar(value=0.40)
        self.v_nudenet_thr    = tk.DoubleVar(value=0.50)
        self.v_tile_cls_gate  = tk.DoubleVar(value=0.02)

        # Tile scanner
        self.v_tile_size      = tk.IntVar(value=256)
        self.v_tile_stride    = tk.IntVar(value=128)

        # Models
        self.v_no_wd14        = tk.BooleanVar(value=False)
        self.v_no_tile        = tk.BooleanVar(value=False)
        self.v_wd14_model     = tk.StringVar(value="SmilingWolf/wd-v1-4-swinv2-tagger-v2")
        self.v_yolo_weights   = tk.StringVar(value="")

        # Text restoration
        self.v_no_restore     = tk.BooleanVar(value=False)
        self.v_restore_lang   = tk.StringVar(value="en")

        # Debug
        self.v_debug_pages    = tk.StringVar(value="")
        self.v_debug_dir      = tk.StringVar(value="./debug")

        # Misc
        self.v_fallback_dpi   = tk.IntVar(value=300)
        self.v_labels         = tk.StringVar(value="")

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Scrollable outer container ─────────────────────────────────────
        # All widgets live inside `inner` (a Frame inside a Canvas), making
        # the entire UI scrollable when the window is too small to show it all.
        outer = tk.Frame(self)
        outer.pack(fill="both", expand=True)

        vbar = ttk.Scrollbar(outer, orient="vertical")
        vbar.pack(side="right", fill="y")

        self._scroll_canvas = tk.Canvas(
            outer, yscrollcommand=vbar.set, highlightthickness=0
        )
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        vbar.config(command=self._scroll_canvas.yview)

        inner = tk.Frame(self._scroll_canvas)
        self._scroll_win = self._scroll_canvas.create_window(
            (0, 0), window=inner, anchor="nw"
        )

        # Keep scroll region and inner width in sync with canvas size
        inner.bind("<Configure>", lambda e: self._scroll_canvas.configure(
            scrollregion=self._scroll_canvas.bbox("all")))
        self._scroll_canvas.bind("<Configure>", lambda e: self._scroll_canvas.itemconfig(
            self._scroll_win, width=e.width))

        # Mousewheel scrolls the outer canvas unless the cursor is over the log
        def _mwheel(e):
            self._scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        self._scroll_canvas.bind_all("<MouseWheel>", _mwheel)
        # macOS two-finger scroll also fires Button-4/5
        self._scroll_canvas.bind_all("<Button-4>", lambda e: self._scroll_canvas.yview_scroll(-1, "units"))
        self._scroll_canvas.bind_all("<Button-5>", lambda e: self._scroll_canvas.yview_scroll( 1, "units"))

        # ── Notebook tabs (inside inner) ───────────────────────────────────
        nb = ttk.Notebook(inner)
        nb.pack(fill="both", expand=False, padx=10, pady=(10, 0))

        tab_main  = ttk.Frame(nb, padding=6)
        tab_model = ttk.Frame(nb, padding=6)
        tab_debug = ttk.Frame(nb, padding=6)
        nb.add(tab_main,  text="  Main  ")
        nb.add(tab_model, text="  Models  ")
        nb.add(tab_debug, text="  Debug  ")

        self._build_tab_main(tab_main)
        self._build_tab_model(tab_model)
        self._build_tab_debug(tab_debug)

        # ── Command preview ────────────────────────────────────────────────
        prev_frame = _section(inner, "Command Preview")
        prev_frame.pack(fill="x", padx=10, pady=(8, 0))

        self._cmd_text = tk.Text(
            prev_frame, height=3, wrap="word", state="disabled",
            bg="#1e1e1e", fg="#d4d4d4",
            font=font.Font(family="Menlo", size=11),
        )
        self._cmd_text.pack(fill="x")

        # ── Run / Stop buttons ─────────────────────────────────────────────
        btn_frame = tk.Frame(inner)
        btn_frame.pack(fill="x", padx=10, pady=8)

        self._btn_run = tk.Button(
            btn_frame, text="▶  Run", width=14,
            bg="#2d7d46", fg="white",
            activebackground="#1e6035", activeforeground="white",
            disabledforeground="#666666",
            font=font.Font(weight="bold", size=12),
            relief="raised", bd=2,
            command=self._run,
        )
        self._btn_run.pack(side="left", padx=(0, 6))

        self._btn_stop = tk.Button(
            btn_frame, text="■  Stop", width=10,
            bg="#555555", fg="#888888",          # starts visually muted (disabled)
            activebackground="#96281b", activeforeground="white",
            disabledforeground="#666666",
            font=font.Font(weight="bold", size=12),
            relief="raised", bd=2,
            state="disabled", command=self._stop,
        )
        self._btn_stop.pack(side="left")

        self._status = tk.Label(btn_frame, text="", fg="#555", anchor="w")
        self._status.pack(side="left", padx=12)

        # ── Log output ─────────────────────────────────────────────────────
        log_frame = _section(inner, "Output Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        self._log = scrolledtext.ScrolledText(
            log_frame, wrap="word", state="disabled",
            bg="#1e1e1e", fg="#cccccc",
            font=font.Font(family="Menlo", size=11),
            height=18,
        )
        self._log.pack(fill="both", expand=True)
        self._log.tag_config("green",  foreground="#4ec9b0")
        self._log.tag_config("yellow", foreground="#dcdcaa")
        self._log.tag_config("red",    foreground="#f44747")
        self._log.tag_config("dim",    foreground="#666666")

        # Suspend outer scroll while mouse is inside the log so its own
        # scrollbar works normally; restore when mouse leaves.
        self._log.bind("<Enter>", lambda e: self._scroll_canvas.unbind_all("<MouseWheel>"))
        self._log.bind("<Leave>", lambda e: self._scroll_canvas.bind_all("<MouseWheel>", _mwheel))

        btn_clear = tk.Button(log_frame, text="Clear log", command=self._clear_log)
        btn_clear.pack(anchor="e", pady=(4, 0))

        # ── Disclaimer banner (bottom of inner — scroll down to always reach it)
        disclaimer = tk.Frame(inner, bg="#3a2a00", bd=1, relief="solid")
        disclaimer.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(
            disclaimer,
            text="⚠  WARNING",
            bg="#3a2a00", fg="#ffd700",
            font=font.Font(weight="bold", size=11),
            anchor="w",
        ).pack(side="left", padx=(10, 6), pady=8)

        tk.Label(
            disclaimer,
            text="This tool is not perfect and does not guarantee 100% accurate results. "
                 "Explicit content may be missed or clean content may be incorrectly censored. "
                 "User discretion and manual review of output files is strongly advised.",
            bg="#3a2a00", fg="#ffcc44",
            font=font.Font(size=10),
            wraplength=540,
            justify="left",
            anchor="w",
        ).pack(side="left", padx=(0, 10), pady=8)

    def _build_tab_main(self, parent):
        # ── Input ──────────────────────────────────────────────────────────
        io_sec = _section(parent, "Input / Output")
        io_sec.pack(fill="x", pady=(0, 8))
        io_sec.columnconfigure(1, weight=1)

        _label(io_sec, "Input (file or folder)").grid(row=0, column=0, sticky="w", pady=3)
        _entry(io_sec, self.v_input).grid(row=0, column=1, sticky="ew", padx=6)
        tk.Button(io_sec, text="Browse file…",   command=lambda: _browse_file(self.v_input)).grid(row=0, column=2, padx=(0, 2))
        tk.Button(io_sec, text="Browse folder…", command=lambda: _browse_dir(self.v_input)).grid(row=0, column=3)

        _label(io_sec, "Output (optional)").grid(row=1, column=0, sticky="w", pady=3)
        _entry(io_sec, self.v_output).grid(row=1, column=1, sticky="ew", padx=6)
        tk.Button(io_sec, text="Browse…", command=lambda: _browse_dir(self.v_output)).grid(row=1, column=2, columnspan=2, sticky="w")

        # ── Thresholds ─────────────────────────────────────────────────────
        thr_sec = _section(parent, "Thresholds")
        thr_sec.pack(fill="x", pady=(0, 8))

        rows = [
            ("Page threshold",    self.v_page_thr,      0.00, 1.00, "WD14 minimum score to run tile scan on a page"),
            ("Tile threshold",    self.v_tile_thr,      0.00, 1.00, "WD14 score for a tile to be censored"),
            ("NudeNet threshold", self.v_nudenet_thr,   0.00, 1.00, "Confidence for NudeNet bounding-box detections"),
            ("Tile cls gate",     self.v_tile_cls_gate, 0.00, 0.20, "Minimum WD14 page score to enable tile scan at all"),
        ]
        for r, (label, var, lo, hi, tip) in enumerate(rows):
            _label(thr_sec, label).grid(row=r, column=0, sticky="w", pady=2)
            tk.Scale(thr_sec, variable=var, from_=lo, to=hi, resolution=0.01,
                     orient="horizontal", length=240,
                     command=lambda *_: None).grid(row=r, column=1, padx=6)
            tk.Spinbox(thr_sec, textvariable=var, from_=lo, to=hi,
                       increment=0.01, width=6, format="%.2f").grid(row=r, column=2)
            _label(thr_sec, tip, fg="#777").grid(row=r, column=3, sticky="w", padx=(10, 0))

        # ── Tile scanner ───────────────────────────────────────────────────
        tile_sec = _section(parent, "Tile Scanner")
        tile_sec.pack(fill="x", pady=(0, 8))

        _label(tile_sec, "Tile size (px)").grid(row=0, column=0, sticky="w", pady=2)
        tk.Spinbox(tile_sec, textvariable=self.v_tile_size,
                   from_=64, to=512, increment=32, width=7).grid(row=0, column=1, sticky="w", padx=6)
        _label(tile_sec, "Smaller = more precise censoring", fg="#777").grid(row=0, column=2, sticky="w")

        _label(tile_sec, "Tile stride (px)").grid(row=1, column=0, sticky="w", pady=2)
        tk.Spinbox(tile_sec, textvariable=self.v_tile_stride,
                   from_=32, to=256, increment=32, width=7).grid(row=1, column=1, sticky="w", padx=6)
        _label(tile_sec, "Smaller = more overlap between tiles", fg="#777").grid(row=1, column=2, sticky="w")

        # ── Text restoration ───────────────────────────────────────────────
        txt_sec = _section(parent, "Text Restoration")
        txt_sec.pack(fill="x")

        ttk.Checkbutton(txt_sec, text="Disable text restoration",
                        variable=self.v_no_restore).grid(row=0, column=0, sticky="w")
        _label(txt_sec, "(on by default — restores speech-bubble text over censor boxes)",
               fg="#777").grid(row=0, column=1, sticky="w", padx=8)

        _label(txt_sec, "OCR language").grid(row=1, column=0, sticky="w", pady=4)
        tk.Entry(txt_sec, textvariable=self.v_restore_lang, width=14).grid(row=1, column=1, sticky="w", padx=8)
        _label(txt_sec, "en · ja · ja+en  etc.", fg="#777").grid(row=1, column=2, sticky="w")

    def _build_tab_model(self, parent):
        wd_sec = _section(parent, "WD14 Tagger  (primary — anime/manga)")
        wd_sec.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(wd_sec, text="Disable WD14 entirely (NudeNet only)",
                        variable=self.v_no_wd14).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(wd_sec, text="Disable tile scanner  (page-level gate only)",
                        variable=self.v_no_tile).grid(row=1, column=0, columnspan=3, sticky="w", pady=4)

        _label(wd_sec, "HuggingFace model repo").grid(row=2, column=0, sticky="w", pady=2)
        _entry(wd_sec, self.v_wd14_model, width=50).grid(row=2, column=1, sticky="w", padx=6)

        yolo_sec = _section(parent, "YOLO  (optional anime-specific weights)")
        yolo_sec.pack(fill="x", pady=(0, 8))
        yolo_sec.columnconfigure(1, weight=1)

        _label(yolo_sec, "Weights path (.pt)").grid(row=0, column=0, sticky="w", pady=2)
        _entry(yolo_sec, self.v_yolo_weights).grid(row=0, column=1, sticky="ew", padx=6)
        tk.Button(yolo_sec, text="Browse…",
                  command=lambda: _browse_file(self.v_yolo_weights)).grid(row=0, column=2)

        misc_sec = _section(parent, "Misc")
        misc_sec.pack(fill="x")

        _label(misc_sec, "Fallback render DPI").grid(row=0, column=0, sticky="w", pady=2)
        tk.Spinbox(misc_sec, textvariable=self.v_fallback_dpi,
                   from_=72, to=600, increment=50, width=7).grid(row=0, column=1, sticky="w", padx=6)
        _label(misc_sec, "Used when native image extraction fails", fg="#777").grid(row=0, column=2, sticky="w")

        _label(misc_sec, "NudeNet labels (space-separated)").grid(row=1, column=0, sticky="w", pady=2)
        _entry(misc_sec, self.v_labels, width=50).grid(row=1, column=1, columnspan=2, sticky="w", padx=6)
        _label(misc_sec, "Leave blank for defaults", fg="#777").grid(row=2, column=1, sticky="w", padx=6)

    def _build_tab_debug(self, parent):
        dbg_sec = _section(parent, "Debug")
        dbg_sec.pack(fill="x", pady=(0, 8))
        dbg_sec.columnconfigure(1, weight=1)

        _label(dbg_sec, "Debug pages").grid(row=0, column=0, sticky="w", pady=2)
        _entry(dbg_sec, self.v_debug_pages, width=24).grid(row=0, column=1, sticky="w", padx=6)
        _label(dbg_sec, "e.g.  41-43  or  7  or  1-3,7\n"
                         "Saves annotated PNGs — no output PDF written.", fg="#777").grid(
            row=0, column=2, sticky="w", padx=6)

        _label(dbg_sec, "Debug output dir").grid(row=1, column=0, sticky="w", pady=2)
        _entry(dbg_sec, self.v_debug_dir).grid(row=1, column=1, sticky="ew", padx=6)
        tk.Button(dbg_sec, text="Browse…",
                  command=lambda: _browse_dir(self.v_debug_dir)).grid(row=1, column=2)

    # ── Command builder ────────────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        cmd = [PYTHON, "-u", str(SCRIPT)]

        inp = self.v_input.get().strip()
        if inp:
            cmd.append(inp)

        out = self.v_output.get().strip()
        if out:
            cmd += ["--output", out]

        # Thresholds (only add if different from defaults)
        def _add_float(flag, var, default):
            if abs(var.get() - default) > 1e-9:
                cmd += [flag, f"{var.get():.2f}"]

        _add_float("--page-threshold",    self.v_page_thr,      0.20)
        _add_float("--tile-threshold",    self.v_tile_thr,      0.40)
        _add_float("--nudenet-threshold", self.v_nudenet_thr,   0.50)
        _add_float("--tile-cls-gate",     self.v_tile_cls_gate, 0.02)

        if self.v_tile_size.get()   != 256:  cmd += ["--tile-size",   str(self.v_tile_size.get())]
        if self.v_tile_stride.get() != 128:  cmd += ["--tile-stride", str(self.v_tile_stride.get())]

        if self.v_no_wd14.get():   cmd.append("--no-wd14")
        if self.v_no_tile.get():   cmd.append("--no-tile")

        wd14 = self.v_wd14_model.get().strip()
        if wd14 and wd14 != "SmilingWolf/wd-v1-4-swinv2-tagger-v2":
            cmd += ["--wd14-model", wd14]

        yolo = self.v_yolo_weights.get().strip()
        if yolo:
            cmd += ["--yolo-weights", yolo]

        if self.v_no_restore.get():
            cmd.append("--no-restore-text")

        lang = self.v_restore_lang.get().strip()
        if lang and lang != "en":
            cmd += ["--restore-text-lang", lang]

        dbg = self.v_debug_pages.get().strip()
        if dbg:
            cmd += ["--debug-pages", dbg]

        dbg_dir = self.v_debug_dir.get().strip()
        if dbg_dir and dbg_dir != "./debug":
            cmd += ["--debug-dir", dbg_dir]

        if self.v_fallback_dpi.get() != 300:
            cmd += ["--fallback-dpi", str(self.v_fallback_dpi.get())]

        labels = self.v_labels.get().strip()
        if labels:
            cmd += ["--labels"] + labels.split()

        return cmd

    def _update_command_preview(self, *_):
        cmd = self._build_command()
        preview = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self._cmd_text.config(state="normal")
        self._cmd_text.delete("1.0", "end")
        self._cmd_text.insert("end", preview)
        self._cmd_text.config(state="disabled")

    # ── Run / Stop ─────────────────────────────────────────────────────────────

    def _run(self):
        inp = self.v_input.get().strip()
        if not inp:
            self._log_append("⚠  No input file or folder selected.\n", "red")
            return

        cmd = self._build_command()
        self._log_append(f"$ {' '.join(repr(c) if ' ' in c else c for c in cmd)}\n", "yellow")

        self._btn_run.config(state="disabled", bg="#555555", fg="#888888")
        self._btn_stop.config(state="normal",   bg="#c0392b", fg="white")
        self._status.config(text="Running…")

        self._thread = threading.Thread(target=self._stream, args=(cmd,), daemon=True)
        self._thread.start()

    def _stream(self, cmd: list[str]):
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._proc.stdout:
                self._log_append(line)
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as exc:
            self._log_append(f"\nError launching process: {exc}\n", "red")
            rc = -1
        finally:
            self._proc = None
            self.after(0, self._on_done, rc)

    def _on_done(self, returncode: int):
        if returncode == 0:
            self._log_append("\n✓ Done.\n", "green")
            self._status.config(text="Done.")
        elif returncode is None or returncode == -1:
            self._status.config(text="Stopped.")
        else:
            self._log_append(f"\n✗ Exited with code {returncode}\n", "red")
            self._status.config(text=f"Error (exit {returncode})")
        self._btn_run.config(state="normal",   bg="#2d7d46", fg="white")
        self._btn_stop.config(state="disabled", bg="#555555", fg="#888888")

    def _stop(self):
        if self._proc:
            self._proc.terminate()
            self._log_append("\n[stopped by user]\n", "red")
        self._btn_stop.config(state="disabled", bg="#555555", fg="#888888")
        self._status.config(text="Stopping…")

    # ── Log helpers ────────────────────────────────────────────────────────────

    def _log_append(self, text: str, tag: str | None = None):
        def _do():
            self._log.config(state="normal")
            if tag:
                self._log.insert("end", text, tag)
            else:
                # Colour-code common patterns
                t = "green" if text.startswith("  Saved") or "Done" in text else \
                    "yellow" if text.strip().startswith("Page") else \
                    "red"    if "Error" in text or "Traceback" in text else \
                    "dim"    if "<gate" in text else None
                self._log.insert("end", text, t or "")
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CensorGUI()
    app.mainloop()
