import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog, Toplevel
from tkinter.filedialog import asksaveasfilename, askdirectory
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageEnhance, ImageGrab, ImageFilter, ImageChops
import os
import math
import numpy as np
import sys
import onnxruntime as ort
from timeit import default_timer as timer
import cv2
from copy import deepcopy
import platform
import json
import threading
import queue
import colorsys


# DPI awareness on Windows: prevents blurriness on high-res displays.
try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass  # Ignore on other platforms/errors.

# Drag and Drop support (preferred method). Fallback to standard Tkinter if module is missing.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    ROOT_CLASS = TkinterDnD.Tk
    DND_AVAILABLE = True
except ImportError:
    print("WARNING: tkinterdnd2 not found. Drag and drop disabled. Run 'pip install tkinterdnd2'")
    ROOT_CLASS = tk.Tk
    DND_AVAILABLE = False

# Constants
DEFAULT_ZOOM_FACTOR = 1.2
MAX_ZOOM_FACTOR = 50.0

# Determine execution environment (frozen executable vs script). Critical for asset location.
if getattr(sys, 'frozen', False):
    SCRIPT_BASE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Working directory: {SCRIPT_BASE_DIR}")
MODEL_ROOT = os.path.join(SCRIPT_BASE_DIR, "Models/")
CONFIG_FILE = os.path.join(SCRIPT_BASE_DIR, "settings.json")

# --- UI THEME PALETTE (Dark Carbon: VS Code style) ---
COLORS = {
    "bg": "#1e1e1e",
    "panel_bg": "#252526",
    "card_bg": "#2d2d30",
    "fg": "#cccccc",
    "header": "#ffffff",
    "border": "#3e3e42",
    "accent": "#007acc",  # Primary blue for selection/focus
    "accent_hover": "#0062a3",
    "add": "#3fb950",  # Green
    "add_hover": "#2ea043",
    "remove": "#D45443",  # Red
    "remove_hover": "#da3633",
    "undo": "#d29922",  # Yellow/Orange
    "undo_hover": "#b0801d",
    "export": "#8957e5",  # Purple
    "export_hover": "#6e40c9",
    "text_dark": "#101010",
    "highlight": "#007acc",
    "dropdown_bg": "#2d2d30",
    "dropdown_fg": "#e0e0e0"
}

STATUS_PROCESSING = "white"
STATUS_NORMAL = "white"

# Editor defaults
PAINT_BRUSH_DIAMETER = 18
UNDO_STEPS = 20
MIN_RECT_SIZE = 12

# --- ONNX Runtime Setup ---
available_providers = ort.get_available_providers()
ONNX_PROVIDERS = []



# Provider priority: 1. CUDA (NVIDIA), 2. DirectML (Windows), 3. CPU (Fallback)
if 'CUDAExecutionProvider' in available_providers:
    ONNX_PROVIDERS.append('CUDAExecutionProvider')
if 'DmlExecutionProvider' in available_providers:
    ONNX_PROVIDERS.append('DmlExecutionProvider')

ONNX_PROVIDERS.append('CPUExecutionProvider')

print(f"Hardware Acceleration: Using {ONNX_PROVIDERS[0]}")


def get_ort_session_options():
    """Configures ONNX Runtime options for performance."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    # Use all available CPU cores for faster CPU inference
    sess_options.intra_op_num_threads = cpu_count
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return sess_options


def set_window_dark_mode(window):
    """Windows-specific hack: Forces the title bar into dark mode (DWMWA_USE_IMMERSIVE_DARK_MODE)."""
    try:
        from ctypes import windll, c_int, byref, sizeof
        window.update()  # Ensure valid HWND
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        get_parent = windll.user32.GetParent

        hwnd = get_parent(window.winfo_id())
        value = c_int(1)
        set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value))
    except Exception as e:
        pass  # Not Windows, or failed. No big deal.


class ModernHelpWindow(tk.Toplevel):
    """Custom-styled modal documentation window. Replaces ugly default messagebox."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(bg=COLORS["bg"])
        self.overrideredirect(True)  # Custom title bar

        # Calculate responsive window geometry based on screen size
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        target_w = max(700, min(1200, int(screen_w * 0.55)))
        target_h = max(500, min(900, int(screen_h * 0.70)))
        self.text_wrap_length = target_w - 300

        x = (screen_w - target_w) // 2
        y = (screen_h - target_h) // 2
        self.geometry(f"{target_w}x{target_h}+{int(x)}+{int(y)}")

        # Main Layout
        self.border_frame = tk.Frame(self, bg=COLORS["accent"], padx=1, pady=1)  # 1px accent border
        self.border_frame.pack(fill="both", expand=True)

        self.main_container = tk.Frame(self.border_frame, bg=COLORS["bg"])
        self.main_container.pack(fill="both", expand=True)

        # Custom Title Bar for dragging
        self.title_bar = tk.Frame(self.main_container, bg=COLORS["panel_bg"], height=35)
        self.title_bar.pack(fill="x", side="top")
        self.title_bar.bind("<Button-1>", self.start_move)
        self.title_bar.bind("<B1-Motion>", self.do_move)

        lbl_title = tk.Label(self.title_bar, text="Documentation & Usage", bg=COLORS["panel_bg"], fg="white", font=("Segoe UI", 10, "bold"))
        lbl_title.pack(side="left", padx=15)
        lbl_title.bind("<Button-1>", self.start_move)

        # Close Button with hover effect
        btn_close = tk.Label(self.title_bar, text="✕", bg=COLORS["panel_bg"], fg=COLORS["fg"], font=("Arial", 11), width=5, cursor="hand2")
        btn_close.pack(side="right", fill="y")
        btn_close.bind("<Button-1>", lambda e: self.close_win())
        btn_close.bind("<Enter>", lambda e: btn_close.config(bg=COLORS["remove"], fg="white"))
        btn_close.bind("<Leave>", lambda e: btn_close.config(bg=COLORS["panel_bg"], fg=COLORS["fg"]))

        # Credits (Nice to have in the title bar)
        credits_frame = tk.Frame(self.title_bar, bg=COLORS["panel_bg"])
        credits_frame.pack(side="right", padx=10)
        credits_frame.bind("<Button-1>", self.start_move)

        lbl_text = tk.Label(credits_frame, text="Made by Alfadoc, based on pricklygorse's project ",
                            bg=COLORS["panel_bg"], fg="#808080", font=("Segoe UI", 9, "italic"))
        lbl_text.pack(side="left")
        lbl_text.bind("<Button-1>", self.start_move)

        lbl_heart = tk.Label(credits_frame, text="♥️",
                             bg=COLORS["panel_bg"], fg=COLORS["remove"], font=("Segoe UI", 10, "bold"))
        lbl_heart.pack(side="left")
        lbl_heart.bind("<Button-1>", self.start_move)

        # Content Area - Scrollable
        header_frame = tk.Frame(self.main_container, bg=COLORS["bg"], padx=30, pady=15)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="Background Remover Pro", font=("Segoe UI", 22, "bold"), bg=COLORS["bg"], fg=COLORS["header"]).pack(anchor="w")
        tk.Label(header_frame, text="Advanced AI Masking & Compositing Workflow", font=("Segoe UI", 11, "italic"), bg=COLORS["bg"], fg=COLORS["accent"]).pack(anchor="w")

        # Scrollable container setup
        self.scroll_container = tk.Frame(self.main_container, bg=COLORS["bg"])
        self.scroll_container.pack(fill="both", expand=True, padx=2, pady=2)

        self.scrollbar = ttk.Scrollbar(self.scroll_container, orient="vertical", style="NoArrow.Vertical.TScrollbar")
        self.canvas = tk.Canvas(self.scroll_container, bg=COLORS["bg"], highlightthickness=0, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.canvas.yview)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner_frame = tk.Frame(self.canvas, bg=COLORS["bg"], padx=30, pady=10)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.populate_help_content()

        # Footer
        footer = tk.Frame(self.main_container, bg=COLORS["panel_bg"], pady=10)
        footer.pack(fill="x", side="bottom")
        ttk.Button(footer, text="Close", command=self.close_win, style="TButton").pack()

        self._bind_mouse_scroll(self.main_container)

        # Modal/Focus
        self.update_idletasks()
        self.wait_visibility()
        self.grab_set()
        self.focus_force()

    def populate_help_content(self):
        """Populates the text content of the help window."""

        # Chapter 1: Navigation
        self.add_chapter("1. Navigation & View")
        self.add_instruction("Zoom", "Mouse Wheel", "Zooms toward the mouse pointer position.")
        self.add_instruction("Pan (Move)", "Right-Click Drag", "Click and drag to move the view area.")
        self.add_instruction("Alternative Pan", "Middle-Click / Space", "Alternative methods for panning.")

        # Chapter 2: AI Modes
        self.add_chapter("2. AI Intelligence Modes")

        self.add_sub_header("Setup & Model Installation")
        self.add_text("To enable AI features, place .onnx model files inside the 'Models' folder.")
        self.add_instruction("Source Models", "HuggingFace / GitHub", "Search for 'RMBG-1.4 ONNX', 'U2Net ONNX', or 'Segment Anything ONNX'.")

        self.add_text("Required Naming Convention (Strict):")
        self.add_instruction("Auto Models", "*.onnx", "Must contain: rmbg, isnet, u2net, or BiRefNet (e.g. rmbg.onnx).")
        self.add_instruction("SAM Models", "*.encoder.onnx AND *.decoder.onnx", "Requires BOTH encoder/decoder files for one model.")
        self.add_note("Tip: 'mobile_sam' is fast; 'sam_vit_h' is precise but slow. Restart app after adding models.")

        ttk.Separator(self.inner_frame, orient="horizontal").pack(fill="x", pady=(10, 5))

        self.add_sub_header("A. Whole Image Mode (Auto)")
        self.add_text("Select model from 'Whole Img' dropdown, click 'Auto-Detect Subject'. Runs on the entire image without interactive input.")

        self.add_sub_header("B. Manual Detection (SAM)")
        self.add_text("Click 'Manual Detection' to activate SAM mode. Status changes to 'Ready'.")
        self.add_instruction("Positive Point", "Left Click (Green)", "Area to keep/include in the mask.")
        self.add_instruction("Negative Point", "Right Click (Red)", "Area to exclude/remove from the mask (SAM active only).")
        self.add_instruction("Box Selection", "Left Drag", "Draws an initial box around the object for segmentation.")
        self.add_note("Note: Every point or box instantly updates the AI mask preview.")

        # Chapter 3: Files
        self.add_chapter("3. File Management & Gallery")
        self.add_sub_header("Importing Images")
        self.add_instruction("Import Files", "Button", "Opens a dialog to select multiple images to add to the gallery.")
        self.add_instruction("Import Folder", "Button", "Imports all valid image files from a selected folder.")
        self.add_instruction("Drag & Drop", "Mouse Action", "Drag files or folders directly onto the window or gallery area.")

        self.add_sub_header("Using the Gallery")
        self.add_text("The gallery strip stores thumbnails for quick access.")
        self.add_instruction("Select Image", "Left Click", "Highlights the thumbnail with a blue border. Hover shows the filename.")
        self.add_instruction("Load to Editor", "Use Button", "Loads the currently selected image to the main workspace.")
        self.add_instruction("Remove Image", "Delete Button", "Removes the selected image from the gallery list.")
        self.add_instruction("Clear Gallery", "Clean Button", "Removes ALL images from the gallery list.")

        # Chapter 4: Tools
        self.add_chapter("4. Manual Tools & Refinement")
        self.add_instruction("Paint Mode", "Press 'P' / Button", "Toggles manual paintbrush. Left Click to draw lines. Slider changes brush size.")
        self.add_instruction("Add to Mask", "Press 'A' / Button", "Permanently adds the current AI/Paint preview to your working result.")
        self.add_instruction("Subtract Mask", "Press 'S' / Button", "Removes the current preview from your result.")
        self.add_instruction("Undo/Redo", "Ctrl+Z / Ctrl+Y", "Steps back/forward through the mask history.")

        # Chapter 5: Export
        self.add_chapter("5. Composition & Export")
        self.add_instruction("Background", "Trans / Color / Blur", "Transparent (PNGs only), Custom Color, or Blurred original background (Inpainted).")
        self.add_instruction("Drop Shadow", "Toggle Shadow", "Adds a customizable shadow effect (opacity, blur, offset).")
        self.add_instruction("Export Format", "PNG / JPG / WEBP", "Warning: JPG does not support transparency (will result in white background).")
        self.add_instruction("Quick Export", "Alt+Q", "Saves the current result immediately to the configured output folder.")

        # Chapter 6: Cheat Sheet
        self.add_chapter("6. Keyboard Shortcuts Cheat Sheet")
        self.add_shortcut_grid([
            ("A", "Add Mask"), ("S", "Subtract Mask"),
            ("D", "Copy Source (Reset)"), ("W", "Clear Output"),
            ("R", "Reset All"), ("C", "Clear SAM Points"),
            ("P", "Toggle Paint"), ("V", "Erase Visible Area"),
            ("Ctrl+S", "Save As..."),
            ("Alt+Q", "Quick Export")
        ])

        # Spacer at bottom
        tk.Label(self.inner_frame, text="", bg=COLORS["bg"], height=2).pack()

    # --- GUI Construction Helpers ---
    def add_chapter(self, text):
        f = tk.Frame(self.inner_frame, bg=COLORS["bg"], pady=10)
        f.pack(fill="x", pady=(10, 0))
        tk.Label(f, text=text, font=("Segoe UI", 14, "bold"), bg=COLORS["bg"], fg=COLORS["accent"]).pack(anchor="w")
        ttk.Separator(self.inner_frame, orient="horizontal").pack(fill="x", pady=(0, 5))

    def add_sub_header(self, text):
        tk.Label(self.inner_frame, text=text, font=("Segoe UI", 11, "bold"), bg=COLORS["bg"], fg="#e0e0e0").pack(anchor="w", pady=(8, 2))

    def add_text(self, text):
        tk.Label(self.inner_frame, text=text, font=("Segoe UI", 10), bg=COLORS["bg"], fg="#999999", justify="left", wraplength=self.text_wrap_length).pack(anchor="w", pady=(0, 5))

    def add_note(self, text):
        tk.Label(self.inner_frame, text=text, font=("Segoe UI", 9, "italic"), bg=COLORS["bg"], fg=COLORS["undo"]).pack(anchor="w", pady=(0, 5))

    def add_instruction(self, action_name, input_trigger, description):
        row = tk.Frame(self.inner_frame, bg=COLORS["bg"], pady=4)
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)

        left_container = tk.Frame(row, bg=COLORS["bg"])
        left_container.grid(row=0, column=0, sticky="nw", padx=(0, 20))

        tk.Label(left_container, text=action_name, font=("Segoe UI", 10, "bold"), bg=COLORS["bg"], fg="white").pack(anchor="w")

        trig_frame = tk.Frame(left_container, bg=COLORS["card_bg"], padx=4, pady=1)
        trig_frame.pack(anchor="w", pady=(2, 0))
        tk.Label(trig_frame, text=input_trigger, font=("Consolas", 8), bg=COLORS["card_bg"], fg=COLORS["accent"]).pack()

        desc_wrap = self.text_wrap_length - 150
        tk.Label(row, text=description, font=("Segoe UI", 10), bg=COLORS["bg"], fg="#aaaaaa", justify="left", wraplength=desc_wrap).grid(row=0, column=1, sticky="w")

    def add_shortcut_grid(self, shortcuts):
        grid_frame = tk.Frame(self.inner_frame, bg=COLORS["bg"], pady=5)
        grid_frame.pack(fill="x")
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        row_idx = 0
        col_idx = 0
        for key, desc in shortcuts:
            item = tk.Frame(grid_frame, bg=COLORS["bg"], pady=2)
            item.grid(row=row_idx, column=col_idx, sticky="w", padx=(0, 20))

            k_f = tk.Frame(item, bg=COLORS["card_bg"], highlightbackground=COLORS["border"], highlightthickness=1, padx=5, pady=2)
            k_f.pack(side="left")
            tk.Label(k_f, text=key, font=("Consolas", 9, "bold"), bg=COLORS["card_bg"], fg=COLORS["accent"]).pack()

            tk.Label(item, text=desc, font=("Segoe UI", 9), bg=COLORS["bg"], fg="#cccccc").pack(side="left", padx=8)
            col_idx += 1
            if col_idx > 1:
                col_idx = 0
                row_idx += 1

    # --- Window Moving Logic ---
    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")

    def close_win(self):
        self.grab_release()
        self.destroy()

    # --- Canvas Scroll Logic ---
    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mouse_scroll(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_mouse_scroll(child)

    def _on_mousewheel(self, event):
        if self.canvas.yview() == (0.0, 1.0): return  # Stop if at end

        scroll_units = 0
        if platform.system() == "Windows":
            scroll_units = int(-1 * (event.delta / 120))
        elif platform.system() == "Darwin":
            scroll_units = int(-1 * event.delta)
        else:
            if event.num == 4:
                scroll_units = -1
            elif event.num == 5:
                scroll_units = 1

        if scroll_units != 0:
            self.canvas.yview_scroll(scroll_units, "units")
        return "break"


class ModernColorPicker(tk.Toplevel):
    """Custom HSV/RGB color picker. Avoids ugly, non-themed OS-native picker."""

    def __init__(self, parent, initial_color, on_update, on_close, geometry=None, scale_factor=1.0):
        super().__init__(parent)
        self.parent = parent
        self.on_update = on_update
        self.on_close = on_close
        self.scale = scale_factor

        # Flash animation state for outside click feedback
        self.style = ttk.Style()
        self.style.configure("PickerFlash.TButton",
                             background="#FFD700",
                             foreground="black",
                             padding=6,
                             relief="flat",
                             borderwidth=0)

        self.last_flash_time = 0
        self.flash_counter = 0
        self.is_flashing = False

        self.title("Color Picker")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        self.after(10, lambda: set_window_dark_mode(self))

        # Scaled dimensions (DPI aware)
        self.sv_size = int(220 * self.scale)
        self.hue_width = int(26 * self.scale)
        self.pad = int(10 * self.scale)
        self.swatch_size = int(26 * self.scale)
        self.swatch_pad = int(3 * self.scale)

        # Predefined colors for quick selection
        self.presets = [
            "#FFFFFF", "#C0C0C0", "#808080", "#000000",
            "#FF0000", "#FFFF00", "#00FF00", "#00FFFF",
            "#0000FF", "#FF00FF", "#800000", "#800000",
            "#008000", "#800080", "#008080", "#000080",
            COLORS["accent"], COLORS["add"], COLORS["remove"], COLORS["undo"],
            "#F0F8FF", "#ADD8E6", "#FAEBD7", "#9ACD32",
            "#CD7F32", "#B8860B", "#2F4F4F", "#191970",
            "#8B4513", "#D2B48C", "#FFC0CB", "#BA55D3",
            "#008B8B", "#7FFF00", "#FF4500", "#A0522D"
        ]

        # Init state
        self.current_rgb = self.hex_to_rgb(initial_color)
        self.current_hsv = colorsys.rgb_to_hsv(self.current_rgb[0] / 255, self.current_rgb[1] / 255, self.current_rgb[2] / 255)

        self.setup_ui()
        self.update_visuals_from_hsv()

        # Window positioning/centering logic
        self.update_idletasks()
        if geometry:
            try:
                parts = geometry.replace("x", "+").split("+")
                if len(parts) >= 3:
                    x, y = parts[-2], parts[-1]
                    self.geometry(f"+{x}+{y}")
                else:
                    self.center_window()
            except:
                self.center_window()
        else:
            self.center_window()

        self.protocol("WM_DELETE_WINDOW", self.close_picker)

        # Modal/Focus
        self.transient(parent)
        self.grab_set()
        self.focus_set()

        # Flash on outside click
        self.bind("<Button-1>", self.check_outside_click)

    def check_outside_click(self, event):
        win_w = self.winfo_width()
        win_h = self.winfo_height()
        # Check if the click coordinates are outside the window's bounding box
        is_outside = (event.x < 0 or event.x > win_w or event.y < 0 or event.y > win_h)

        if is_outside:
            self.trigger_close_flash()

    def trigger_close_flash(self):
        # 4 second cooldown for flashing
        now = timer()
        if now - self.last_flash_time < 4.0:
            return

        if not self.is_flashing:
            self.last_flash_time = now
            self.flash_counter = 0
            self.is_flashing = True
            self.animate_flash()

    def animate_flash(self):
        # Flash close button 3 times
        if self.flash_counter >= 6:
            self.close_btn.configure(style="TButton")
            self.is_flashing = False
            return

        is_highlight = (self.flash_counter % 2 == 0)

        if is_highlight:
            self.close_btn.configure(style="PickerFlash.TButton")
        else:
            self.close_btn.configure(style="TButton")

        self.flash_counter += 1
        self.after(150, self.animate_flash)

    def center_window(self):
        """Calculates center relative to the parent window."""
        req_w = self.winfo_reqwidth()
        req_h = self.winfo_reqheight()
        root_x = self.parent.winfo_rootx()
        root_y = self.parent.winfo_rooty()
        root_w = self.parent.winfo_width()
        root_h = self.parent.winfo_height()
        pos_x = root_x + (root_w // 2) - (req_w // 2)
        pos_y = root_y + (root_h // 2) - (req_h // 2)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        safe_bottom = screen_h - 80
        pos_x = max(0, min(pos_x, screen_w - req_w))
        pos_y = max(0, min(pos_y, safe_bottom - req_h))
        self.geometry(f"+{pos_x}+{pos_y}")

    def setup_ui(self):
        container = tk.Frame(self, bg=COLORS["bg"], padx=self.pad, pady=self.pad)
        container.pack(fill="both", expand=True)

        # === Left Column: The Picker ===
        left_col = tk.Frame(container, bg=COLORS["bg"])
        left_col.pack(side="left", fill="y", padx=(0, self.pad))

        # Bottom Buttons
        btn_row = tk.Frame(left_col, bg=COLORS["bg"])
        btn_row.pack(side="bottom", fill="x", pady=(int(10 * self.scale), 0))

        ttk.Button(btn_row, text="Apply", command=self.apply_color, style="Accent.TButton").pack(side="right", padx=(int(5 * self.scale), 0), expand=True, fill="x")

        self.close_btn = ttk.Button(btn_row, text="Close", command=self.close_picker, style="TButton")
        self.close_btn.pack(side="right", expand=True, fill="x")

        # Hex Entry and Preview Swatch
        ctrl_row = tk.Frame(left_col, bg=COLORS["bg"], pady=self.pad)
        ctrl_row.pack(side="bottom", fill="x")

        preview_size = int(32 * self.scale)
        self.preview_frame = tk.Frame(ctrl_row, width=preview_size, height=preview_size, bg=self.rgb_to_hex(self.current_rgb),
                                      relief="flat", highlightthickness=1, highlightbackground=COLORS["border"])
        self.preview_frame.pack(side="left", padx=(0, self.pad))

        entry_container = tk.Frame(ctrl_row, bg=COLORS["card_bg"], highlightthickness=1, highlightbackground=COLORS["border"])
        entry_container.pack(side="left", fill="y")

        self.hex_var = tk.StringVar(value=self.rgb_to_hex(self.current_rgb))
        font_size = int(10 * self.scale)
        self.hex_entry = tk.Entry(entry_container, textvariable=self.hex_var, width=10,
                                  bg=COLORS["card_bg"], fg="white", borderwidth=0, font=("Segoe UI", font_size))
        self.hex_entry.pack(side="left", padx=int(5 * self.scale), pady=int(5 * self.scale))

        # Bind hex input validation
        self.hex_entry.bind("<Return>", self.on_hex_enter)
        self.hex_entry.bind("<FocusOut>", self.on_hex_enter)
        self.hex_entry.bind("<<Paste>>", self.on_paste)

        # Gradient Canvas (Saturation/Value and Hue)
        picker_row = tk.Frame(left_col, bg=COLORS["bg"])
        picker_row.pack(side="top", fill="both", expand=True)

        self.sv_canvas = tk.Canvas(picker_row, width=self.sv_size, height=self.sv_size,
                                   bg="white", highlightthickness=1, highlightbackground=COLORS["border"])
        self.sv_canvas.pack(side="left", padx=(0, self.pad))
        self.sv_canvas.bind("<Button-1>", self.on_sv_click)
        self.sv_canvas.bind("<B1-Motion>", self.on_sv_click)

        self.hue_canvas = tk.Canvas(picker_row, width=self.hue_width, height=self.sv_size,
                                    bg="white", highlightthickness=1, highlightbackground=COLORS["border"])
        self.hue_canvas.pack(side="left")
        self.hue_canvas.bind("<Button-1>", self.on_hue_click)
        self.hue_canvas.bind("<B1-Motion>", self.on_hue_click)
        self.draw_hue_gradient()

        # === Right Column: Presets ===
        ttk.Separator(container, orient="vertical").pack(side="left", fill="y", padx=int(5 * self.scale))
        right_col = tk.Frame(container, bg=COLORS["bg"])
        right_col.pack(side="left", fill="both", expand=True, padx=(self.pad, 0))

        tk.Label(right_col, text="PRESETS", bg=COLORS["bg"], fg=COLORS["header"],
                 font=("Segoe UI", int(8 * self.scale), "bold"), anchor="w").pack(fill="x", pady=(0, int(8 * self.scale)))

        grid_frame = tk.Frame(right_col, bg=COLORS["bg"])
        grid_frame.pack(fill="both", expand=True)

        cols = 4
        for i, color_hex in enumerate(self.presets):
            r = i // cols
            c = i % cols
            f = tk.Frame(grid_frame, bg=color_hex, width=self.swatch_size, height=self.swatch_size,
                         cursor="hand2", highlightthickness=1, highlightbackground=COLORS["border"])
            f.grid(row=r, column=c, padx=self.swatch_pad, pady=self.swatch_pad)
            f.bind("<Button-1>", lambda e, h=color_hex: self.load_preset(h))
            # Hover effect
            f.bind("<Enter>", lambda e, w=f: w.config(highlightbackground="white"))
            f.bind("<Leave>", lambda e, w=f: w.config(highlightbackground=COLORS["border"]))

    def on_paste(self, event):
        # Delay hex validation to allow paste buffer update
        self.after(50, lambda: self.on_hex_enter(None))

    def load_preset(self, hex_code):
        # Load color preset from hex
        self.current_rgb = self.hex_to_rgb(hex_code)
        self.current_hsv = colorsys.rgb_to_hsv(self.current_rgb[0] / 255, self.current_rgb[1] / 255, self.current_rgb[2] / 255)
        self.update_visuals_from_hsv()
        self.focus_set()

    def draw_hue_gradient(self):
        # Renders the vertical hue strip
        for y in range(self.sv_size):
            hue = y / self.sv_size
            r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
            color = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            self.hue_canvas.create_line(0, y, self.hue_width, y, fill=color)

    def redraw_sv_gradient(self):
        """Renders the Saturation/Value square based on current Hue (using PIL/ImageOps for speed)."""
        hue = self.current_hsv[0]
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        base_color = (int(r * 255), int(g * 255), int(b * 255))

        # Create color gradient (Hue + Saturation mask)
        img = Image.new("RGB", (64, 64), base_color)
        white = Image.new("RGB", (64, 64), (255, 255, 255))
        mask_s = Image.linear_gradient("L").rotate(90).resize((64, 64))
        img = Image.composite(img, white, mask_s)

        # Apply Value mask (Black fade)
        black = Image.new("RGB", (64, 64), (0, 0, 0))
        mask_v = Image.linear_gradient("L").resize((64, 64))
        img = Image.composite(black, img, mask_v)

        self.sv_image = img.resize((self.sv_size, self.sv_size), Image.Resampling.BICUBIC)
        self.tk_sv_image = ImageTk.PhotoImage(self.sv_image)
        self.sv_canvas.create_image(0, 0, anchor="nw", image=self.tk_sv_image)
        self.draw_sv_cursor()

    def draw_sv_cursor(self):
        # Draws the cursor on the S/V canvas
        self.sv_canvas.delete("cursor")
        s, v = self.current_hsv[1], self.current_hsv[2]
        x = s * self.sv_size
        y = (1 - v) * self.sv_size
        r = int(6 * self.scale)
        # Double circle for max visibility
        self.sv_canvas.create_oval(x - r, y - r, x + r, y + r, outline="white", width=2, tags="cursor")
        self.sv_canvas.create_oval(x - r + 1, y - r + 1, x + r - 1, y + r - 1, outline="black", width=1, tags="cursor")

    def draw_hue_cursor(self):
        # Draws the cursor on the Hue strip
        self.hue_canvas.delete("cursor")
        y = self.current_hsv[0] * self.sv_size
        self.hue_canvas.create_line(0, y, self.hue_width, y, fill="black", width=3, tags="cursor")
        self.hue_canvas.create_line(0, y, self.hue_width, y, fill="white", width=1, tags="cursor")

    def update_visuals_from_hsv(self):
        # Recalculate visuals from HSV state
        self.redraw_sv_gradient()
        self.draw_hue_cursor()
        r, g, b = colorsys.hsv_to_rgb(self.current_hsv[0], self.current_hsv[1], self.current_hsv[2])
        self.current_rgb = (int(r * 255), int(g * 255), int(b * 255))
        hex_code = self.rgb_to_hex(self.current_rgb)
        self.preview_frame.config(bg=hex_code)

        # Update hex entry if not currently typing there
        if self.focus_get() != self.hex_entry:
            self.hex_var.set(hex_code)

    def on_hue_click(self, event):
        # Updates hue based on Y-coordinate click
        y = max(0, min(event.y, self.sv_size))
        hue = y / self.sv_size
        s, v = self.current_hsv[1], self.current_hsv[2]
        self.current_hsv = (hue, s, v)
        self.update_visuals_from_hsv()
        self.focus_set()

    def on_sv_click(self, event):
        # Updates saturation and value based on X/Y-coordinate click
        x = max(0, min(event.x, self.sv_size))
        y = max(0, min(event.y, self.sv_size))
        s = x / self.sv_size
        v = 1 - (y / self.sv_size)
        hue = self.current_hsv[0]
        self.current_hsv = (hue, s, v)
        self.update_visuals_from_hsv()
        self.focus_set()

    def on_hex_enter(self, event=None):
        # Validates and applies hex code input
        h = self.hex_var.get().strip()
        if not h.startswith("#"):
            h = "#" + h

        if len(h) == 7:
            try:
                rgb = self.hex_to_rgb(h)
                self.current_rgb = rgb
                self.current_hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
                self.hex_var.set(h)
                self.update_visuals_from_hsv()
                self.focus_set()
            except:
                pass  # Invalid hex code

    def apply_color(self):
        # Executes the callback with the final selected color
        self.on_update(self.rgb_to_hex(self.current_rgb))

    def close_picker(self):
        # Executes close callback and destroys window
        self.on_close(self.geometry())
        self.destroy()

    @staticmethod
    def hex_to_rgb(hex_val):
        h = hex_val.lstrip('#')
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb


class LoadingOverlay(tk.Frame):
    """'Glass' overlay to block input during heavy AI processing. Faux-transparency hack via ImageGrab."""

    def __init__(self, master, text="Processing..."):
        super().__init__(master)
        self.running = True

        # Input blocking
        self.bind("<Button-1>", lambda e: "break")
        self.bind("<Button-2>", lambda e: "break")
        self.bind("<Button-3>", lambda e: "break")
        self.bind("<MouseWheel>", lambda e: "break")

        # Faux-Transparency Hack: grab screenshot, darken, and blur.
        self.update_idletasks()

        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        h = master.winfo_height()
        self.bg_photo = None

        try:
            shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            enhancer = ImageEnhance.Brightness(shot)
            darkened = enhancer.enhance(0.4)  # Dim to 40%
            blurred = darkened.filter(ImageFilter.GaussianBlur(radius=3))
            self.bg_photo = ImageTk.PhotoImage(blurred)
        except Exception:
            pass  # Fallback if screenshot fails

        # Main Canvas
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#1e1e1e")
        self.canvas.pack(fill="both", expand=True)

        if self.bg_photo:
            self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        else:
            self.canvas.configure(bg="#151515")  # Plain dark fallback

        # Central Loading Bar UI
        self.text = text
        self.bar_w = 300
        self.bar_h = 6

        # Draw UI elements (coords set in _on_resize)
        self.band_bg = self.canvas.create_rectangle(0, 0, 0, 0, fill="#252526", outline="")
        self.line_top = self.canvas.create_line(0, 0, 0, 0, fill=COLORS["accent"], width=2)
        self.line_bot = self.canvas.create_line(0, 0, 0, 0, fill=COLORS["accent"], width=2)
        self.text_id = self.canvas.create_text(0, 0, text=text, fill="white", font=("Segoe UI", 14, "bold"))
        self.track_id = self.canvas.create_line(0, 0, 0, 0, fill="#3e3e42", width=self.bar_h, capstyle=tk.ROUND)
        self.thumb_id = self.canvas.create_line(0, 0, 0, 0, fill=COLORS["accent"], width=self.bar_h, capstyle=tk.ROUND)

        self.anim_step = 0
        self.bind("<Configure>", self._on_resize)

        self.lift()
        self._animate_bar()

    def _on_resize(self, event):
        """Re-centers the loading bar elements on window resize."""
        w = event.width
        h = event.height
        cy = h // 2
        bh = 120  # Dark band height

        self.canvas.coords(self.band_bg, 0, cy - bh // 2, w, cy + bh // 2)
        self.canvas.coords(self.line_top, 0, cy - bh // 2, w, cy - bh // 2)
        self.canvas.coords(self.line_bot, 0, cy + bh // 2, w, cy + bh // 2)
        self.canvas.coords(self.text_id, w // 2, cy - 15)

        self.bar_x = (w - self.bar_w) // 2
        self.bar_y = cy + 25
        self.canvas.coords(self.track_id, self.bar_x, self.bar_y, self.bar_x + self.bar_w, self.bar_y)

    def _animate_bar(self):
        """Bouncing animation logic (like a marquee)."""
        if not self.running or not self.winfo_exists():
            return

        self.anim_step += 0.08
        norm_pos = (math.sin(self.anim_step) + 1) / 2  # Sine wave for bounce

        thumb_len = 80
        travel_area = self.bar_w - thumb_len

        if hasattr(self, 'bar_x'):
            start_x = self.bar_x + (travel_area * norm_pos)
            end_x = start_x + thumb_len
            self.canvas.coords(self.thumb_id, start_x, self.bar_y, end_x, self.bar_y)

        self.after(16, self._animate_bar)

    def set_text(self, text):
        self.canvas.itemconfig(self.text_id, text=text)

    def destroy(self):
        self.running = False
        super().destroy()


# --- CUSTOM WIDGETS ---

class SleekSlider(tk.Canvas):
    """Custom-drawn canvas slider. Much better look/feel than default ttk.Scale."""

    def __init__(self, master, width=200, height=30, min_val=0, max_val=100, init_val=50, command=None, bg_color="#1e1e1e", accent_color="#007acc"):
        super().__init__(master, width=width, height=height, bg=bg_color, highlightthickness=0)
        self.min_val = min_val
        self.max_val = max_val
        self.value = init_val
        self.command = command
        self.accent = accent_color
        self.track_color = "#3e3e42"
        self.width = width
        self.height = height
        self.pad_x = 10
        self.track_y = height // 2
        self.track_width = width - (self.pad_x * 2)

        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.draw()

    def _on_resize(self, event):
        self.width = event.width
        self.track_width = self.width - (self.pad_x * 2)
        self.draw()

    def val_to_pixel(self, val):
        if self.max_val == self.min_val: return self.pad_x
        percent = (val - self.min_val) / (self.max_val - self.min_val)
        return self.pad_x + (percent * self.track_width)

    def pixel_to_val(self, x):
        # Clamps the pixel position and converts to a value
        x = max(self.pad_x, min(x, self.width - self.pad_x))
        percent = (x - self.pad_x) / self.track_width
        return self.min_val + (percent * (self.max_val - self.min_val))

    def set_value(self, val):
        self.value = max(self.min_val, min(val, self.max_val))
        self.draw()
        if self.command:
            self.command(self.value)

    def get(self):
        return self.value

    def _on_click(self, event):
        self.set_value(self.pixel_to_val(event.x))

    def _on_drag(self, event):
        self.set_value(self.pixel_to_val(event.x))

    def draw(self):
        self.delete("all")
        x_start = self.pad_x
        x_end = self.width - self.pad_x
        x_curr = self.val_to_pixel(self.value)

        # Track base
        self.create_line(x_start, self.track_y, x_end, self.track_y, fill=self.track_color, width=4, capstyle=tk.ROUND)
        # Active part
        if x_curr > x_start:
            self.create_line(x_start, self.track_y, x_curr, self.track_y, fill=self.accent, width=4, capstyle=tk.ROUND)
        # Thumb
        r = 7
        self.create_oval(x_curr - r, self.track_y - r, x_curr + r, self.track_y + r, fill="#ffffff", outline=self.track_color, width=1)


class ModernComboGroup(tk.Frame):
    """Custom dropdown menu using Toplevel/Canvas. Avoids unstylable ttk.Combobox on Windows."""

    def __init__(self, parent, label_text, values=None):
        super().__init__(parent, bg=COLORS["card_bg"], highlightthickness=1, highlightbackground=COLORS["border"])
        self.parent = parent
        self.values = values or []
        self.is_open = False
        self.dropdown_window = None
        self.last_close_time = 0
        self.scroll_bind_ids = []

        self.columnconfigure(1, weight=1)

        # Label on the left
        self.lbl_frame = tk.Frame(self, bg=COLORS["panel_bg"], width=90)
        self.lbl_frame.grid(row=0, column=0, sticky="ns")

        self.lbl = tk.Label(self.lbl_frame, text=label_text,
                            bg=COLORS["panel_bg"], fg=COLORS["accent"],
                            font=("Segoe UI", 9, "bold"), anchor="e", padx=8)
        self.lbl.pack(fill="both", expand=True, pady=3)

        # Vertical Separator
        sep = tk.Frame(self, bg=COLORS["border"], width=1)
        sep.grid(row=0, column=0, sticky="nse")

        # Clickable area/display
        self.click_frame = tk.Frame(self, bg=COLORS["card_bg"], cursor="hand2")
        self.click_frame.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        self.value_var = tk.StringVar()
        self.display_lbl = tk.Label(self.click_frame, textvariable=self.value_var,
                                    bg=COLORS["card_bg"], fg=COLORS["fg"], anchor="w", font=("Segoe UI", 9))
        self.display_lbl.pack(side="left", fill="x", expand=True, padx=5)

        # Dropdown arrow
        self.arrow_canvas = tk.Canvas(self.click_frame, width=16, height=16, bg=COLORS["card_bg"], highlightthickness=0)
        self.arrow_canvas.pack(side="right", padx=5)
        self.draw_arrow("down")

        # Bindings
        for w in [self.click_frame, self.display_lbl, self.arrow_canvas, self.lbl]:
            w.bind("<Button-1>", self.toggle_dropdown)
            w.bind("<Enter>", self.on_hover)
            w.bind("<Leave>", self.on_leave)

    def draw_arrow(self, direction):
        self.arrow_canvas.delete("all")
        color = COLORS["fg"]
        if direction == "down":
            self.arrow_canvas.create_polygon(4, 6, 12, 6, 8, 11, fill=color, outline="")
        else:
            self.arrow_canvas.create_polygon(4, 10, 12, 10, 8, 5, fill=color, outline="")

    def on_hover(self, e):
        self.configure(highlightbackground=COLORS["accent"])

    def on_leave(self, e):
        if not self.is_open:
            self.configure(highlightbackground=COLORS["border"])

    def toggle_dropdown(self, event=None):
        # Debounce quick double-clicks
        if (timer() - self.last_close_time) < 0.2:
            return

        if self.is_open:
            self.close_dropdown()
        else:
            self.open_dropdown()

    def open_dropdown(self):
        if not self.values: return
        self.is_open = True
        self.draw_arrow("up")
        self.configure(highlightbackground=COLORS["accent"])

        # Calculate position for the floating Toplevel window
        ROW_HEIGHT = 42
        SPACING = 5
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        w = self.winfo_width()
        total_content_height = len(self.values) * (ROW_HEIGHT + SPACING)
        window_height = min(300, total_content_height)

        # Toplevel setup
        self.dropdown_window = Toplevel(self)
        self.dropdown_window.wm_overrideredirect(True)
        self.dropdown_window.geometry(f"{w}x{window_height}+{x}+{y}")
        self.dropdown_window.configure(bg=COLORS["border"])

        # Scrollable content using Canvas/Frame trick
        canvas_container = tk.Canvas(self.dropdown_window, bg=COLORS["dropdown_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.dropdown_window, orient="vertical", command=canvas_container.yview, style="NoArrow.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas_container, bg=COLORS["dropdown_bg"])

        scrollable_frame.bind("<Configure>", lambda e: canvas_container.configure(scrollregion=canvas_container.bbox("all")))

        canvas_container.create_window((0, 0), window=scrollable_frame, anchor="nw", width=w)
        canvas_container.configure(yscrollcommand=scrollbar.set)
        canvas_container.pack(side="left", fill="both", expand=True, padx=1, pady=1)

        if total_content_height > 300:
            scrollbar.pack(side="right", fill="y", pady=1)

        # Populate list items
        for val in self.values:
            item_frame = tk.Frame(scrollable_frame, bg=COLORS["dropdown_bg"], height=ROW_HEIGHT, highlightthickness=1, highlightbackground="#444444")
            item_frame.pack_propagate(False)
            item_frame.pack(fill="x", pady=SPACING // 2, padx=4)

            lbl = tk.Label(item_frame, text=f"  {val}", anchor="w",
                           bg=COLORS["dropdown_bg"], fg=COLORS["dropdown_fg"],
                           font=("Segoe UI", 9), cursor="hand2")
            lbl.pack(fill="both", expand=True)

            lbl.bind("<Button-1>", lambda e, v=val: self.on_select_val(v))
            # Item hover effect
            lbl.bind("<Enter>", lambda e, l=lbl: l.configure(bg=COLORS["accent"], fg="white"))
            lbl.bind("<Leave>", lambda e, l=lbl: l.configure(bg=COLORS["dropdown_bg"], fg=COLORS["dropdown_fg"]))

        self.dropdown_window.focus_set()

        # Close on outside click or Escape
        self.dropdown_window.bind("<FocusOut>", self.check_focus_loss)
        self.dropdown_window.bind("<Escape>", lambda e: self.close_dropdown())

        # Forward scrolling from the main window to the dropdown
        top_lvl = self.winfo_toplevel()
        # Note: bind IDs stored for cleanup
        self.scroll_bind_ids.append(top_lvl.bind("<MouseWheel>", self.handle_global_scroll, add="+"))
        self.scroll_bind_ids.append(top_lvl.bind("<Button-4>", self.handle_global_scroll, add="+"))
        self.scroll_bind_ids.append(top_lvl.bind("<Button-5>", self.handle_global_scroll, add="+"))

    def handle_global_scroll(self, event):
        # If user scrolls but mouse is not over the dropdown, close it.
        if not self.is_open or not self.dropdown_window:
            return
        x, y = self.winfo_pointerxy()
        try:
            widget_under_mouse = self.dropdown_window.winfo_containing(x, y)
            if widget_under_mouse and str(widget_under_mouse).startswith(str(self.dropdown_window)):
                return  # Mouse is over the dropdown, don't close.
        except:
            pass
        self.close_dropdown()

    def check_focus_loss(self, event):
        # Checks if focus loss was due to a click outside the app, or click on a widget inside.
        if self.dropdown_window:
            x, y = self.winfo_pointerxy()
            widget_under_mouse = self.dropdown_window.winfo_containing(x, y)
            try:
                # If the mouse is still over a child widget, don't close.
                if widget_under_mouse and str(widget_under_mouse).startswith(str(self.dropdown_window)):
                    return
            except:
                pass
            self.close_dropdown()

    def close_dropdown(self):
        # Remove global scroll bindings
        top_lvl = self.winfo_toplevel()
        for bind_id in self.scroll_bind_ids:
            try:
                top_lvl.unbind("<MouseWheel>", bind_id)
                top_lvl.unbind("<Button-4>", bind_id)
                top_lvl.unbind("<Button-5>", bind_id)
            except:
                pass
        self.scroll_bind_ids = []

        if self.dropdown_window:
            self.dropdown_window.destroy()
            self.dropdown_window = None

        self.is_open = False
        self.last_close_time = timer()  # Reset debounce timer
        self.draw_arrow("down")
        self.configure(highlightbackground=COLORS["border"])

    def on_select_val(self, val):
        self.value_var.set(val)
        self.close_dropdown()

    def configure(self, cnf=None, **kwargs):
        if 'values' in kwargs:
            self.values = kwargs.pop('values')
        super().configure(cnf, **kwargs)

    def get(self):
        return self.value_var.get()

    def set(self, val):
        self.value_var.set(val)

    def current(self, index):
        if 0 <= index < len(self.values):
            self.value_var.set(self.values[index])


# --- MAIN APPLICATION LOGIC ---

class BackgroundRemoverGUI:
    def __init__(self, root, image_paths):
        self.root = root

        # DPI scaling for better appearance on high-res screens
        self.dpi_scale = self.root.winfo_fpixels('1i') / 96.0
        print(f"DPI Scale Factor detected: {self.dpi_scale}")

        self.image_paths = image_paths if image_paths else []
        self.current_image_index = 0
        self.cached_blurred_shadow = None
        self._last_shadow_radius = None
        self.config = self.load_config()

        # --- Gallery State ---
        self.gallery_files = []  # Stores {'path', 'name', 'thumb'}
        self.selected_gallery_index = None
        self.tooltip_after_id = None

        # --- Export State ---
        self.export_format = self.config.get("save_file_type", "png")
        self._flash_trans_step = 0  # For transparent/JPG conflict feedback

        # Title: shows HW mode and image count
        self.file_count = ""
        if len(self.image_paths) >= 1:
            self.file_count = f' - Image {self.current_image_index + 1} of {len(self.image_paths)}' if len(self.image_paths) > 1 else ' - Image 1 of 1'

        hw_mode = "GPU" if "CPU" not in ONNX_PROVIDERS[0] else "CPU"
        self.root.title(f"Background Remover Pro [{hw_mode}]" + self.file_count)

        self.setup_theme()
        self.root.configure(bg=COLORS["bg"])

        set_window_dark_mode(self.root)

        # Drag and Drop registration (handled in build_gui for specific widgets)
        if DND_AVAILABLE:
            self.root.drop_target_register(DND_FILES)

        # UI State Variables
        self.coordinates = []  # SAM positive/negative points
        self.labels = []
        self.dots = []
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.click_start_x = 0
        self.click_start_y = 0
        self.last_x, self.last_y = 0, 0
        self.pan_step = 30
        self.area_enabled = False  # Box selection mode
        self.move_enabled = True  # Pan mode (default)
        self.lines = []  # Paint mode lines
        self.lines_id = []
        self.lines_id2 = []

        # Marquee Animation State for folder path
        self.marquee_text = ""
        self.marquee_index = 0
        self.marquee_direction = 1
        self.marquee_after_id = None

        # Background Editing State
        self.bg_mode = "transparent"
        self.bg_custom_color = "#0000FF"
        self.cached_blur_image = None
        self.current_blur_radius = 20
        self.loading_overlay = None

        available = ort.get_available_providers()
        self.gpu_providers = []

        # GPU Priority Logic: CUDA -> DirectML -> Fallback
        if 'CUDAExecutionProvider' in available:
            self.gpu_providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in available:
            self.gpu_providers.append('DmlExecutionProvider')

        # Always add CPU as the final fallback for the GPU list
        self.gpu_providers.append('CPUExecutionProvider')

        # Set default (GPU if available, otherwise CPU)
        if len(self.gpu_providers) > 1:
            self.active_providers = self.gpu_providers
            self.current_hw_mode = "GPU"
        else:
            self.active_providers = ['CPUExecutionProvider']
            self.current_hw_mode = "CPU"


        # Image Loading/Placeholder
        if self.image_paths:
            self.load_image_path(self.image_paths[self.current_image_index])
        else:
            # Default empty canvas
            self.original_image = Image.new("RGBA", (800, 600), (200, 200, 200, 255))
            self.image_exif = None

        # Config Params (sync with loaded)
        self.save_file_type = self.config.get("save_file_type", "png")
        self.save_file_quality = self.config.get("save_file_quality", 90)
        self.save_mask = self.config.get("save_mask", False)

        # Zoom / View State
        self.after_id = None
        self.zoom_delay = 0.2
        self.canvas_w = 200
        self.canvas_h = 200
        self.init_width = 200
        self.init_height = 200

        self.setup_image_display()
        self.root.bind("<Configure>", self.on_resize)

        # Build UI
        self.build_gui()

        # Smart Window Positioning
        if self.config.get("window_zoomed", True):
            if platform.system() == "Windows":
                self.root.state('zoomed')
            else:
                self.root.attributes('-zoomed', True)
        else:
            # Center and resize to saved dimensions
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            target_w = self.config.get("window_width", 1200)
            target_h = self.config.get("window_height", 800)
            target_w = min(target_w, int(screen_w * 0.9))
            target_h = min(target_h, int(screen_h * 0.9))
            pos_x = (screen_w - target_w) // 2
            pos_y = (screen_h - target_h) // 2
            self.root.geometry(f"{target_w}x{target_h}+{pos_x}+{pos_y}")

        self.root.update_idletasks()

        # AI State (Models)
        self.model_output_mask = None  # Mask preview (blue overlay)
        self.raw_model_mask = None  # Raw mask output (from whole image model)
        self.raw_sam_logits = None  # Raw logits (from SAM)
        self.sam_active = False
        self.last_flash_time = 0

        self.update_input_image_preview()
        self.set_keybindings()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Magic Button Animation State
        self._magic_hovering = False
        self._magic_cycle_id = None
        self._magic_colors = ["#00E5FF", "#69F0AE", "#FFFF00", "#FFAB40", "#FF4081", "#EA80FC"]
        self._magic_idx = 0
        self._magic_step = 0
        self._magic_steps_total = 25

    # --- THREADING & OVERLAY HELPERS ---

    def show_loading(self, message="Processing AI..."):
        """Shows the glass overlay to block UI interaction during heavy tasks."""
        self.loading_overlay = LoadingOverlay(self.root, text=message)
        self.loading_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.root.update()

    def hide_loading(self):
        """Destroys the overlay."""
        if hasattr(self, 'loading_overlay') and self.loading_overlay:
            self.loading_overlay.destroy()
            self.loading_overlay = None

    def start_threaded_task(self, target_func, callback_func, error_callback=None):
        """Generic thread runner. Executes task and uses polling for result in main thread."""
        self.result_queue = queue.Queue()

        def worker():
            try:
                result = target_func()
                self.result_queue.put(("success", result))
            except Exception as e:
                self.result_queue.put(("error", e))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Start polling in the main loop
        self.root.after(100, lambda: self._monitor_thread(callback_func, error_callback))

    def _monitor_thread(self, callback_func, error_callback):
        """Checks the result queue every 100ms."""
        try:
            status, payload = self.result_queue.get_nowait()

            # Thread finished
            self.hide_loading()

            if status == "success":
                callback_func(payload)
            else:
                if error_callback:
                    error_callback(payload)
                else:
                    messagebox.showerror("AI Error", f"An error occurred:\n{payload}")
                    print(f"Thread Error: {payload}")

        except queue.Empty:
            # Not ready, keep polling
            self.root.after(100, lambda: self._monitor_thread(callback_func, error_callback))

    # --- HEADLESS / THREAD SAFE METHODS (No GUI access here!) ---

    def set_hardware_mode(self, mode):
        """Changes hardware mode and clears the session to force a reload."""
        if mode == self.current_hw_mode:
            return  # No change

        self.current_hw_mode = mode

        if mode == "CPU":
            self.active_providers = ['CPUExecutionProvider']
            print("Switched to CPU Mode")
        else:
            # GPU Mode: Use the list calculated in init (CUDA or DML)
            self.active_providers = self.gpu_providers
            print(f"Switched to GPU Mode (Providers: {self.active_providers})")

        # Update UI
        self.update_hardware_buttons_visual()

        # Update window title
        title_parts = self.root.title().split("[")
        base_title = title_parts[0].strip()
        self.root.title(f"{base_title} [{mode}]" + self.file_count)

        # Reset loaded models to force use of the new provider on next execution
        self.unload_all_models()
        self.status_label.config(text=f"Hardware switched to {mode}. Models unloaded.", fg="white")

    def update_hardware_buttons_visual(self):
        """Manages the requested Toggle appearance with dark gray color."""
        active_bg = COLORS["accent"]  # Blue (or current accent color)
        active_fg = "white"

        inactive_bg = "#2D2D30"  # THE REQUESTED COLOR
        inactive_fg = COLORS["fg"]  # Standard text color (light gray)

        if self.current_hw_mode == "CPU":
            # CPU Active
            self.btn_run_cpu.config(bg=active_bg, fg=active_fg)
            self.btn_run_gpu.config(bg=inactive_bg, fg=inactive_fg)
        else:
            # GPU Active
            self.btn_run_cpu.config(bg=inactive_bg, fg=inactive_fg)
            self.btn_run_gpu.config(bg=active_bg, fg=active_fg)

    def unload_all_models(self):
        """Removes ONNX sessions from memory to allow provider change."""
        # Remove Whole Image sessions
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            if attr.endswith("_session") or attr in ["sam_encoder", "sam_decoder"]:
                delattr(self, attr)

        # Reset SAM embeddings if present
        if hasattr(self, "encoder_output"):
            delattr(self, "encoder_output")

        # Reset button state
        self.load_models_btn.configure(state="normal", text='Pre Load Models')
        self.sam_active = False

    def thread_safe_load_model(self, model_name):
        """Loads or retrieves a cached ONNX session using CURRENT HW providers."""
        if not hasattr(self, f"{model_name}_session"):
            path = f'{MODEL_ROOT}{model_name}.onnx'
            sess_opts = get_ort_session_options()
            sess = ort.InferenceSession(path, sess_opts, providers=self.active_providers)
            setattr(self, f"{model_name}_session", sess)
        return getattr(self, f"{model_name}_session")

    def _initialise_sam_model_headless(self):
        """Loads SAM Encoder/Decoder using CURRENT HW providers."""
        if not hasattr(self, "sam_encoder") or self.sam_model != MODEL_ROOT + self.sam_combo.get():
            if self.sam_combo.get() == "No Models Found":
                raise Exception("No SAM models found.")

            self.sam_model = MODEL_ROOT + self.sam_combo.get()
            sess_opts = get_ort_session_options()

            # MODIFICATION HERE: Use self.active_providers
            self.sam_encoder = ort.InferenceSession(self.sam_model + ".encoder.onnx", sess_opts, providers=self.active_providers)
            self.sam_decoder = ort.InferenceSession(self.sam_model + ".decoder.onnx", sess_opts, providers=self.active_providers)

            if hasattr(self, "encoder_output"): delattr(self, "encoder_output")

    def calculate_sam_embedding_headless(self):
        """Calculates the static image embedding (the heaviest part of SAM)."""
        target_size = 1024
        input_size = (684, 1024)  # Internal fixed size for SAM
        encoder_input_name = self.sam_encoder.get_inputs()[0].name
        img = self.original_image.convert("RGB")
        cv_image = np.array(img)

        # SAM's pre-processing transforms the image to a fixed size
        scale_x = input_size[1] / cv_image.shape[1]
        scale_y = input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        cv_image = cv2.warpAffine(cv_image, transform_matrix[:2], (input_size[1], input_size[0]), flags=cv2.INTER_LINEAR)

        encoder_inputs = {encoder_input_name: cv_image.astype(np.float32)}
        self.encoder_output = self.sam_encoder.run(None, encoder_inputs)

    # -----------------------------------------------

    def setup_theme(self):
        """Configures all custom ttk styles."""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # File Management Buttons (specific hover effects)
        FILE_BTN_BG = "#2D2D30"

        self.style.configure("Use.File.TButton", background=FILE_BTN_BG, foreground="white", borderwidth=0, focuscolor=FILE_BTN_BG)
        self.style.map("Use.File.TButton", background=[('active', FILE_BTN_BG), ('pressed', FILE_BTN_BG)], foreground=[('active', COLORS["add"]), ('pressed', COLORS["add"])])

        self.style.configure("Delete.File.TButton", background=FILE_BTN_BG, foreground="white", borderwidth=0, focuscolor=FILE_BTN_BG)
        self.style.map("Delete.File.TButton", background=[('active', FILE_BTN_BG), ('pressed', FILE_BTN_BG)], foreground=[('active', "#D29922"), ('pressed', "#D29922")])

        self.style.configure("Clean.File.TButton", background=FILE_BTN_BG, foreground="white", borderwidth=0, focuscolor=FILE_BTN_BG)
        self.style.map("Clean.File.TButton", background=[('active', FILE_BTN_BG), ('pressed', FILE_BTN_BG)], foreground=[('active', COLORS["remove"]), ('pressed', COLORS["remove"])])

        # Copy In->Out button
        COPY_TEXT_HOVER = "#112A46"
        COPY_BG_HOVER = "#ACC8E5"
        self.style.configure("Copy.TButton", padding=6, relief="flat", background=COLORS["card_bg"], foreground=COLORS["fg"], borderwidth=0, focusthickness=0, focuscolor=COLORS["card_bg"])
        self.style.map("Copy.TButton", background=[('active', COPY_BG_HOVER), ('pressed', COPY_BG_HOVER)], foreground=[('active', COPY_TEXT_HOVER), ('pressed', COPY_TEXT_HOVER)],
                       focuscolor=[('active', COPY_BG_HOVER), ('!active', COLORS["card_bg"])])

        # Magic Button (for gradient animation)
        self.style.configure("Magic.TButton", font=("Segoe UI", 10, "bold"), background="#6200EA", foreground="#FFFFFF", borderwidth=0, focusthickness=0)
        self.style.map("Magic.TButton", foreground=[('active', '#101010'), ('pressed', '#101010')], background=[('pressed', '#EA80FC')])

        # General Styles
        default_font = ("Segoe UI", 10) if platform.system() == "Windows" else ("Helvetica", 10)
        bold_font = ("Segoe UI", 10, "bold") if platform.system() == "Windows" else ("Helvetica", 10, "bold")

        self.style.configure(".", background=COLORS["bg"], foreground=COLORS["fg"], fieldbackground=COLORS["panel_bg"], font=default_font)
        self.style.configure("TButton", padding=6, relief="flat", background=COLORS["card_bg"], foreground=COLORS["fg"], borderwidth=0, focusthickness=0, focuscolor=COLORS["card_bg"])
        self.style.map("TButton", background=[('active', COLORS["accent"]), ('pressed', COLORS["accent_hover"])], foreground=[('active', 'white')],
                       focuscolor=[('active', COLORS["accent"]), ('!active', COLORS["card_bg"])])

        self.style.configure("Accent.TButton", background=COLORS["accent"], foreground="white", font=bold_font, focuscolor=COLORS["accent"])
        self.style.map("Accent.TButton", background=[('active', COLORS["accent_hover"])], focuscolor=[('active', COLORS["accent_hover"])])
        self.style.configure("Success.TButton", background=COLORS["card_bg"], foreground=COLORS["add"], bordercolor=COLORS["card_bg"], focuscolor=COLORS["card_bg"])
        self.style.map("Success.TButton", background=[('active', COLORS["add"]), ('pressed', COLORS["add_hover"])], foreground=[('active', COLORS["text_dark"])],
                       focuscolor=[('active', COLORS["add"]), ('!active', COLORS["card_bg"])])
        self.style.configure("Danger.TButton", background=COLORS["card_bg"], foreground=COLORS["remove"], focuscolor=COLORS["card_bg"])
        self.style.map("Danger.TButton", background=[('active', COLORS["remove"]), ('pressed', COLORS["remove_hover"])], foreground=[('active', COLORS["text_dark"])],
                       focuscolor=[('active', COLORS["remove"]), ('!active', COLORS["card_bg"])])
        self.style.configure("Warning.TButton", background=COLORS["card_bg"], foreground=COLORS["undo"], focuscolor=COLORS["card_bg"])
        self.style.map("Warning.TButton", background=[('active', COLORS["undo"]), ('pressed', COLORS["undo_hover"])], foreground=[('active', COLORS["text_dark"])],
                       focuscolor=[('active', COLORS["undo"]), ('!active', COLORS["card_bg"])])
        self.style.configure("Flash.TButton", background="#FFD700", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.style.configure("Export.TButton", background=COLORS["export"], foreground="white", font=bold_font, focuscolor=COLORS["export"])
        self.style.map("Export.TButton", background=[('active', COLORS["export_hover"]), ('pressed', "#5830a8")], foreground=[('active', 'white')], focuscolor=[('active', COLORS["export_hover"])])
        self.style.configure("Clear.TButton", background="#2D2D30", foreground="#DEA1CD", bordercolor="#2D2D30", focuscolor="#2D2D30")
        self.style.map("Clear.TButton", background=[('active', "#DEA1CD"), ('pressed', "#c58ebf")], foreground=[('active', "#101027"), ('pressed', "#101027")],
                       focuscolor=[('active', "#DEA1CD"), ('!active', "#2D2D30")])
        self.style.configure("Reset.TButton", background=COLORS["card_bg"], foreground=COLORS["fg"], focuscolor=COLORS["card_bg"])
        self.style.map("Reset.TButton", background=[('active', '#141414'), ('pressed', '#141414')], foreground=[('active', '#FF4242'), ('pressed', '#FF4242')])

        self.style.configure("Processing.TButton", foreground=COLORS["undo"])
        self.style.configure("TFrame", background=COLORS["bg"])
        self.style.configure("Card.TFrame", background=COLORS["panel_bg"], relief="flat")
        self.style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["fg"])
        self.style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=COLORS["header"], background=COLORS["panel_bg"])
        self.style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#999999", background=COLORS["panel_bg"])

        self.style.configure("TCombobox", fieldbackground=COLORS["card_bg"], background=COLORS["bg"], foreground=COLORS["fg"], arrowcolor=COLORS["fg"], borderwidth=0)
        self.style.map('TCombobox', fieldbackground=[('readonly', COLORS["card_bg"])], selectbackground=[('readonly', COLORS["accent"])])

        self.style.configure("Horizontal.TScale", background=COLORS["panel_bg"], troughcolor=COLORS["bg"], borderwidth=0)
        self.style.configure("TSeparator", background=COLORS["border"])

        # Scrollbar: remove standard arrows/grips (Dark mode aesthetic)
        try:
            self.style.element_create("NoGrip.Vertical.Scrollbar.thumb", "from", "alt")
            self.style.layout('NoArrow.Vertical.TScrollbar', [('Vertical.Scrollbar.trough', {'children': [('NoGrip.Vertical.Scrollbar.thumb', {'expand': '1', 'sticky': 'nswe'})], 'sticky': 'ns'})])
            self.style.configure('NoArrow.Vertical.TScrollbar', troughcolor=COLORS["bg"], background=COLORS["card_bg"], relief="flat", borderwidth=0, width=8, arrowsize=8)
            self.style.map('NoArrow.Vertical.TScrollbar', background=[('active', COLORS["accent"]), ('pressed', COLORS["accent_hover"])])

            self.style.element_create("NoGrip.Horizontal.Scrollbar.thumb", "from", "alt")
            self.style.layout('NoArrow.Horizontal.TScrollbar',
                              [('Horizontal.Scrollbar.trough', {'children': [('NoGrip.Horizontal.Scrollbar.thumb', {'expand': '1', 'sticky': 'we'})], 'sticky': 'we'})])
            self.style.configure('NoArrow.Horizontal.TScrollbar', troughcolor=COLORS["bg"], background=COLORS["card_bg"], relief="flat", borderwidth=0, height=8)
            self.style.map('NoArrow.Horizontal.TScrollbar', background=[('active', COLORS["accent"]), ('pressed', COLORS["accent_hover"])])
        except tk.TclError:
            pass  # Ignore if Tcl version doesn't support the hack.

    def _interpolate_color(self, color1, color2, t):
        # Utility for smooth color transitions (used for the 'Magic' button)
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _get_contrast_text_color(self, hex_color):
        """Returns white or black text color based on background brightness (luminance check)."""
        if not hex_color: return "#ffffff"
        h = hex_color.lstrip('#')
        try:
            r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            return "#000000" if luminance > 128 else "#ffffff"
        except Exception:
            return "#ffffff"

    def load_config(self):
        """Loads or creates default configuration from settings.json."""
        default_config = {
            "output_folder": "", "save_file_type": "png", "save_file_quality": 90, "save_mask": False,
            "bg_mode": "transparent", "bg_custom_color": "#0000FF", "enable_shadow": False,
            "shadow_opacity": 0.5, "shadow_radius": 10, "shadow_x": 50, "shadow_y": 50,
            "soften_radius": 0, "last_sam_model": "No Models Found", "last_whole_model": "No Models Found",
            "window_zoomed": True,
            "picker_geometry": None
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)  # Merge loaded over defaults
            except Exception as e:
                print(f"Error loading config: {e}")
        return default_config

    def save_config(self):
        """Saves current state variables to settings.json."""
        # Update config dictionary with current UI state
        self.config["bg_mode"] = self.bg_mode
        self.config["bg_custom_color"] = self.bg_custom_color
        self.config["enable_shadow"] = self.enable_shadow_var.get()
        self.config["shadow_opacity"] = self.shadow_opacity_slider.get()
        self.config["shadow_radius"] = self.shadow_radius_slider.get()
        self.config["shadow_x"] = self.shadow_x_slider.get()
        self.config["shadow_y"] = self.shadow_y_slider.get()
        self.config["soften_radius"] = self.blur_radius_var.get()
        self.config["last_sam_model"] = self.sam_combo.get()
        self.config["last_whole_model"] = self.whole_image_combo.get()
        self.config["save_file_type"] = self.export_format

        # Save window state/geometry
        if platform.system() == "Windows":
            self.config["window_zoomed"] = (self.root.state() == 'zoomed')
        if not self.config["window_zoomed"]:
            self.config["window_width"] = self.root.winfo_width()
            self.config["window_height"] = self.root.winfo_height()

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    # --- DRAG AND DROP HANDLING ---

    def parse_tkdnd_paths(self, data):
        """Parses the DND string format into a list of file paths."""
        import re
        # Pattern matches content inside curly braces OR text without spaces
        pattern = re.compile(r'\{.*?\}|\S+')
        paths = []
        for match in pattern.findall(data):
            path = match.strip('{}')  # Remove the curly braces if present
            if path:
                paths.append(path)
        return paths

    def on_drop(self, event):
        """Handles drop event, distinguishing files/folders and single/multiple drops."""
        raw_paths = self.parse_tkdnd_paths(event.data)

        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        import_paths = []

        # Recursively scan folders or add files
        for path in raw_paths:
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        import_paths.append(os.path.join(path, f))
            elif os.path.isfile(path):
                if os.path.splitext(path)[1].lower() in valid_exts:
                    import_paths.append(path)

        if not import_paths:
            return

        # Check where the user dropped the file to decide on load vs. gallery add
        target_widget = self.root.winfo_containing(event.x_root, event.y_root)
        is_gallery_target = (target_widget and str(target_widget) == str(self.gallery_canvas))

        if len(import_paths) > 1 or is_gallery_target:
            # Multiple files or dropped on gallery -> Add to Gallery only
            self.process_import_paths(import_paths)
            self.status_label.config(text=f"Added {len(import_paths)} files to gallery.")

        elif len(import_paths) == 1:
            single_path = import_paths[0]
            # Dropped on Main Window -> Load into editor AND add to gallery
            self.process_import_paths([single_path])
            self.load_image_path(single_path)
            self.image_paths = [single_path]
            self.current_image_index = 0
            self.initialise_new_image()
            self.root.title("Background Remover - " + os.path.basename(single_path))
            self.status_label.config(text=f"Loaded & Added: {os.path.basename(single_path)}")

    def load_image_path(self, path):
        """Loads image, handles EXIF rotation, and stores metadata."""
        print(f"Image Loaded: {path}")
        self.original_image = Image.open(path)
        self.original_image = ImageOps.exif_transpose(self.original_image)  # Correct rotation
        self.image_exif = self.original_image.info.get('exif')  # Preserve EXIF for saving

    def on_resize(self, event):
        """Handles responsive resizing. Recalculates zoom factor based on new canvas size."""
        if event.widget == self.root:
            if not self.root.winfo_height() == self.init_height or not self.root.winfo_width() == self.init_width:
                # Reset heavy objects if size changes (e.g., SAM embeddings)
                if hasattr(self, "encoder_output"):
                    delattr(self, "encoder_output")

                self.init_width = self.root.winfo_width()
                self.init_height = self.root.winfo_height()
                self.root.update()

                self.canvas_w = self.canvas.winfo_width()
                self.canvas_h = self.canvas.winfo_height()

                # Calculate new minimum zoom to fit the image
                self.lowest_zoom_factor = min(self.canvas_w / self.original_image.width, self.canvas_h / self.original_image.height)
                self.zoom_factor = self.lowest_zoom_factor
                self.view_x = 0
                self.view_y = 0
                self.min_zoom = True
                self.checkerboard = self.create_checkerboard(self.canvas_w * 2, self.canvas_h * 2, square_size=10)

                self.update_input_image_preview(Image.BOX)
                self.update_zoom_label()

    def update_zoom_label(self):
        effective_zoom = int(self.zoom_factor * 100)
        self.zoom_label.config(text=f"Pixel Ratio: {effective_zoom}%")

    def setup_image_display(self):
        """Initializes workspace state when a new image is loaded."""
        self.lowest_zoom_factor = min(self.canvas_w / self.original_image.width, self.canvas_h / self.original_image.height)
        self.working_image = Image.new("RGBA", self.original_image.size, (0, 0, 0, 0))  # Final cut-out
        self.working_mask = Image.new("L", self.original_image.size, 0)  # Grayscale mask (L for Luminance)

        # Undo/Redo Stacks
        self.undo_history_mask = []
        self.undo_history_mask.append(self.working_mask.copy())
        self.redo_history_mask = []

        self.zoom_factor = self.lowest_zoom_factor
        self.view_x = 0
        self.view_y = 0
        self.min_zoom = True
        self.checkerboard = self.create_checkerboard(self.canvas_w * 2, self.canvas_h * 2, square_size=10)
        if hasattr(self, 'zoom_label'):
            self.update_zoom_label()

    def create_checkerboard(self, width, height, square_size):
        """Generates a tiled background image for transparency visualization."""
        num_squares_x = width // square_size
        num_squares_y = height // square_size
        img = Image.new('RGBA', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        colors = [(40, 40, 40), (60, 60, 60)]  # Dark gray shades
        for row in range(num_squares_y):
            for col in range(num_squares_x):
                color = colors[(row + col) % 2]
                draw.rectangle([(col * square_size, row * square_size), ((col + 1) * square_size, (row + 1) * square_size)], fill=color)
        return img

    def update_button_visual(self, btn, variable):
        """Helper to style toggle buttons based on their state."""
        if variable.get():
            btn.configure(bg=COLORS["accent"], fg="white")
        else:
            btn.configure(bg=COLORS["card_bg"], fg=COLORS["fg"])

    def create_flat_toggle(self, parent, text, variable, command=None):
        """Creates a custom flat toggle button (manual styling)."""
        btn = tk.Button(parent, text=text, relief="flat", borderwidth=0, highlightthickness=0, bg=COLORS["card_bg"], fg=COLORS["fg"], activebackground=COLORS["accent_hover"], activeforeground="white")

        def internal_toggle():
            variable.set(not variable.get())
            self.update_button_visual(btn, variable)
            if command: command()  # Run user command

        btn.configure(command=internal_toggle)
        self.update_button_visual(btn, variable)
        return btn

    def build_gui(self):
        """Constructs the main application window and layout."""
        try:
            current_dpi = self.root.winfo_fpixels('1i')
            scale_factor = current_dpi / 96.0
        except Exception:
            scale_factor = 1.0

        # Responsive Min-Size calculation
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        base_target_w = 1200
        base_target_h = 700
        scaled_w = int(base_target_w * scale_factor)
        scaled_h = int(base_target_h * scale_factor)

        final_min_w = min(scaled_w, int(screen_w * 0.9))
        final_min_h = min(scaled_h, int(screen_h * 0.85))
        self.root.minsize(width=final_min_w, height=final_min_h)

        # --- Main Layout (Frame structure) ---
        self.main_frame = tk.Frame(self.root, container=False, name="main_frame", bg=COLORS["bg"])
        self.editor_frame = ttk.Frame(self.main_frame, name="editor_frame")

        # Left Column: Input Canvas
        self.input_frame = tk.Frame(self.editor_frame, name="input_frame", bg=COLORS["bg"])
        self.input_header = tk.Frame(self.input_frame, bg=COLORS["bg"])
        self.input_header.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(self.input_header, text='INPUT SOURCE', style="Header.TLabel", background=COLORS["bg"]).pack(side="left")
        self.canvas = tk.Canvas(self.input_frame, name="canvas", bg="#101010", highlightthickness=1, highlightbackground=COLORS["border"], borderwidth=0)
        self.canvas.pack(expand=True, fill="both", side="top", padx=10, pady=5)
        self.input_frame.pack(expand=True, fill="both", side="left")

        # Separator
        separator_1 = ttk.Separator(self.editor_frame, orient="vertical")
        separator_1.pack(expand=False, fill="y", side="left", padx=0, pady=0)

        # Center Column: Output Canvas
        self.output_frame = tk.Frame(self.editor_frame, name="output_frame", bg=COLORS["bg"])
        self.output_header = tk.Frame(self.output_frame, bg=COLORS["bg"])
        self.output_header.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(self.output_header, text='RESULT COMPOSITE', style="Header.TLabel", background=COLORS["bg"]).pack(side="left")
        self.canvas2 = tk.Canvas(self.output_frame, name="canvas2", bg="#101010", highlightthickness=1, highlightbackground=COLORS["border"], borderwidth=0)
        self.canvas2.pack(expand=True, fill="both", side="top", padx=10, pady=5)
        self.output_frame.pack(expand=True, fill="both", side="left")

        # Right Column: Sidebar (Fixed/responsive width)
        sidebar_width = int(340 * self.dpi_scale)
        if screen_w < 1300:
            sidebar_width = int(280 * self.dpi_scale)

        self.Controls = tk.Frame(self.editor_frame, name="controls", bg=COLORS["panel_bg"], width=sidebar_width)
        self.Controls.pack_propagate(False)

        # Sidebar Scroll (Canvas/Frame trick for scrollable sidebar)
        self.ctrl_scrollbar = ttk.Scrollbar(self.Controls, orient="vertical", style="NoArrow.Vertical.TScrollbar", command=lambda *args: self.ctrl_canvas.yview(*args))
        self.ctrl_canvas = tk.Canvas(self.Controls, bg=COLORS["panel_bg"], highlightthickness=0, yscrollcommand=self.ctrl_scrollbar.set)
        self.ctrl_scrollbar.pack(side="right", fill="y")
        self.ctrl_canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_inner = tk.Frame(self.ctrl_canvas, bg=COLORS["panel_bg"])
        self.canvas_window_id = self.ctrl_canvas.create_window((0, 0), window=self.scrollable_inner, anchor="nw")
        self.scrollable_inner.bind("<Configure>", lambda e: self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all")))
        self.ctrl_canvas.bind("<Configure>", self._configure_canvas_window)

        # Pad Container for content inside scrollable frame
        pad_ctrl = tk.Frame(self.scrollable_inner, bg=COLORS["panel_bg"])
        pad_ctrl.pack(fill="both", expand=True, padx=15, pady=15)

        # --- SECT 1: Model Selection ---
        self.ModelSelection = ttk.Frame(pad_ctrl, style="Card.TFrame", padding=10)
        ttk.Label(self.ModelSelection, text='AI INTELLIGENCE', style="Header.TLabel").pack(anchor="w", fill="x", pady=(0, 10))

        # --- CPU/GPU BUTTONS (MOVED HERE) ---
        # Container frame with requested background
        self.hw_selection_frame = tk.Frame(self.ModelSelection, bg="#2D2D30")
        self.hw_selection_frame.pack(fill="x", pady=(0, 12))  # Some space below before the dropdowns

        self.btn_run_cpu = tk.Button(self.hw_selection_frame, text="Run on CPU", relief="flat",
                                     borderwidth=0, command=lambda: self.set_hardware_mode("CPU"))
        self.btn_run_cpu.pack(side="left", fill="x", expand=True, padx=(0, 1), ipady=4)

        self.btn_run_gpu = tk.Button(self.hw_selection_frame, text="Run on GPU", relief="flat",
                                     borderwidth=0, command=lambda: self.set_hardware_mode("GPU"))
        self.btn_run_gpu.pack(side="left", fill="x", expand=True, padx=(1, 0), ipady=4)

        # SAM Model Dropdown (Custom widget)
        self.sam_combo = ModernComboGroup(self.ModelSelection, label_text="SAM MODEL")
        self.sam_combo.pack(fill="x", pady=(0, 8))

        # Whole Image Model Dropdown (Custom widget)
        self.whole_image_combo = ModernComboGroup(self.ModelSelection, label_text="WHOLE IMG")
        self.whole_image_combo.pack(fill="x", pady=(0, 10))

        # Pre-load models button (runs in thread)
        self.load_models_btn = ttk.Button(self.ModelSelection, text='Pre Load Models', command=self.load_selected_models, style="TButton")
        self.load_models_btn.pack(fill="x", pady=(1, 0))

        self.update_hardware_buttons_visual()
        self.ModelSelection.pack(fill="x", pady=(0, 10), side="top")

        # --- SECT 2: File Manager ---
        self.FileManager = ttk.Frame(pad_ctrl, style="Card.TFrame", padding=10)
        ttk.Label(self.FileManager, text='FILE MANAGEMENT', style="Header.TLabel").pack(anchor="w", fill="x", pady=(0, 10))

        # Import buttons
        self.import_buttons_row = tk.Frame(self.FileManager, bg=COLORS["panel_bg"])
        self.btn_import_files = ttk.Button(self.import_buttons_row, text='Import Files', command=self._import_files_action, style="TButton")
        self.btn_import_files.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_import_folder = ttk.Button(self.import_buttons_row, text='Import Folder', command=self._import_folder_action, style="TButton")
        self.btn_import_folder.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.import_buttons_row.pack(fill="x", pady=(0, 8))

        # Gallery Container (Horizontal scroll)
        self.gallery_container = tk.Frame(self.FileManager, bg=COLORS["bg"], height=170)
        self.gallery_container.pack(fill="x", pady=(0, 8))
        self.gallery_container.pack_propagate(False)

        self.gallery_scroll = ttk.Scrollbar(self.gallery_container, orient="horizontal", style="NoArrow.Horizontal.TScrollbar")
        self.gallery_canvas = tk.Canvas(self.gallery_container, bg="#151515", highlightthickness=0, height=160, xscrollcommand=self.gallery_scroll.set)

        self.gallery_scroll.config(command=self.gallery_canvas.xview)
        self.gallery_scroll.pack(side="bottom", fill="x")
        self.gallery_canvas.pack(side="top", fill="both", expand=True)

        # Gallery bindings
        self.gallery_canvas.bind("<Button-1>", self.on_gallery_click)
        self.gallery_canvas.bind("<Motion>", self.on_gallery_hover_move)
        self.gallery_canvas.bind("<Leave>", self.on_gallery_leave)

        # Gallery action buttons (Use, Delete, Clean)
        self.action_row = tk.Frame(self.FileManager, bg=COLORS["panel_bg"])
        self.btn_use_img = ttk.Button(self.action_row, text='Use', command=self.use_gallery_image, style="Use.File.TButton", takefocus=False)
        self.btn_use_img.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.btn_del_img = ttk.Button(self.action_row, text='Delete', command=self.delete_gallery_image, style="Delete.File.TButton", takefocus=False)
        self.btn_del_img.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_clean_all = ttk.Button(self.action_row, text='Clean', command=self.clean_gallery, style="Clean.File.TButton", takefocus=False)
        self.btn_clean_all.pack(side="left", fill="x", expand=True, padx=(2, 0))
        self.action_row.pack(fill="x", pady=0)
        self.FileManager.pack(fill="x", pady=(0, 10), side="top")

        # --- SECT 3: Mask Generation ---
        self.Edit = ttk.Frame(pad_ctrl, style="Card.TFrame", padding=10)
        ttk.Label(self.Edit, text='MASK GENERATION', style="Header.TLabel").pack(anchor="w", fill="x", pady=(0, 10))

        # Whole Image Auto-Detect (The 'Magic' button)
        self.whole_image_button = ttk.Button(self.Edit, text='AUTO-DETECT SUBJECT', command=lambda: self.run_whole_image_model(None), style="Magic.TButton", takefocus=False)
        self.whole_image_button.pack(fill="x", pady=(0, 10), ipady=5)
        # Magic button hover animation binds
        self.whole_image_button.bind("<Enter>", self.start_magic_anim)
        self.whole_image_button.bind("<Leave>", self.stop_magic_anim)

        # SAM Manual Detection activation
        self.manual_sam_button = ttk.Button(self.Edit, text='MANUAL DETECTION', command=self.activate_sam_mode, style="Accent.TButton")
        self.manual_sam_button.pack(fill="x", pady=(0, 15), ipady=5)

        tk.Label(self.Edit, text="REFINE MASK", bg=COLORS["panel_bg"], fg=COLORS["fg"], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))

        # Mask Refinement Actions (Add, Subtract, Undo)
        self.EditCluster = tk.Frame(self.Edit, bg=COLORS["panel_bg"])
        self.Add = ttk.Button(self.EditCluster, text='Add', command=self.add_to_working_image, style="Success.TButton")
        self.Add.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.Remove = ttk.Button(self.EditCluster, text='Subtract', command=self.subtract_from_working_image, style="Danger.TButton")
        self.Remove.pack(side="left", fill="x", expand=True, padx=4)
        self.Undo = ttk.Button(self.EditCluster, text='Undo', command=self.undo, style="Warning.TButton")
        self.Undo.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.EditCluster.pack(expand=True, fill="x", pady=(0, 10))

        # Additional tools (Clear, Redo)
        self.tools_frame = tk.Frame(self.Edit, bg=COLORS["panel_bg"])
        ttk.Button(self.tools_frame, text='Clear source', command=self.clear_coord_overlay, style="Clear.TButton").pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(self.tools_frame, text='Clear result', command=self.clear_working_image, style="Clear.TButton").pack(side="left", expand=True, fill="x", padx=4)
        self.btn_redo = ttk.Button(self.tools_frame, text='Redo', command=self.redo, style="Warning.TButton")
        self.btn_redo.pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.tools_frame.pack(fill="x", pady=(0, 10))

        # Unified Threshold Slider (SleekSlider widget)
        self.unified_slider_frame = tk.Frame(self.Edit, bg=COLORS["panel_bg"])
        self.unified_slider_frame.pack(fill="x", pady=(5, 10))
        slider_row = tk.Frame(self.unified_slider_frame, bg=COLORS["panel_bg"])
        slider_row.pack(fill="x", expand=True)
        self.slider_val_label = tk.Label(slider_row, text="50%", bg=COLORS["panel_bg"],
                                         fg=COLORS["accent"], font=("Segoe UI", 10, "bold"), width=4)
        self.slider_val_label.pack(side="right", padx=(5, 0))
        self.unified_var = tk.DoubleVar(value=50.0)

        def slider_callback(val):
            self.unified_var.set(val)
            self.on_unified_slider_change(val)

        self.custom_slider = SleekSlider(slider_row, height=20, min_val=0, max_val=100, init_val=50,
                                         bg_color=COLORS["panel_bg"],
                                         accent_color=COLORS["accent"],
                                         command=slider_callback)
        self.custom_slider.pack(side="left", fill="x", expand=True)

        # Global actions (Reset / Copy In->Out)
        ttk.Separator(self.Edit, orient="horizontal").pack(fill="x", pady=8)
        self.global_ops_frame = tk.Frame(self.Edit, bg=COLORS["panel_bg"])
        ttk.Button(self.global_ops_frame, text='Reset Workpage', command=self.reset_all, style="Reset.TButton", takefocus=False).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ttk.Button(self.global_ops_frame, text='Copy In→Out', command=self.copy_entire_image, style="Copy.TButton").pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.global_ops_frame.pack(fill="x", pady=2)
        self.Edit.pack(fill="x", pady=(0, 15), side="top")

        # --- SECT 4: Compositing ---
        self.Options = ttk.Frame(pad_ctrl, style="Card.TFrame", padding=10)
        ttk.Label(self.Options, text='TOOLS & COMPOSITING', style="Header.TLabel").pack(anchor="w", fill="x", pady=(0, 10))

        self.StackFrame = tk.Frame(self.Options, bg=COLORS["panel_bg"])
        self.StackFrame.pack(fill="x")

        # Viewport Interaction Toggle Buttons
        self.ToolToggleFrame = tk.Frame(self.StackFrame, bg=COLORS["panel_bg"])
        self.btn_get_area = tk.Button(self.ToolToggleFrame, text="Box Mode", command=self.toggle_area_mode, relief="flat", bg=COLORS["card_bg"], fg=COLORS["fg"], borderwidth=0, highlightthickness=0)
        self.btn_get_area.pack(side="left", fill="x", expand=True, padx=(0, 1), pady=0, ipady=5)
        self.btn_move = tk.Button(self.ToolToggleFrame, text="Pan Mode", command=self.toggle_move_mode, relief="flat", bg=COLORS["accent"], fg="white", borderwidth=0, highlightthickness=0)
        self.btn_move.pack(side="left", fill="x", expand=True, padx=0, pady=0, ipady=5)
        self.ToolToggleFrame.pack(fill="x", pady=(0, 2))

        # Background Selector (Transparent / Color / Blur)
        self.BgSel = tk.Frame(self.StackFrame, bg=COLORS["panel_bg"])
        self.bg_btn_row = tk.Frame(self.BgSel, bg=COLORS["panel_bg"])
        self.btn_bg_trans = tk.Button(self.bg_btn_row, text="Transparent", relief="flat", bg=COLORS["accent"], fg="white", borderwidth=0, command=lambda: self.set_bg_mode("transparent"))
        self.btn_bg_trans.pack(side="left", fill="x", expand=True, padx=(0, 1), ipady=3)
        self.btn_bg_color = tk.Button(self.bg_btn_row, text="Color", relief="flat", bg=COLORS["card_bg"], fg=COLORS["fg"], borderwidth=0, command=lambda: self.set_bg_mode("color"))
        self.btn_bg_color.pack(side="left", fill="x", expand=True, padx=1, ipady=3)
        self.btn_bg_blur = tk.Button(self.bg_btn_row, text="Blur", relief="flat", bg=COLORS["card_bg"], fg=COLORS["fg"], borderwidth=0, command=lambda: self.set_bg_mode("blur"))
        self.btn_bg_blur.pack(side="left", fill="x", expand=True, padx=(1, 0), ipady=3)
        self.bg_btn_row.pack(fill="x", pady=(0, 5))

        # Background Color/Blur Options (Dynamically packed)
        self.bg_extra_frame = tk.Frame(self.BgSel, bg=COLORS["panel_bg"])
        self.bg_extra_frame.pack(fill="x")
        self.btn_pick_color = tk.Button(self.bg_extra_frame, text="Color Picker", bg=self.bg_custom_color, fg="white", relief="flat", command=self.pick_bg_color)

        self.blur_slider_frame = tk.Frame(self.bg_extra_frame, bg=COLORS["panel_bg"])
        tk.Label(self.blur_slider_frame, text="Intensity:", bg=COLORS["panel_bg"], fg=COLORS["fg"], font=("Arial", 8)).pack(side="left")
        self.bg_blur_slider = SleekSlider(self.blur_slider_frame, width=150, height=20, min_val=5, max_val=100, init_val=20, command=self.on_blur_slider_change, bg_color=COLORS["panel_bg"],
                                          accent_color=COLORS["accent"])
        self.bg_blur_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.BgSel.pack(fill="x", pady=(15, 15))

        # Paint Mode Toggle
        self.paint_mode = tk.BooleanVar(value=False)
        self.btn_paint = self.create_flat_toggle(self.StackFrame, "Manual Paintbrush", self.paint_mode, self.paint_mode_toggle)
        self.btn_paint.pack(fill="x", pady=(0, 1), ipady=5)

        # Brush Size Slider (Dynamically packed)
        self.brush_options_frame = tk.Frame(self.StackFrame, bg=COLORS["panel_bg"])
        self.brush_size_var = tk.IntVar(value=PAINT_BRUSH_DIAMETER)
        f_brush = tk.Frame(self.brush_options_frame, bg=COLORS["panel_bg"])
        tk.Label(f_brush, text="Size:", width=6, bg=COLORS["panel_bg"], fg=COLORS["fg"]).pack(side="left")

        def update_brush_var(v):
            self.brush_size_var.set(int(v))

        self.brush_slider = SleekSlider(f_brush, width=150, height=20, min_val=1, max_val=100, init_val=self.brush_size_var.get(), command=update_brush_var, bg_color=COLORS["panel_bg"],
                                        accent_color=COLORS["accent"])
        self.brush_slider.pack(side="left", fill="x", expand=True, padx=5)
        f_brush.pack(fill="x", padx=10, pady=5)

        # Alpha Channel Toggle
        self.show_mask_var = tk.BooleanVar(value=False)
        self.btn_mask = self.create_flat_toggle(self.StackFrame, "Alpha Channel", self.show_mask_var, self.on_alpha_channel_toggle)
        self.btn_mask.pack(fill="x", pady=(0, 1), ipady=5)

        # Soften Edges Toggle (Gaussian Blur on mask)
        self.soften_mask_var = tk.BooleanVar(value=False)
        self.btn_soften = self.create_flat_toggle(self.StackFrame, "Soften Edges", self.soften_mask_var, self.toggle_soften_options)
        self.btn_soften.pack(fill="x", pady=(0, 1), ipady=5)

        # Soften Options (Slider, dynamically packed)
        self.soften_options_frame = tk.Frame(self.StackFrame, bg=COLORS["panel_bg"])
        f_soften_inner = tk.Frame(self.soften_options_frame, bg=COLORS["panel_bg"])
        f_soften_inner.pack(fill="x", padx=10, pady=5)
        loaded_soften = self.config.get("soften_radius", 0)
        self.blur_radius_var = tk.IntVar(value=loaded_soften)
        tk.Label(f_soften_inner, text="Blur:", bg=COLORS["panel_bg"], fg=COLORS["fg"], font=("Segoe UI", 8)).pack(side="left")
        self.lbl_soften_val = tk.Label(f_soften_inner, text=str(loaded_soften), width=4, bg=COLORS["panel_bg"], fg=COLORS["fg"], font=("Segoe UI", 8, "bold"))
        self.lbl_soften_val.pack(side="right")

        def update_soften_label(val):
            int_val = int(float(val))
            self.blur_radius_var.set(int_val)
            self.lbl_soften_val.config(text=str(int_val))
            if hasattr(self, 'working_mask'):
                self.add_drop_shadow()

        self.soften_slider = SleekSlider(f_soften_inner, width=150, height=20, min_val=0, max_val=100, init_val=loaded_soften, command=update_soften_label, bg_color=COLORS["panel_bg"],
                                         accent_color=COLORS["accent"])
        self.soften_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Drop Shadow Toggle
        self.enable_shadow_var = tk.BooleanVar(value=False)
        self.btn_shadow = self.create_flat_toggle(self.StackFrame, "Drop Shadow", self.enable_shadow_var, self.toggle_shadow_options)
        self.btn_shadow.pack(fill="x", pady=0, ipady=5)
        self.shadow_options_frame = tk.Frame(self.Options, bg=COLORS["panel_bg"])

        # Helper to create shadow sliders
        def make_shadow_slider(parent, txt, vmin, vmax, attr, init_val):
            f = tk.Frame(parent, bg=COLORS["panel_bg"])
            tk.Label(f, text=txt, width=6, bg=COLORS["panel_bg"], fg=COLORS["fg"], font=("Arial", 8)).pack(side="left")

            def cb(val): self.add_drop_shadow()

            s = SleekSlider(f, width=150, height=20, min_val=vmin, max_val=vmax, init_val=init_val, command=cb, bg_color=COLORS["panel_bg"], accent_color=COLORS["accent"])
            s.pack(side="left", fill="x", expand=True, padx=5)
            f.pack(fill="x", pady=2)
            setattr(self, attr, s)

        # Shadow Options Sliders
        make_shadow_slider(self.shadow_options_frame, "Opac:", 0, 1, "shadow_opacity_slider", self.config.get("shadow_opacity", 0.5))
        make_shadow_slider(self.shadow_options_frame, "Blur:", 1, 50, "shadow_radius_slider", self.config.get("shadow_radius", 10))
        make_shadow_slider(self.shadow_options_frame, "X:", -100, 100, "shadow_x_slider", self.config.get("shadow_x", 50))
        make_shadow_slider(self.shadow_options_frame, "Y:", -100, 100, "shadow_y_slider", self.config.get("shadow_y", 50))

        self.Options.pack(fill="x", pady=(0, 15), side="top")

        # --- SECT 5: Export ---
        self.SaveFrame = ttk.Frame(pad_ctrl, style="Card.TFrame", padding=10)
        ttk.Label(self.SaveFrame, text="FINAL EXPORT", style="Header.TLabel").pack(anchor="w", fill="x", pady=(0, 10))

        # Export Format Buttons (PNG/JPG/WEBP)
        self.fmt_btn_row = tk.Frame(self.SaveFrame, bg=COLORS["panel_bg"])
        self.btn_fmt_png = tk.Button(self.fmt_btn_row, text="PNG", relief="flat", bg=COLORS["card_bg"], fg=COLORS["fg"], borderwidth=0, command=lambda: self.set_export_format("png"))
        self.btn_fmt_png.pack(side="left", fill="x", expand=True, padx=(0, 1), ipady=3)
        self.btn_fmt_jpg = tk.Button(self.fmt_btn_row, text="JPG", relief="flat", bg=COLORS["card_bg"], fg=COLORS["fg"], borderwidth=0, command=lambda: self.set_export_format("jpg"))
        self.btn_fmt_jpg.pack(side="left", fill="x", expand=True, padx=1, ipady=3)
        self.btn_fmt_webp = tk.Button(self.fmt_btn_row, text="WEBP", relief="flat", bg=COLORS["card_bg"], fg=COLORS["fg"], borderwidth=0, command=lambda: self.set_export_format("webp"))
        self.btn_fmt_webp.pack(side="left", fill="x", expand=True, padx=(1, 0), ipady=3)
        self.fmt_btn_row.pack(fill="x", pady=(0, 8))

        # Output Folder path/selection
        self.out_folder_btn = ttk.Button(self.SaveFrame, text='Set Output Folder', command=self.set_output_folder)
        self.out_folder_btn.pack(fill="x", pady=2)
        self.folder_path_var = tk.StringVar()
        self.folder_label = tk.Label(self.SaveFrame, textvariable=self.folder_path_var, width=32, anchor='center', bg=COLORS["card_bg"], fg="gray")
        self.folder_label.pack(fill="x", pady=(0, 5))
        # Quick Export button
        self.auto_save_btn = ttk.Button(self.SaveFrame, text='QUICK EXPORT', command=self.quick_save_automatic, style="Export.TButton")
        self.auto_save_btn.pack(fill="x", pady=4)
        # Save As dialog button
        self.btn_save_as = ttk.Button(self.SaveFrame, text='Save As...', command=self.save_as_image, style="TButton")
        self.btn_save_as.pack(fill="x", pady=2)

        self.SaveFrame.pack(fill="x", pady=5, side="top")

        # Footer (Help button)
        tk.Frame(pad_ctrl, bg=COLORS["panel_bg"]).pack(side="top", expand=True, fill="both")
        ttk.Button(pad_ctrl, text='Help / About', command=self.show_help).pack(fill="x", pady=(20, 10), side="bottom")

        # Pack Control Panel
        self.Controls.pack(fill="y", side="right")
        self.editor_frame.pack(expand=True, fill="both", side="top")

        # Status Bar (at the bottom)
        self.messagerow = tk.Frame(self.main_frame, background=COLORS["accent"], height=28)
        self.status_label = tk.Label(self.messagerow, text='Status: Ready', bg=COLORS["accent"], fg="white", anchor="w", font=("Segoe UI", 9, "bold"))
        self.status_label.pack(expand=True, fill="x", side="left", padx=10)
        self.zoom_label = tk.Label(self.messagerow, text='Zoom: 100%', bg=COLORS["accent"], fg="white", width=15)
        self.zoom_label.pack(side="left")
        self.messagerow.pack(fill="x", side="bottom")
        self.main_frame.pack(expand=True, fill="both", side="top")

        # Final setup & restore state
        if DND_AVAILABLE:
            self.canvas.drop_target_register(DND_FILES);
            self.canvas.dnd_bind('<<Drop>>', self.on_drop)
            self.canvas2.drop_target_register(DND_FILES);
            self.canvas2.dnd_bind('<<Drop>>', self.on_drop)
            self.gallery_canvas.drop_target_register(DND_FILES);
            self.gallery_canvas.dnd_bind('<<Drop>>', self.on_drop)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

        # Restore Background/Export state from config
        self.set_bg_mode(self.config.get("bg_mode", "transparent"))
        self.set_export_format(self.export_format)
        if self.config["bg_custom_color"]:
            self.btn_pick_color.config(bg=self.bg_custom_color, fg=self._get_contrast_text_color(self.bg_custom_color))

        self.enable_shadow_var.set(False)  # Toggle state is handled by toggle_shadow_options below
        self.shadow_opacity_slider.set_value(self.config["shadow_opacity"])
        self.shadow_radius_slider.set_value(self.config["shadow_radius"])
        self.shadow_x_slider.set_value(self.config["shadow_x"])
        self.shadow_y_slider.set_value(self.config["shadow_y"])
        self.toggle_shadow_options()  # Apply initial shadow state

        self.update_folder_marquee()
        self.populate_models()
        self._bind_mouse_scroll(self.scrollable_inner)  # Sidebar scroll bind

    def set_export_format(self, fmt):
        """Switches the export format visual state and updates config."""
        self.export_format = fmt
        self.config["save_file_type"] = fmt

        # Reset all buttons
        base_bg = COLORS["card_bg"];
        base_fg = COLORS["fg"]
        active_bg = COLORS["accent"];
        active_fg = "white"

        self.btn_fmt_png.config(bg=base_bg, fg=base_fg)
        self.btn_fmt_jpg.config(bg=base_bg, fg=base_fg)
        self.btn_fmt_webp.config(bg=base_bg, fg=base_fg)

        # Highlight active button
        if fmt == "png":
            self.btn_fmt_png.config(bg=active_bg, fg=active_fg)
        elif fmt == "jpg":
            self.btn_fmt_jpg.config(bg=active_bg, fg=active_fg)
        elif fmt == "webp":
            self.btn_fmt_webp.config(bg=active_bg, fg=active_fg)

        self.save_config()

    def show_floating_error_x(self):
        """Displays a temporary red 'X' under the cursor for invalid actions (e.g., JPG + Transp)."""
        try:
            x, y = self.root.winfo_pointerxy()
            size = 20;
            line_width = 3

            top = Toplevel(self.root)
            top.overrideredirect(True)
            top.attributes('-topmost', True)
            top.geometry(f"{size}x{size}+{x - size // 2}+{y - size // 2}")

            # Windows transparency hack
            bg_col = "#000001"
            if platform.system() == "Windows":
                try:
                    top.attributes('-transparentcolor', bg_col)
                except:
                    pass

            cv = tk.Canvas(top, bg=bg_col, highlightthickness=0)
            cv.pack(fill="both", expand=True)

            pad = 2
            cv.create_line(pad, pad, size - pad, size - pad, fill="#FF0000", width=line_width, capstyle=tk.ROUND)
            cv.create_line(pad, size - pad, size - pad, pad, fill="#FF0000", width=line_width, capstyle=tk.ROUND)

            self.root.after(800, top.destroy)

        except Exception as e:
            print(f"Error showing floating X: {e}")

    def validate_export_config(self, triggering_widget=None, event=None):
        """Checks for the 'JPG + Transparent Background' conflict."""
        if self.export_format == "jpg" and self.bg_mode == "transparent":
            self.show_floating_error_x()
            self.status_label.config(text="Error: JPG format does not support transparency.", fg="white")

            # Flash relevant buttons for user feedback (with cooldown)
            now = timer()
            if not hasattr(self, 'last_trans_flash_time'): self.last_trans_flash_time = 0

            if now - self.last_trans_flash_time > 4.0:
                self.last_trans_flash_time = now
                if hasattr(self, 'btn_bg_trans'):
                    self._flash_trans_step = 0
                    self.animate_transparent_flash()
            return False
        return True

    def animate_transparent_flash(self):
        """Flashes 'Transparent' and 'JPG' buttons yellow to indicate conflict."""
        if not hasattr(self, 'btn_bg_trans') or not self.btn_bg_trans.winfo_exists(): return
        if not hasattr(self, 'btn_fmt_jpg') or not self.btn_fmt_jpg.winfo_exists(): return

        # FIX: Determine correct color based on current selection
        is_jpg_selected = (self.export_format == "jpg")
        
        jpg_rest_bg = COLORS["accent"] if is_jpg_selected else COLORS["card_bg"]
        jpg_rest_fg = "white" if is_jpg_selected else COLORS["fg"]

        if self._flash_trans_step >= 6:
            # End of animation: Restore original state
            self.btn_bg_trans.config(bg=COLORS["accent"], fg="white")
            self.btn_fmt_jpg.config(bg=jpg_rest_bg, fg=jpg_rest_fg) 
            return

        is_yellow = (self._flash_trans_step % 2 == 0)
        if is_yellow:
            self.btn_bg_trans.config(bg="#FFD700", fg="black")
            self.btn_fmt_jpg.config(bg="#FFD700", fg="black")
        else:
            # Animation off-cycle: Restore original state
            self.btn_bg_trans.config(bg=COLORS["accent"], fg="white")
            self.btn_fmt_jpg.config(bg=jpg_rest_bg, fg=jpg_rest_fg)

        self._flash_trans_step += 1
        self.root.after(150, self.animate_transparent_flash)

    def trigger_alpha_conflict_warning(self):
        """Triggers visual feedback when attempting effects while Alpha Channel is active."""
        self.show_floating_error_x()
        self.status_label.config(text="Error: Cannot use effects while viewing Alpha Channel.", fg="white")

        # Cooldown check
        now = timer()
        if not hasattr(self, 'last_alpha_flash_time'): self.last_alpha_flash_time = 0

        if now - self.last_alpha_flash_time > 4.0:
            self.last_alpha_flash_time = now
            self._alpha_flash_step = 0
            self._animate_alpha_flash_loop()

    def _animate_alpha_flash_loop(self):
        """Flashes the Alpha Channel button yellow 3 times."""
        if not hasattr(self, 'btn_mask') or not self.btn_mask.winfo_exists(): return

        if self._alpha_flash_step >= 6:
            # Restore original visual state (logically ON, but showing the variable state)
            self.update_button_visual(self.btn_mask, self.show_mask_var)
            return

        is_yellow = (self._alpha_flash_step % 2 == 0)

        if is_yellow:
            self.btn_mask.configure(bg="#FFD700", fg="black")
        else:
            # Restore the 'ON' state look temporarily
            self.btn_mask.configure(bg=COLORS["accent"], fg="white")

        self._alpha_flash_step += 1
        self.root.after(150, self._animate_alpha_flash_loop)

    def set_bg_mode(self, mode):
        """Switches the background compositing mode (Transparent/Color/Blur)."""
        self.bg_mode = mode

        # Reset button visuals
        self.btn_bg_trans.config(bg=COLORS["card_bg"], fg=COLORS["fg"])
        self.btn_bg_color.config(bg=COLORS["card_bg"], fg=COLORS["fg"])
        self.btn_bg_blur.config(bg=COLORS["card_bg"], fg=COLORS["fg"])

        # Hide all extra options
        self.bg_extra_frame.pack_forget()
        self.btn_pick_color.pack_forget()
        self.blur_slider_frame.pack_forget()

        if mode == "transparent":
            self.btn_bg_trans.config(bg=COLORS["accent"], fg="white")

        elif mode == "color":
            self.btn_bg_color.config(bg=COLORS["accent"], fg="white")
            self.bg_extra_frame.pack(fill="x")
            self.btn_pick_color.pack(fill="x", padx=10, pady=2)

        elif mode == "blur":
            self.btn_bg_blur.config(bg=COLORS["accent"], fg="white")
            self.bg_extra_frame.pack(fill="x")
            self.blur_slider_frame.pack(fill="x", padx=10, pady=2)
            if self.cached_blur_image is None:
                self.regenerate_smart_blur()  # Pre-generate blur if needed

        self.update_output_image_preview()
        self.scrollable_inner.update_idletasks()
        self.refresh_sidebar_scroll()  # Update scroll region

    def pick_bg_color(self):
        """Opens the custom color picker window."""
        # Prevent multiple windows
        if hasattr(self, 'color_picker_window') and self.color_picker_window is not None and self.color_picker_window.winfo_exists():
            self.color_picker_window.lift()
            return

        def on_picker_update(hex_color):
            self.bg_custom_color = hex_color
            text_color = self._get_contrast_text_color(self.bg_custom_color)
            self.btn_pick_color.config(bg=self.bg_custom_color, fg=text_color)
            self.update_output_image_preview()

        def on_picker_close(geo):
            self.config["picker_geometry"] = geo
            self.color_picker_window = None

        self.color_picker_window = ModernColorPicker(
            self.root,
            self.bg_custom_color,
            on_picker_update,
            on_picker_close,
            self.config.get("picker_geometry")
        )

    def on_blur_slider_change(self, val):
        self.current_blur_radius = float(val)
        self.regenerate_smart_blur()
        self.update_output_image_preview()

    def regenerate_smart_blur(self):
        """Creates an Inpainted + Blurred version of the background. Expensive, so it's cached."""
        if not hasattr(self, "original_image"): return

        # Downscale for performance during computationally heavy inpainting
        max_dim = 512
        scale = min(max_dim / self.original_image.width, max_dim / self.original_image.height)
        new_w = int(self.original_image.width * scale)
        new_h = int(self.original_image.height * scale)

        small_img = self.original_image.resize((new_w, new_h), Image.BILINEAR).convert("RGB")
        small_mask = self.working_mask.resize((new_w, new_h), Image.NEAREST)

        cv_img = np.array(small_img)
        mask_arr = np.array(small_mask)

        # Inpainting fills the area occupied by the subject
        try:
            inpainted = cv2.inpaint(cv_img, mask_arr, 3, cv2.INPAINT_TELEA)
        except:
            inpainted = cv_img  # Fallback if cv2 fails

        # Apply Gaussian Blur (Radius scaled by downsample factor)
        radius = int(self.current_blur_radius * scale)
        if radius % 2 == 0: radius += 1  # Radius must be odd
        blurred = cv2.GaussianBlur(inpainted, (radius, radius), 0)

        # Upscale back to original image size
        final_blur = Image.fromarray(blurred).resize(self.original_image.size, Image.BILINEAR)
        self.cached_blur_image = final_blur

    def refresh_sidebar_scroll(self):
        """Forces the scrollbar to recalculate its visible area after packing/forgetting a widget."""
        self.scrollable_inner.update_idletasks()
        req_height = self.scrollable_inner.winfo_reqheight()
        canvas_height = self.ctrl_canvas.winfo_height()
        new_height = max(canvas_height, req_height)
        self.ctrl_canvas.itemconfig(self.canvas_window_id, height=new_height)
        self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all"))

    def _configure_canvas_window(self, event):
        """Ensures the inner frame correctly fills the canvas width."""
        self.ctrl_canvas.itemconfig(self.canvas_window_id, width=event.width)
        req_height = self.scrollable_inner.winfo_reqheight()
        new_height = max(event.height, req_height)
        self.ctrl_canvas.itemconfig(self.canvas_window_id, height=new_height)

    def _bind_mouse_scroll(self, widget):
        """Recursively binds mouse scroll to all child widgets in the sidebar for smooth scrolling."""
        widget.bind("<MouseWheel>", self._on_ctrl_mousewheel)
        widget.bind("<Button-4>", self._on_ctrl_mousewheel)
        widget.bind("<Button-5>", self._on_ctrl_mousewheel)
        for child in widget.winfo_children():
            self._bind_mouse_scroll(child)

    def _on_ctrl_mousewheel(self, event):
        """Handles scrolling the sidebar's canvas."""
        # Close any open dropdowns on scroll
        if hasattr(self, "sam_combo") and self.sam_combo.is_open: self.sam_combo.close_dropdown()
        if hasattr(self, "whole_image_combo") and self.whole_image_combo.is_open: self.whole_image_combo.close_dropdown()

        if self.ctrl_canvas.yview() == (0.0, 1.0): return  # At end

        scroll_units = 0
        if event.num == '??':
            if platform.system() == "Windows":
                scroll_units = int(-1 * (event.delta / 120))
            else:
                scroll_units = int(-1 * event.delta)
        elif event.num == 4:
            scroll_units = -1
        elif event.num == 5:
            scroll_units = 1

        if scroll_units != 0:
            self.ctrl_canvas.yview_scroll(scroll_units, "units")
        return "break"

    def update_folder_marquee(self):
        """Starts the animation for scrolling the output folder path if it's too long."""
        if self.marquee_after_id:
            self.root.after_cancel(self.marquee_after_id)
            self.marquee_after_id = None

        path = self.config["output_folder"]
        if not path: path = "(Input Folder)"

        self.marquee_text = f" {path} "  # Add padding for wrapping effect
        self.marquee_index = 0
        self.marquee_direction = 1
        self.animate_marquee()

    def animate_marquee(self):
        """Marquee animation loop."""
        window_len = 30
        text_len = len(self.marquee_text)

        if text_len <= window_len:
            self.folder_path_var.set(self.marquee_text.strip())
            return

        start = self.marquee_index
        end = start + window_len
        display_text = self.marquee_text[start:end]
        self.folder_path_var.set(display_text)

        # Move logic
        if self.marquee_direction == 1:
            if self.marquee_index < (text_len - window_len):
                self.marquee_index += 1
            else:
                self.marquee_direction = -1
                self.marquee_index -= 1
        else:
            if self.marquee_index > 0:
                self.marquee_index -= 1
            else:
                self.marquee_direction = 1
                self.marquee_index += 1

        self.marquee_after_id = self.root.after(300, self.animate_marquee)

    def toggle_area_mode(self):
        self.area_enabled = not self.area_enabled
        self.update_button_visual(self.btn_get_area, tk.BooleanVar(value=self.area_enabled))

    def toggle_move_mode(self):
        self.move_enabled = not self.move_enabled
        self.update_button_visual(self.btn_move, tk.BooleanVar(value=self.move_enabled))

    def load_selected_models(self):
        """Starts background model loading to avoid UI freeze."""
        self.load_models_btn.configure(state="disabled")
        self.show_loading("Pre-loading Models")

        def heavy_load():
            # Load SAM parts
            self._initialise_sam_model_headless()

            # Load Whole Image Model
            whole_model_name = self.whole_image_combo.get()
            if whole_model_name and whole_model_name != "No Models Found":
                self.thread_safe_load_model(whole_model_name)

            return "Done"

        def on_complete(res):
            self.load_models_btn.configure(state="normal", text="Models Loaded")
            self.status_label.config(text="Models loaded successfully.", fg="white")

        def on_fail(err):
            self.load_models_btn.configure(state="normal", text="Load Models")
            self.hide_loading()
            messagebox.showerror("Load Error", str(err))

        self.start_threaded_task(heavy_load, on_complete, on_fail)

    def populate_models(self):
        """Scans Models/ directory and populates dropdown lists."""

        # 1. SAM Models: find both encoder and decoder parts
        sam_matches = []
        for partial_name in ["mobile_sam", "sam_vit_b", "sam_vit_h", "sam_vit_l"]:
            if os.path.exists(MODEL_ROOT):
                for filename in os.listdir(MODEL_ROOT):
                    # Check for either part and extract the common name
                    if partial_name in filename and (".encoder.onnx" in filename or ".decoder.onnx" in filename):
                        cln = filename.replace(".encoder.onnx", "").replace(".decoder.onnx", "")
                        sam_matches.append(cln)

        if sam_matches:
            models = list(dict.fromkeys(sam_matches))  # Remove duplicates
            self.sam_combo.configure(values=models)

            # Restore last selected model
            if self.config["last_sam_model"] in models:
                self.sam_combo.set(self.config["last_sam_model"])
            else:
                self.sam_combo.current(0)
        else:
            self.sam_combo.set("No Models Found")

        # 2. Whole Image Models: find single ONNX files
        whole_matches = []
        for partial_name in ["rmbg", "isnet", "u2net", "BiRefNet"]:
            if os.path.exists(MODEL_ROOT):
                for filename in os.listdir(MODEL_ROOT):
                    if partial_name in filename and ".onnx" in filename:
                        whole_matches.append(filename.replace(".onnx", ""))

        if whole_matches:
            models = list(dict.fromkeys(whole_matches))
            self.whole_image_combo.configure(values=models)

            if self.config["last_whole_model"] in models:
                self.whole_image_combo.set(self.config["last_whole_model"])
            else:
                self.whole_image_combo.current(0)
        else:
            self.whole_image_combo.set("No Models Found")

    def set_keybindings(self):
        """Global keyboard and mouse bindings for navigation and actions."""
        # Canvas bindings (Zoom/Pan/Click)
        for canvas in [self.canvas, self.canvas2]:
            canvas.bind("<ButtonPress-1>", self.start_box)
            canvas.bind("<B1-Motion>", self.draw_box)
            canvas.bind("<ButtonRelease-1>", self.end_box)
            canvas.bind("<ButtonPress-3>", self.start_pan_mouse)
            canvas.bind("<B3-Motion>", self.pan_mouse)
            canvas.bind("<ButtonRelease-3>", self.end_pan_mouse)
            canvas.bind("<Button-4>", self.zoom)  # Linux scroll up
            canvas.bind("<Button-5>", self.zoom)  # Linux scroll down
            canvas.bind("<MouseWheel>", self.zoom)  # Windows/Mac scroll
            canvas.bind("<ButtonPress-2>", self.start_pan_mouse)  # Middle click pan
            canvas.bind("<B2-Motion>", self.pan_mouse)
            canvas.bind("<ButtonRelease-2>", self.end_pan_mouse)

        # Global Hotkeys
        self.root.bind("<c>", lambda e: self.clear_coord_overlay())  # Clear SAM points/box
        self.root.bind("<d>", lambda e: self.copy_entire_image())  # Copy Input to Output
        self.root.bind("<r>", lambda e: self.reset_all())  # Reset Everything
        self.root.bind("<a>", lambda e: self.add_to_working_image())  # Add to Mask
        self.root.bind("<s>", lambda e: self.subtract_from_working_image())  # Subtract from Mask
        self.root.bind("<w>", lambda e: self.clear_working_image())  # Clear Output Mask
        self.root.bind("<p>", self.paint_mode_toggle)  # Toggle Paintbrush
        self.root.bind("<v>", lambda e: self.clear_visible_area())  # Clear Visible Area

        # OS-agnostic Undo/Redo/Save
        is_mac = platform.system() == "Darwin"
        cmd = "Command" if is_mac else "Control"
        self.root.bind(f"<{cmd}-z>", lambda e: self.undo())
        self.root.bind(f"<{cmd}-y>", lambda e: self.redo())
        self.root.bind(f"<{cmd}-Shift-Z>", lambda e: self.redo())
        self.root.bind(f"<{cmd}-s>", lambda e: self.save_as_image())

        self.root.bind("<Alt-q>", lambda e: self.quick_save_automatic())  # Quick Export

        # Keyboard Panning
        self.root.bind("<Left>", self.pan_left_keyboard)
        self.root.bind("<Right>", self.pan_right_keyboard)
        self.root.bind("<Up>", self.pan_up_keyboard)
        self.root.bind("<Down>", self.pan_down_keyboard)

    # --- Panning/Zooming Logic ---

    def pan_left_keyboard(self, event):
        self.do_pan(-self.pan_step, 0)

    def pan_right_keyboard(self, event):
        self.do_pan(self.pan_step, 0)

    def pan_up_keyboard(self, event):
        self.do_pan(0, -self.pan_step)

    def pan_down_keyboard(self, event):
        self.do_pan(0, self.pan_step)

    def do_pan(self, dx, dy):
        """Updates the viewport coordinates and clamps them to image bounds."""
        self.view_x = max(0, min(self.view_x + dx, self.original_image.width - self.canvas_w / self.zoom_factor))
        self.view_y = max(0, min(self.view_y + dy, self.original_image.height - self.canvas_h / self.zoom_factor))
        self.update_input_image_preview(Image.NEAREST)
        self.schedule_preview_update()

    def start_pan_mouse(self, event):
        """Initializes panning state for mouse drag."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.click_start_x = event.x
        self.click_start_y = event.y
        self.panning = self.move_enabled  # Only pan if move mode is enabled

    def pan_mouse(self, event):
        """Calculates and applies viewport movement during mouse drag."""
        if self.panning:
            # Only allow panning if zoomed in past the minimum fit level
            if self.zoom_factor > (self.lowest_zoom_factor + 0.01):
                dx = event.x - self.pan_start_x
                dy = event.y - self.pan_start_y
                # Scale movement by zoom factor
                self.view_x -= dx / self.zoom_factor
                self.view_y -= dy / self.zoom_factor
                # Clamp to bounds
                self.view_x = max(0, min(self.view_x, self.original_image.width - self.canvas_w / self.zoom_factor))
                self.view_y = max(0, min(self.view_y, self.original_image.height - self.canvas_h / self.zoom_factor))
                self.pan_start_x = event.x
                self.pan_start_y = event.y
                self.clear_coord_overlay()  # Clear SAM/Box if panning
                self.model_output_mask = None
                self.update_input_image_preview(Image.NEAREST)  # Use fast nearest neighbor resampling

    def end_pan_mouse(self, event):
        was_panning = self.panning
        self.panning = False
        total_dx = abs(event.x - self.click_start_x)
        total_dy = abs(event.y - self.click_start_y)
        is_static_click = total_dx < 5 and total_dy < 5

        # Right click static -> Add negative point (SAM mode)
        if is_static_click and event.num == 3:
            if self.sam_active:
                self.generate_sam_mask(event)
        elif was_panning:
            # Re-render zoomed image using quality resampling (Box filter)
            self.update_input_image_preview(Image.BOX)

    def zoom(self, event):
        """Handles mouse wheel zooming relative to the cursor position."""
        self.pan_step = 30 / self.zoom_factor  # Scale keyboard pan step

        # Calculate cursor position in un-zoomed image coordinates
        mouse_x = self.view_x + event.x / self.zoom_factor
        mouse_y = self.view_y + event.y / self.zoom_factor
        new_zoom_factor = self.zoom_factor

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            check_zoom = new_zoom_factor * DEFAULT_ZOOM_FACTOR
            if check_zoom <= MAX_ZOOM_FACTOR:
                new_zoom_factor = check_zoom
                self.min_zoom = False
        elif event.num == 5 or event.delta < 0:
            new_zoom_factor = max(new_zoom_factor / DEFAULT_ZOOM_FACTOR, self.lowest_zoom_factor)

        self.zoom_factor = new_zoom_factor

        # Adjust view coordinates to keep mouse position constant after zoom
        self.view_x = max(0, min(mouse_x - event.x / self.zoom_factor, self.original_image.width - self.canvas_w / self.zoom_factor))
        self.view_y = max(0, min(mouse_y - event.y / self.zoom_factor, self.original_image.height - self.canvas_h / self.zoom_factor))

        self.update_zoom_label()

        # Fast update using NEAREST filter while zooming, defer quality update
        if self.min_zoom == False:
            self.update_input_image_preview(resampling_filter=Image.NEAREST)
            self.clear_coord_overlay()
        if self.lowest_zoom_factor == self.zoom_factor: self.min_zoom = True
        self.schedule_preview_update()

    def schedule_preview_update(self):
        """Debounces quality re-rendering after fast operations (zoom/pan)."""
        if self.after_id: self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(int(self.zoom_delay * 1000), self.update_preview_delayed)

    def update_preview_delayed(self):
        """Performs the quality re-render with the slower but better BOX filter."""
        self.update_input_image_preview(resampling_filter=Image.BOX)
        self.after_id = None

    def _calculate_preview_image(self, image, resampling_filter):
        """Crops and resizes an image based on current zoom/pan state."""
        view_width = self.canvas_w / self.zoom_factor
        view_height = self.canvas_h / self.zoom_factor

        # Crop box coordinates
        left = int(self.view_x)
        top = int(self.view_y)
        right = int(self.view_x + min(math.ceil(view_width), image.width))
        bottom = int(self.view_y + min(math.ceil(view_height), image.height))
        image_to_display = image.crop((left, top, right, bottom))

        # Scaled canvas size
        image_preview_w = int(image_to_display.width * self.zoom_factor)
        image_preview_h = int(image_to_display.height * self.zoom_factor)

        # Calculate padding to center the image on the canvas
        self.pad_x = max(0, (self.canvas_w - image_preview_w) // 2)
        self.pad_y = max(0, (self.canvas_h - image_preview_h) // 2)

        # Resize to canvas dimensions
        displayed_image = image_to_display.resize((image_preview_w, image_preview_h), resampling_filter)
        return displayed_image, image_to_display

    def update_input_image_preview(self, resampling_filter=Image.BOX):
        """Renders the Input Canvas (image + checkerboard + SAM overlay)."""
        displayed_image, self.orig_image_crop = self._calculate_preview_image(self.original_image, resampling_filter)

        if displayed_image.mode == "RGBA":
            # Composite with checkerboard for transparency preview
            image_preview_w, image_preview_h = displayed_image.size
            checkerboard = self.checkerboard.crop((0, 0, image_preview_w, image_preview_h))
            displayed_image = Image.alpha_composite(checkerboard, displayed_image)

        self.input_displayed = displayed_image
        self.tk_image = ImageTk.PhotoImage(self.input_displayed, master=self.root)

        # Draw on canvas
        self.canvas.delete("all")
        self.canvas.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.tk_image)

        # Re-draw SAM/Paint overlays if present
        self.generate_coloured_overlay()

        # Update Output canvas as well (they are linked in pan/zoom)
        self.update_output_image_preview(resampling_filter=resampling_filter)

    def update_output_image_preview(self, resampling_filter=Image.BOX, event=None):
        """Renders the Output Canvas (final composite + background + shadow)."""
        show_mask = self.show_mask_var.get() if hasattr(self, 'show_mask_var') else False

        # Determine which image to display
        if show_mask == False:
            displayed_image = self.working_image  # Actual cut-out
        else:
            displayed_image = self.working_mask.convert("RGBA")  # Alpha channel visualization

        # Apply background (color/blur/none)
        displayed_image = self.apply_background_color(displayed_image)

        # Crop and scale for current view
        displayed_image, _ = self._calculate_preview_image(displayed_image, resampling_filter)

        # Apply checkerboard *if* the mode is transparent (PNG export default)
        if self.bg_mode == "transparent":
            image_preview_w, image_preview_h = displayed_image.size
            checkerboard = self.checkerboard.crop((0, 0, image_preview_w, image_preview_h))
            displayed_image = Image.alpha_composite(checkerboard, displayed_image)

        self.output_displayed = displayed_image
        self.outputpreviewtk = ImageTk.PhotoImage(self.output_displayed, master=self.root)

        # Draw on canvas
        self.canvas2.delete("all")
        self.canvas2.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.outputpreviewtk)
        self.root.update_idletasks()

    # --- Mask Management / History ---

    def add_undo_step(self):
        """Saves a copy of the current mask to the undo stack."""
        self.undo_history_mask.append(self.working_mask.copy())
        if len(self.undo_history_mask) > UNDO_STEPS:
            self.undo_history_mask.pop(0)  # Keep history manageable
        self.redo_history_mask = []

    def clear_working_image(self):
        """Resets the output mask/image to completely transparent (empty)."""
        self.canvas2.delete(self.outputpreviewtk)
        self.working_image = Image.new(mode="RGBA", size=self.original_image.size)
        self.working_mask = Image.new(mode="L", size=self.original_image.size, color=0)
        self.add_undo_step()

        # Reset caches for blur/shadow
        if hasattr(self, "cached_blurred_shadow"): delattr(self, "cached_blurred_shadow")
        self.cached_blur_image = None

        self.update_output_image_preview()

    def reset_source_image(self):
        """Resets the input source to the default grey placeholder image."""
        self.original_image = Image.new("RGBA", (800, 600), (200, 200, 200, 255))
        self.image_exif = None
        self.image_paths = []
        self.current_image_index = 0
        self.file_count = ""
        self.root.title("Background Remover Pro")
        self.status_label.config(text='Status: Ready', fg=STATUS_NORMAL)

    def reset_all(self):
        """Performs a factory reset of the entire workspace."""
        # Clear UI elements
        self.coordinates = []
        self.labels = []
        self.clear_coord_overlay()
        self.canvas.delete(self.dots)
        if hasattr(self, 'overlay_item'): self.canvas.delete(self.overlay_item)

        # Reset caches
        if hasattr(self, "cached_blurred_shadow"): delattr(self, "cached_blurred_shadow")
        self.cached_blur_image = None

        # Reset AI state
        self.sam_active = False
        self.raw_model_mask = None
        self.raw_sam_logits = None

        # Reset images and history
        self.reset_source_image()
        self.canvas2.delete("all")
        self.working_image = Image.new(mode="RGBA", size=self.original_image.size)
        self.working_mask = Image.new(mode="L", size=self.original_image.size, color=0)
        self.undo_history_mask = [self.working_mask.copy()]
        self.redo_history_mask = []

        # Reset buttons/models
        if self.load_models_btn['text'] != 'Models Loaded':
            self.load_models_btn.configure(state="normal", text='Pre Load Models')

        self.whole_image_button.configure(state="normal", style="Magic.TButton", text='AUTO-DETECT SUBJECT')
        self.manual_sam_button.configure(state="normal", style="Accent.TButton", text='MANUAL DETECTION')

        # Final UI update
        self.setup_image_display()
        self.update_input_image_preview()
        self.status_label.config(text="Workpage has been reset.", fg="white")

    def undo(self):
        """Reverts the working mask to the previous state in the history."""
        if len(self.undo_history_mask) > 1:
            current_state = self.undo_history_mask.pop()
            self.redo_history_mask.append(current_state)
            self.working_mask = self.undo_history_mask[-1].copy()

            # Recalculate dependent elements
            if self.bg_mode == "blur": self.regenerate_smart_blur()
            self.add_drop_shadow()  # Triggers preview update
            self.status_label.config(text="Undo performed", fg="white")

    def redo(self):
        """Re-applies the next mask state from the redo stack."""
        if self.redo_history_mask:
            next_state = self.redo_history_mask.pop()
            self.undo_history_mask.append(next_state)
            self.working_mask = next_state.copy()

            # Recalculate dependent elements
            if self.bg_mode == "blur": self.regenerate_smart_blur()
            self.add_drop_shadow()
            self.status_label.config(text="Redo performed", fg="white")
        else:
            self.status_label.config(text="Nothing to Redo", fg="white")

    def copy_entire_image(self):
        """Sets the working mask to fully white (255), including the entire image."""
        self.working_mask = Image.new(mode="L", size=self.original_image.size, color=255)
        self.add_undo_step()
        self.add_drop_shadow()

    def cutout_working_image(self):
        """Applies the working mask to the original image to create the RGBA cut-out (working_image)."""
        empty = Image.new("RGBA", self.original_image.size, 0)
        mask_to_use = self.working_mask

        # Apply mask softening if enabled
        if hasattr(self, 'soften_mask_var') and self.soften_mask_var.get():
            radius = self.blur_radius_var.get()
            if radius > 0:
                mask_to_use = self.working_mask.filter(ImageFilter.GaussianBlur(radius=radius))

        self.working_image = Image.composite(self.original_image, empty, mask_to_use)

    def _apply_mask_modification(self, operation):
        """Generic method to Add (ImageChops.add) or Subtract (ImageChops.subtract) a preview mask."""
        if self.paint_mode.get():
            mask = self.generate_paint_mode_mask()
        else:
            mask = self.model_output_mask

        if mask == None:
            self.status_label.config(text="Warning: No mask generated to add/subtract. Run a model first.", fg="white")
            return

        # Calculate the paste box (the cropped area currently visible in the canvas)
        paste_box = (int(self.view_x), int(self.view_y),
                     int(self.view_x + self.orig_image_crop.width),
                     int(self.view_y + self.orig_image_crop.height))

        # Create a full-size mask to contain the cropped mask operation
        temp_fullsize_mask = Image.new("L", self.working_mask.size, 0)

        try:
            temp_fullsize_mask.paste(mask, paste_box)
            self.working_mask = operation(self.working_mask, temp_fullsize_mask)  # Apply operation
            self.add_undo_step()

            # Reset caches
            if hasattr(self, "cached_blurred_shadow"): delattr(self, "cached_blurred_shadow")
            if self.bg_mode == "blur": self.regenerate_smart_blur()

            self.add_drop_shadow()
            self.status_label.config(text="Mask applied.", fg="white")
        except Exception as e:
            self.status_label.config(text=f"Error applying mask: {e}", fg="white")

    def add_to_working_image(self):
        self._apply_mask_modification(ImageChops.add)

    def subtract_from_working_image(self):
        self._apply_mask_modification(ImageChops.subtract)

    def clear_visible_area(self):
        """Subtracts a mask of the entire currently visible area. Useful for bulk cleaning."""
        # Temporarily create a mask of the visible area
        mask_old = self.model_output_mask.copy() if self.model_output_mask else None
        self.model_output_mask = Image.new("L", self.orig_image_crop.size, 255)  # White rectangle
        self.subtract_from_working_image()
        self.model_output_mask = mask_old  # Restore preview mask

    def draw_dot(self, x, y, col):
        """Draws the SAM input points (green/red dots) on the canvas."""
        fill = COLORS["add"] if col == 1 else COLORS["remove"]
        radius = 4
        self.dots.append(self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline='white'))
        for i in self.dots: self.canvas.tag_raise(i)

    def clear_coord_overlay(self):
        """Clears all SAM points, bounding boxes, and paint lines."""
        self.coordinates = []
        self.labels = []
        for i in self.dots: self.canvas.delete(i)
        for i in self.lines_id: self.canvas.delete(i)
        self.lines = []
        for i in self.lines_id2: self.canvas2.delete(i)
        self.lines = []
        if hasattr(self, 'overlay_item'): self.canvas.delete(self.overlay_item)
        self.model_output_mask = Image.new("L", self.orig_image_crop.size, 0)  # Clear preview mask

    def trigger_inactive_feedback(self, event=None, x=None, y=None, size=8, target_canvas=None):
        """Shows visual feedback (Red X) and flashes the SAM button when an interactive action is invalid."""
        canvas_to_draw = None
        cx, cy = 0, 0

        # Get coordinates and target canvas
        if event:
            cx, cy = event.x, event.y
            canvas_to_draw = event.widget
        elif x is not None and y is not None and target_canvas is not None:
            cx, cy = x, y
            canvas_to_draw = target_canvas

        if canvas_to_draw:
            l1 = canvas_to_draw.create_line(cx - size, cy - size, cx + size, cy + size, fill="#FF0000", width=3, tags="error_x")
            l2 = canvas_to_draw.create_line(cx - size, cy + size, cx + size, cy - size, fill="#FF0000", width=3, tags="error_x")

        self.root.after(1000, lambda: canvas_to_draw.delete(l1, l2) if hasattr(canvas_to_draw, 'delete') else None)

        # Flash the "Manual Detection" button (with cooldown)
        now = timer()
        if now - self.last_flash_time > 4.0:
            self.last_flash_time = now
            self.flash_step = 0
            self.animate_manual_button_flash()

    def animate_manual_button_flash(self):
        """Flashes the MANUAL DETECTION button (3 times)."""
        if self.flash_step >= 6:
            self.manual_sam_button.configure(style="Accent.TButton")
            return

        if self.flash_step % 2 == 0:
            self.manual_sam_button.configure(style="Flash.TButton")
        else:
            self.manual_sam_button.configure(style="Accent.TButton")

        self.flash_step += 1
        self.root.after(150, self.animate_manual_button_flash)

    def start_box(self, event):
        """Initializes SAM box drawing or ignores if not in box/sam mode."""
        self.active_box_canvas = event.widget

        # If neither mode is active, provide feedback
        if not self.sam_active and not self.area_enabled:
            self.trigger_inactive_feedback(event=event)

        self.box_start_x = event.x
        self.box_start_y = event.y
        self.box_rectangle = None
        self.box_rectangle2 = None

    def draw_box(self, event):
        """Draws the selection box in both canvases during mouse drag."""
        if not self.area_enabled: return  # Only draw if in Box Mode

        dx = abs(event.x - self.box_start_x)
        dy = abs(event.y - self.box_start_y)

        # Cleanup previous rectangles
        if self.box_rectangle: self.canvas.delete(self.box_rectangle)
        if self.box_rectangle2: self.canvas2.delete(self.box_rectangle2)

        # Don't draw if too small (avoids accidental dot placement)
        if dx < MIN_RECT_SIZE and dy < MIN_RECT_SIZE:
            self.box_rectangle = None
            return

        # Draw new rectangles
        self.box_rectangle = self.canvas.create_rectangle(self.box_start_x, self.box_start_y, event.x, event.y, outline=COLORS["accent"], width=2)
        self.box_rectangle2 = self.canvas2.create_rectangle(self.box_start_x, self.box_start_y, event.x, event.y, outline=COLORS["accent"], width=2)

    def end_box(self, event):
        """Finalizes box selection. Runs SAM if active, otherwise gives feedback."""
        if hasattr(self, 'box_rectangle') and self.box_rectangle:
            # Box was drawn
            rect_coords = self.canvas.coords(self.box_rectangle)

            center_x = (rect_coords[0] + rect_coords[2]) / 2
            center_y = (rect_coords[1] + rect_coords[3]) / 2

            self.canvas.delete(self.box_rectangle)
            self.canvas2.delete(self.box_rectangle2)

            # Scale coordinates from canvas pixels to original image pixels (relative to the current view)
            scaled_coords = [(rect_coords[0] - self.pad_x) / self.zoom_factor,
                             (rect_coords[1] - self.pad_y) / self.zoom_factor,
                             (rect_coords[2] - self.pad_x) / self.zoom_factor,
                             (rect_coords[3] - self.pad_y) / self.zoom_factor]

            if self.sam_active:
                self.box_event(scaled_coords)  # Run SAM with box input
            else:
                target = getattr(self, 'active_box_canvas', self.canvas)
                self.trigger_inactive_feedback(x=center_x, y=center_y, size=20, target_canvas=target)
        else:
            # Single Click / Too Small Drag
            if self.sam_active:
                self.generate_sam_mask(event)  # Run SAM with point input
            else:
                self.trigger_inactive_feedback(event=event)

    def box_event(self, scaled_coords):
        """Handles running SAM with the bounding box input."""
        self._initialise_sam_model()
        self.clear_coord_overlay()

        # Convert scaled coordinates (relative to view) to global image coordinates
        global_x1 = self.view_x + scaled_coords[0]
        global_y1 = self.view_y + scaled_coords[1]
        global_x2 = self.view_x + scaled_coords[2]
        global_y2 = self.view_y + scaled_coords[3]

        # SAM box input is [x1, y1, x2, y2]
        self.coordinates = [[global_x1, global_y1], [global_x2, global_y2]]
        self.labels = [2, 3]  # SAM label for a bounding box

        self.raw_sam_logits = self.sam_calculate_mask(self.original_image, self.sam_encoder, self.sam_decoder, self.coordinates, self.labels)
        self.raw_model_mask = None
        self.on_unified_slider_change(self.unified_var.get())  # Update preview mask

        self.coordinates = []
        self.labels = []

    def activate_sam_mode(self):
        """Starts SAM mode: loads model, calculates embedding, sets interactive state."""
        self.raw_model_mask = None
        self.model_output_mask = None
        self.sam_active = True

        # Reset UI elements
        if hasattr(self, 'custom_slider'):
            self.custom_slider.set_value(50.0)
        else:
            self.unified_var.set(50.0)
        self.clear_coord_overlay()

        self.show_loading("Running SAM")  # Show overlay

        def heavy_task():
            self._initialise_sam_model_headless()
            if not hasattr(self, "encoder_output"):
                self.calculate_sam_embedding_headless()  # Calculate heavy embedding once
            return "Ready"

        def on_done(res):
            self.status_label.config(text="SAM Ready. Click on the image to add points.", fg="white")

        def on_err(e):
            self.sam_active = False
            self.hide_loading()
            messagebox.showerror("SAM Error", str(e))

        self.start_threaded_task(heavy_task, on_done, on_err)

    def run_whole_image_model(self, model_name):
        """Runs the selected whole-image background removal model in a thread."""
        if self.paint_mode.get(): return

        # Reset SAM state
        self.sam_active = False
        self.clear_coord_overlay()
        self.raw_sam_logits = None
        self.coordinates = []
        self.labels = []

        # Reset threshold slider to default
        self.unified_var.set(50.0)
        self.slider_val_label.config(text="50%")

        if model_name is None:
            model_name = self.whole_image_combo.get()
            if model_name == "No Models Found":
                messagebox.showerror("Error", "No whole image models found.")
                return

        # Determine target input size based on model type (optimization)
        target_size = 320 if "u2net" in model_name.lower() else 1024

        self.show_loading(f"Running {model_name}")
        self.whole_image_button.configure(state="disabled")

        def heavy_lifting():
            session = self.thread_safe_load_model(model_name)
            mask = self.generate_whole_image_model_mask(self.orig_image_crop, session, target_size)
            return mask

        def on_complete(result_mask):
            self.whole_image_button.configure(style="Magic.TButton", state="normal")
            self.raw_model_mask = result_mask
            self.on_unified_slider_change(self.unified_var.get())  # Apply initial threshold
            if self._magic_hovering: self.animate_magic_button()
            self.status_label.config(text=f"Inference Complete", fg="white")

        def on_fail(err):
            self.whole_image_button.configure(style="Magic.TButton", state="normal")
            self.hide_loading()
            self.status_label.config(text=f"Error running model: {err}", fg="white")

        self.start_threaded_task(heavy_lifting, on_complete, on_fail)

    def on_unified_slider_change(self, val):
        """Maps the 0-100 slider value to the appropriate model threshold range."""
        val = float(val)
        self.slider_val_label.config(text=f"{int(val)}%")
        t = 1.0 - (val / 100.0)  # Map 0-100 slider to 1.0-0.0 range

        if self.sam_active:
            # SAM threshold range: low for aggressive cutout, high for loose cutout
            min_sam = -3.0
            max_sam = 2.0
            sam_val = min_sam + (t * (max_sam - min_sam))
            self.update_sam_threshold(sam_val)
        else:
            # Whole Image threshold range (0-255 grayscale)
            min_thresh = 10
            max_thresh = 250
            whole_val = int(min_thresh + (t * (max_thresh - min_thresh)))
            self.update_mask_threshold(whole_val)

    def update_mask_threshold(self, threshold):
        """Applies binary thresholding to a whole-image model's raw mask."""
        if self.raw_model_mask is None: return
        # Create a new binary mask from the raw grayscale output
        self.model_output_mask = self.raw_model_mask.point(lambda p: 255 if p > threshold else 0)
        self.generate_coloured_overlay()

    def update_sam_threshold(self, threshold):
        """Applies thresholding to SAM's raw logits output."""
        if self.raw_sam_logits is None: return

        masks = self.raw_sam_logits
        mask_binary = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
        current_mask = masks[0, 0, :, :]

        # Apply threshold to logits
        mask_binary[current_mask > threshold] = [255, 255, 255]
        full_mask = Image.fromarray(mask_binary).convert("L")

        # Crop the full mask to the current visible viewport size
        crop_box = (int(self.view_x), int(self.view_y),
                    int(self.view_x + self.orig_image_crop.width),
                    int(self.view_y + self.orig_image_crop.height))

        try:
            self.model_output_mask = full_mask.crop(crop_box)
        except Exception:
            # Fallback resize if the crop fails due to dimension mismatch
            self.model_output_mask = full_mask.resize(self.orig_image_crop.size)

        self.generate_coloured_overlay()

    def generate_whole_image_model_mask(self, image, session, target_size=1024):
        """Runs the ONNX inference for whole-image models (RMBG, U2Net, etc)."""
        input_image = image.convert("RGB").resize((target_size, target_size), Image.BICUBIC)

        # Determine normalization based on model type
        model_path = os.path.basename(session._model_path)
        if "isnet" in model_path or "rmbg1_4" in model_path:
            std = (1.0, 1.0, 1.0)
            mean = (0.5, 0.5, 0.5)
        else:
            # Standard ImageNet normalization (for U2Net, BiRefNet)
            std = (0.229, 0.224, 0.225)
            mean = (0.485, 0.456, 0.406)

        # Preprocessing: normalize and reshape for ONNX
        im_ary = np.array(input_image)
        im_ary = im_ary / np.max(im_ary)
        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(tmpImg, 0).astype(np.float32)  # Add batch dimension

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # ONNX Inference
        result = session.run([output_name], {input_name: input_image})
        mask = result[0]

        # Post-processing: sigmoid/scaling and resizing
        if "BiRefNet" in model_path:
            # BiRefNet requires a manual sigmoid function
            def sigmoid(mat):
                return 1 / (1 + np.exp(-mat))

            pred = sigmoid(result[0][:, 0, :, :])
            ma, mi = np.max(pred), np.min(pred)
            pred = (pred - mi) / (ma - mi)  # Normalize to 0-1
            mask = Image.fromarray((np.squeeze(pred) * 255).astype("uint8")).resize(image.size, Image.Resampling.LANCZOS)
        else:
            # Default scaling
            mask = mask.squeeze()
            mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)

        return mask.convert("L")

    def generate_coloured_overlay(self):
        """Draws the transparent blue overlay on the input image canvas to show the mask preview."""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            self.canvas.delete(self.overlay_item)

        if self.model_output_mask is None or not hasattr(self, 'orig_image_crop'): return

        try:
            # Create a blue image from the original's shape
            self.overlay = ImageOps.colorize(self.orig_image_crop.convert("L"), black=COLORS["accent"], white="white")
            # Apply the current mask as the alpha channel
            self.overlay.putalpha(self.model_output_mask)

            # Scale the overlay to the canvas size
            image_preview_w = int(self.orig_image_crop.width * self.zoom_factor)
            image_preview_h = int(self.orig_image_crop.height * self.zoom_factor)
            self.scaled_overlay = self.overlay.resize((image_preview_w, image_preview_h), Image.NEAREST)
            self.tk_overlay = ImageTk.PhotoImage(self.scaled_overlay, master=self.root)

            # Draw on canvas
            self.overlay_item = self.canvas.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.tk_overlay)
            self.canvas.tag_raise(self.overlay_item)  # Ensure overlay is on top of the image

            # Ensure dots are on top of the overlay
            for dot in self.dots: self.canvas.tag_raise(dot)

        except Exception as e:
            print(f"Overlay Error: {e}")

    def generate_paint_mode_mask(self):
        """Converts user drawn line coordinates into a binary mask image for the current viewport."""
        img = Image.new('L', (self.orig_image_crop.width, self.orig_image_crop.height), color='black')  # Start with black (transparent)
        draw = ImageDraw.Draw(img)

        # Iterate over all stored line segments
        for x1, y1, x2, y2, size in self.lines:
            scaled_size = size / self.zoom_factor
            draw.line((x1, y1, x2, y2), fill='white', width=int(scaled_size))
            # Draw circles at endpoints for round brush tips
            r = scaled_size / 2
            draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill='white')
            draw.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), fill='white')

        return img  # Returns mask for the current view area

    def update_brush_cursor(self, event):
        """Draws a circle cursor showing the brush size on both canvases."""
        size = self.brush_size_var.get()
        r = size / 2
        self.canvas.delete("brush_cursor")
        self.canvas2.delete("brush_cursor")
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, outline="white", width=1, tag="brush_cursor")
        self.canvas2.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, outline="white", width=1, tag="brush_cursor")

    def paint_mode_toggle(self, event=None):
        """Toggles manual paint mode, updating UI state and bindings."""
        if event:
            self.paint_mode.set(not self.paint_mode.get())
            self.update_button_visual(self.btn_paint, self.paint_mode)

        if self.paint_mode.get():
            self.brush_options_frame.pack(fill="x", after=self.btn_paint)
            self.clear_coord_overlay()

            # Set drawing bindings
            for canvas in [self.canvas, self.canvas2]:
                canvas.config(cursor="crosshair")
                canvas.bind("<ButtonPress-1>", self.paint_draw_point)
                canvas.bind("<B1-Motion>", self.paint_draw_line)
                canvas.bind("<ButtonRelease-1>", self.paint_reset_coords)
                canvas.bind("<Motion>", self.update_brush_cursor)
        else:
            self.brush_options_frame.pack_forget()
            self.canvas.delete("brush_cursor")
            self.canvas2.delete("brush_cursor")
            self.clear_coord_overlay()

            # Restore default bindings
            for canvas in [self.canvas, self.canvas2]:
                canvas.config(cursor="")
                canvas.unbind("<Motion>")
            self.set_keybindings()  # Rebind standard navigation

        self.refresh_sidebar_scroll()

    def paint_draw_point(self, event):
        """Initializes a paint line segment."""
        current_size = self.brush_size_var.get()
        # Draw dotted line preview on the canvas (visual feedback only)
        self.lines_id.append(self.canvas.create_line(event.x, event.y, event.x, event.y, width=current_size, capstyle=tk.ROUND, fill="red", stipple="gray50"))
        self.lines_id2.append(self.canvas2.create_line(event.x, event.y, event.x, event.y, width=current_size, capstyle=tk.ROUND, fill="red", stipple="gray50"))

        # Store unscaled image coordinates for mask generation
        unscaled_x = (event.x - self.pad_x) / self.zoom_factor
        unscaled_y = (event.y - self.pad_y) / self.zoom_factor
        self.lines.append((unscaled_x, unscaled_y, unscaled_x, unscaled_y, current_size))

        self.last_x, self.last_y = event.x, event.y

    def paint_draw_line(self, event):
        """Continuously extends the paint line segment during drag."""
        current_size = self.brush_size_var.get()
        self.update_brush_cursor(event)

        if self.last_x and self.last_y:
            # Draw dotted line preview on the canvas
            self.lines_id.append(self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=current_size, capstyle=tk.ROUND, fill="red", stipple="gray50"))
            self.lines_id2.append(self.canvas2.create_line(self.last_x, self.last_y, event.x, event.y, width=current_size, capstyle=tk.ROUND, fill="red", stipple="gray50"))

            # Store line coordinates (unscaled)
            unscaled_last_x = (self.last_x - self.pad_x) / self.zoom_factor
            unscaled_last_y = (self.last_y - self.pad_y) / self.zoom_factor
            unscaled_curr_x = (event.x - self.pad_x) / self.zoom_factor
            unscaled_curr_y = (event.y - self.pad_y) / self.zoom_factor
            self.lines.append((unscaled_last_x, unscaled_last_y, unscaled_curr_x, unscaled_curr_y, current_size))

        self.last_x, self.last_y = event.x, event.y

    def paint_reset_coords(self, event):
        """Resets the last position after mouse button release."""
        self.last_x, self.last_y = 0, 0

    def _initialise_sam_model(self):
        # Wrapper for thread-safe model init
        self._initialise_sam_model_headless()

    def generate_sam_mask(self, event):
        """Handles a single point click in SAM mode (left=positive, right=negative)."""
        if not self.sam_active: return

        self._initialise_sam_model_headless()

        # Calculate unscaled coordinates in the original image space
        x = self.view_x + (event.x - self.pad_x) / self.zoom_factor
        y = self.view_y + (event.y - self.pad_y) / self.zoom_factor

        self.coordinates.append([x, y])
        label = 1 if event.num == 1 else 0  # 1=positive (left-click), 0=negative (right-click)
        self.labels.append(label)

        self.draw_dot(event.x, event.y, event.num)
        self.root.update()

        # Run SAM inference
        self.raw_sam_logits = self.sam_calculate_mask(self.original_image, self.sam_encoder, self.sam_decoder, self.coordinates, self.labels)
        self.raw_model_mask = None
        self.on_unified_slider_change(self.unified_var.get())  # Update preview mask

    def sam_calculate_mask(self, img, sam_encoder, sam_decoder, coordinates, labels, ):
        """Performs SAM decoder inference using the cached image embedding."""
        target_size = 1024
        input_size = (684, 1024)
        img = img.convert("RGB")
        cv_image = np.array(img)
        original_size = cv_image.shape[:2]

        # SAM's fixed scale transformation parameters
        scale_x = input_size[1] / cv_image.shape[1]
        scale_y = input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

        # Force embedding calculation if missing
        if not hasattr(self, "encoder_output"): self.calculate_sam_embedding_headless()

        image_embedding = self.encoder_output[0]

        # Prepare point inputs for ONNX
        input_points = np.array(coordinates)
        input_labels = np.array(labels)

        # Pad points/labels for SAM's input format (requires an extra negative point)
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

        # Scale coordinates according to SAM's internal transformation
        onnx_coord = self.apply_coords(onnx_coord, input_size, target_size).astype(np.float32)
        onnx_coord = np.concatenate([onnx_coord, np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32)], axis=2)
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        # Decoder required inputs
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(input_size, dtype=np.float32),
        }

        # Run decoder
        masks, _, _ = sam_decoder.run(None, decoder_inputs)

        # Inverse transform masks back to original image size
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        masks = self.transform_masks(masks, original_size, inv_transform_matrix)

        return masks  # Raw logits output

    # --- SAM Helper Functions ---

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        # Determines the scale factor for SAM's fixed input size
        scale = long_side_length * 1.0 / max(oldh, oldw)
        return (int(oldh * scale + 0.5), int(oldw * scale + 0.5))

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        # Scales the input coordinates (points/boxes)
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def transform_masks(self, masks, original_size, transform_matrix):
        # Applies the inverse affine transformation to masks
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(mask, transform_matrix[:2], (original_size[1], original_size[0]), flags=cv2.INTER_LINEAR)
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    # --- File/Export Actions ---

    def set_output_folder(self):
        """Opens a dialog to select the quick export folder."""
        initial = self.config["output_folder"] if self.config["output_folder"] else os.getcwd()
        folder = askdirectory(title="Select Output Folder", initialdir=initial)
        if folder:
            self.config["output_folder"] = folder
            self.status_label.config(text=f"Output folder set to: {folder}")
            self.save_config()
            self.update_folder_marquee()  # Refresh marquee text

    def quick_save_automatic(self):
        """Saves the current result to the configured output folder with an automatic filename."""
        if not self.validate_export_config(event=None): return  # Check for JPG/Transp conflict

        self.add_drop_shadow()  # Finalize composite image
        workimg = self.working_image
        if self.bg_mode != "transparent":
            workimg = self.apply_background_color(workimg)

        # Determine base filename
        if self.image_paths:
            original_filename = os.path.basename(self.image_paths[self.current_image_index])
            name, ext = os.path.splitext(original_filename)
        else:
            name, ext = "clipboard_image", ".png"

        # Determine output directory
        out_dir = self.config["output_folder"] if self.config["output_folder"] else os.path.dirname(self.image_paths[self.current_image_index]) if self.image_paths else os.getcwd()

        file_type = self.export_format
        save_ext = f".{file_type}"

        # Generate unique filename (e.g., image_nobg.png, image_nobg_1.png)
        counter = 0
        while True:
            suffix = f"_nobg_{counter}" if counter > 0 else "_nobg"
            filename = f"{name}{suffix}{save_ext}"
            full_path = os.path.join(out_dir, filename)
            if not os.path.exists(full_path): break
            counter += 1

        save_params = {}
        if self.image_exif: save_params['exif'] = self.image_exif  # Preserve EXIF data

        try:
            if file_type == "jpg":
                # JPG save: must be RGB, uses quality setting
                workimg = workimg.convert("RGB")
                workimg.save(full_path, quality=self.config.get("save_file_quality", 90), **save_params)
            elif "webp" in file_type:
                # WEBP save: supports transparency, uses quality setting
                workimg.save(full_path, quality=self.config.get("save_file_quality", 90), **save_params)
            else:
                # PNG save: supports transparency, uses optimization
                workimg.save(full_path, optimize=True, **save_params)

            # Optional: save the raw mask as a separate PNG
            if self.config.get("save_mask", False):
                mask_path = os.path.join(out_dir, f"{name}{suffix}_mask.png")
                self.working_mask.save(mask_path, "PNG")

            self.status_label.config(text=f"Auto-saved: {filename}", fg="white")
            print(f"Quick Saved to {full_path}")

            # Visual feedback: flash output canvas green
            self.canvas2.config(bg=COLORS["add"])
            self.root.after(200, lambda: self.canvas2.config(bg="#101010"))

        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def save_as_image(self):
        """Opens the OS 'Save As' dialog for custom path and filename selection."""
        if not self.validate_export_config(event=None): return

        # Initial filename/extension setup
        if self.image_paths:
            initial_file = os.path.splitext(os.path.basename(self.image_paths[self.current_image_index]))[0] + "_nobg"
        else:
            initial_file = "clipboard_nobg"

        def_ext = f".{self.export_format}"

        # Prioritize selected format in file dialog list
        filetypes = []
        if self.export_format == "png":
            filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("WEBP", "*.webp")]
        elif self.export_format == "jpg":
            filetypes = [("JPEG", "*.jpg"), ("PNG", "*.png"), ("WEBP", "*.webp")]
        else:
            filetypes = [("WEBP", "*.webp"), ("PNG", "*.png"), ("JPEG", "*.jpg")]

        user_filename = asksaveasfilename(title="Save as",
                                          defaultextension=def_ext,
                                          filetypes=filetypes,
                                          initialdir=self.config["output_folder"] if self.config["output_folder"] else None,
                                          initialfile=initial_file)

        if not user_filename: return  # Canceled

        self.add_drop_shadow()  # Finalize composite image
        workimg = self.working_image
        if self.bg_mode != "transparent":
            workimg = self.apply_background_color(workimg)

        ext = os.path.splitext(user_filename)[1].lower()
        try:
            if ext in [".jpg", ".jpeg"]:
                # Handle forced JPG save: force white background if transparent
                if self.bg_mode == "transparent":
                    workimg = Image.alpha_composite(Image.new("RGBA", workimg.size, "white"), workimg)
                    self.status_label.config(text="Warning: Saved as JPG with White Background (No Transparency)", fg="white")

                workimg.convert("RGB").save(user_filename, quality=90)
            else:
                workimg.save(user_filename)

            self.status_label.config(text=f"Saved to {user_filename}", fg=STATUS_NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_background_color(self, img):
        """Composites the cut-out image onto the selected background (blur, color, or none)."""
        if self.bg_mode == "blur":
            if self.cached_blur_image is None: self.regenerate_smart_blur()  # Re-generate if cache missed

            bg = self.cached_blur_image.convert("RGBA")
            try:
                combined = Image.alpha_composite(bg, img)
                return combined
            except ValueError:
                # Resize and try again if dimension mismatch
                bg = bg.resize(img.size)
                return Image.alpha_composite(bg, img)

        elif self.bg_mode == "color":
            colored_image = Image.new("RGBA", img.size, self.bg_custom_color)
            colored_image = Image.alpha_composite(colored_image, img)
            return colored_image

        return img  # Transparent background (default)

    def on_closing(self):
        """Cleanup before application exit: child windows, timers, and config save."""
        # Cleanup color picker state
        if hasattr(self, 'color_picker_window') and self.color_picker_window is not None and self.color_picker_window.winfo_exists():
            self.config["picker_geometry"] = self.color_picker_window.geometry()
            self.color_picker_window.destroy()

        # Cancel marquee timer
        if hasattr(self, 'marquee_after_id') and self.marquee_after_id:
            self.root.after_cancel(self.marquee_after_id)

        self.save_config()
        self.root.destroy()

    def add_drop_shadow(self):
        """Applies a blurred, offset drop shadow behind the subject."""
        if not self.enable_shadow_var.get():
            self.cutout_working_image()  # Re-cutout without shadow
            self.update_output_image_preview()
            return False

        self.cutout_working_image()  # Ensure working_image is up to date

        # Get current shadow parameters
        shadow_opacity = self.shadow_opacity_slider.get()
        shadow_radius = int(self.shadow_radius_slider.get())
        shadow_x = int(self.shadow_x_slider.get())
        shadow_y = int(self.shadow_y_slider.get())

        alpha = self.working_mask
        original_size = alpha.size

        # Cache control: Check if radius changed before re-blurring (expensive)
        cached = getattr(self, 'cached_blurred_shadow', None)
        last_radius = getattr(self, '_last_shadow_radius', None)
        if last_radius != shadow_radius:
            cached = None
            self._last_shadow_radius = shadow_radius

        if cached is None:
            # Downsample for faster Gaussian Blur performance
            downsample_factor = 0.5
            new_size = (int(original_size[0] * downsample_factor), int(original_size[1] * downsample_factor))
            alpha_resized = alpha.resize(new_size, Image.NEAREST)

            # Perform Gaussian Blur on the downsampled mask
            blurred_alpha_resized = alpha_resized.filter(ImageFilter.GaussianBlur(radius=shadow_radius * downsample_factor))

            # Upsample the blurred mask back to original size
            cached = blurred_alpha_resized.resize(original_size, Image.NEAREST)
            self.cached_blurred_shadow = cached

        blurred_alpha = cached

        # Adjust blurred alpha by opacity multiplier
        shadow_opacity_alpha = blurred_alpha.point(lambda p: int(p * shadow_opacity))

        # Create shadow image and apply offset
        shadow_image = Image.new("RGBA", self.working_image.size, (0, 0, 0, 0))  # Fully transparent black
        shadow_image.putalpha(shadow_opacity_alpha)  # Set alpha to the blurred mask

        shadow_with_offset = Image.new("RGBA", self.working_image.size, (0, 0, 0, 0))
        shadow_with_offset.paste(shadow_image, (shadow_x, shadow_y), shadow_image)  # Paste with offset

        # Composite shadow under the main cut-out
        self.working_image = Image.alpha_composite(shadow_with_offset, self.working_image)
        self.update_output_image_preview()
        return True

    def on_alpha_channel_toggle(self):
        """Called when Alpha Channel button is clicked. Disables conflicting effects."""
        self.update_output_image_preview()  # Update to show/hide mask

        if self.show_mask_var.get():
            # If Alpha is ON, disable Shadow and Soften Edges

            # Disable Shadow
            if self.enable_shadow_var.get():
                self.enable_shadow_var.set(False)
                self.update_button_visual(self.btn_shadow, self.enable_shadow_var)
                self.shadow_options_frame.pack_forget()

            # Disable Soften
            if self.soften_mask_var.get():
                self.soften_mask_var.set(False)
                self.update_button_visual(self.btn_soften, self.soften_mask_var)
                self.soften_options_frame.pack_forget()

            self.refresh_sidebar_scroll()

    def toggle_shadow_options(self):
        """Toggles the drop shadow feature and options display."""

        # 1. Conflict Check
        if self.show_mask_var.get():
            # Revert the toggle action if conflict is detected
            self.enable_shadow_var.set(False)
            self.update_button_visual(self.btn_shadow, self.enable_shadow_var)
            self.trigger_alpha_conflict_warning()
            return

        # 2. Standard Logic
        if self.enable_shadow_var.get():
            self.shadow_options_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.shadow_options_frame.pack_forget()

        self.add_drop_shadow()  # Recalculate and update preview
        self.refresh_sidebar_scroll()

    def toggle_soften_options(self):
        """Toggles the mask softening feature and options display."""

        # 1. Conflict Check
        if self.show_mask_var.get():
            # Revert the toggle action
            self.soften_mask_var.set(False)
            self.update_button_visual(self.btn_soften, self.soften_mask_var)
            self.trigger_alpha_conflict_warning()
            return

        # 2. Standard Logic
        if self.soften_mask_var.get():
            self.soften_options_frame.pack(fill="x", before=self.btn_shadow)
        else:
            self.soften_options_frame.pack_forget()

        self.add_drop_shadow()  # Recalculate and update preview (softening is done inside add_drop_shadow/cutout_working_image)
        self.refresh_sidebar_scroll()

    def initialise_new_image(self):
        """Resets all working state variables for a newly loaded image."""
        self.cached_blur_image = None
        self.canvas2.delete("all")
        self.setup_image_display()
        self.update_input_image_preview()
        self.clear_coord_overlay()

        self.working_image = Image.new(mode="RGBA", size=self.original_image.size)
        self.working_mask = Image.new(mode="L", size=self.original_image.size, color=0)

        self.undo_history_mask = [self.working_mask.copy()]
        self.redo_history_mask = []

        self.sam_active = False
        self.raw_model_mask = None
        self.raw_sam_logits = None
        if hasattr(self, "encoder_output"): delattr(self, "encoder_output")  # Reset SAM embedding cache

    def show_help(self):
        """Opens the custom documentation window."""
        if hasattr(self, 'help_window') and self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.lift()
            return

        self.help_window = ModernHelpWindow(self.root)

    def start_magic_anim(self, event):
        """Starts the gradient animation on the AUTO-DETECT button."""
        if not self._magic_hovering:
            self._magic_hovering = True
            self.animate_magic_button()

    def stop_magic_anim(self, event):
        """Stops the gradient animation and resets the button color."""
        self._magic_hovering = False
        if self._magic_cycle_id:
            self.root.after_cancel(self._magic_cycle_id)
            self._magic_cycle_id = None
        self.style.configure("Magic.TButton", background="#6200EA", foreground="white")  # Reset color

    def animate_magic_button(self):
        """Cycles through colors to create a rainbow gradient effect."""
        if self._magic_hovering:
            # Interpolate between current and next color
            color_start = self._magic_colors[self._magic_idx]
            color_end = self._magic_colors[(self._magic_idx + 1) % len(self._magic_colors)]
            t = self._magic_step / self._magic_steps_total
            current_color = self._interpolate_color(color_start, color_end, t)

            self.style.configure("Magic.TButton", background=current_color, foreground="#101010")

            self._magic_step += 1
            if self._magic_step > self._magic_steps_total:
                self._magic_step = 0
                self._magic_idx = (self._magic_idx + 1) % len(self._magic_colors)

            self._magic_cycle_id = self.root.after(20, self.animate_magic_button)

    # --- FILE GALLERY LOGIC ---

    def _import_files_action(self):
        """Opens file dialog for multi-file selection."""
        filetypes = [("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff")]
        files = filedialog.askopenfilenames(title="Import Images", filetypes=filetypes)
        if files:
            self.process_import_paths(files)

    def _import_folder_action(self):
        """Opens directory dialog and imports all valid images."""
        folder = filedialog.askdirectory(title="Import Folder")
        if folder:
            valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
            # Filter for valid image files inside the folder
            files = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
            self.process_import_paths(files)

    def process_import_paths(self, path_list):
        """Orchestrates image import, running thumbnail creation in a background thread."""
        if not path_list: return

        # Show loading bar only for bulk import (4+ files)
        if len(path_list) >= 4:
            self.show_loading(f"Importing {len(path_list)} images...")

        existing_paths = set(f['path'] for f in self.gallery_files)

        # Define the heavy lifting task
        def task():
            return self._worker_import_thumbnails(path_list, existing_paths)

        self.start_threaded_task(task, self._finalize_import, error_callback=self._on_import_error)

    def _worker_import_thumbnails(self, path_list, existing_paths):
        """THREADED: Creates PIL thumbnails. Must AVOID all Tkinter/ImageTk calls."""
        THUMB_SIZE = (70, 70)
        BG_COLOR = (30, 30, 30, 255)

        processed_items = []

        for path in path_list:
            if not os.path.exists(path) or path in existing_paths: continue

            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)

                # Create letterbox/padding canvas
                thumb_base = Image.new('RGBA', THUMB_SIZE, BG_COLOR)

                # Resize image for thumbnail (Lanczos for quality)
                target_size = (THUMB_SIZE[0] - 2, THUMB_SIZE[1] - 2)
                img.thumbnail(target_size, Image.Resampling.LANCZOS)

                # Center on the thumbnail canvas
                offset_x = (THUMB_SIZE[0] - img.width) // 2
                offset_y = (THUMB_SIZE[1] - img.height) // 2
                thumb_base.paste(img, (offset_x, offset_y))

                # Store PIL object
                processed_items.append({
                    'path': path,
                    'name': os.path.basename(path),
                    'pil_thumb': thumb_base
                })
            except Exception as e:
                print(f"[ERROR] Import failed for {path}: {e}")

        return processed_items

    def _finalize_import(self, results):
        """MAIN THREAD: Converts PIL images to ImageTk and updates the gallery UI."""
        if not results:
            self.hide_loading()
            return

        for item in results:
            # Convert PIL image to ImageTk (Must be done on Main Thread)
            tk_thumb = ImageTk.PhotoImage(item['pil_thumb'])

            self.gallery_files.append({
                'path': item['path'],
                'name': item['name'],
                'thumb': tk_thumb
            })

        self.redraw_gallery()
        self.status_label.config(text=f"Added {len(results)} images to gallery.")
        self.hide_loading()

    def _on_import_error(self, error):
        """Handles errors from the import thread."""
        self.hide_loading()
        messagebox.showerror("Import Error", str(error))
        print(f"Thread Error during import: {error}")

    def redraw_gallery(self):
        """Re-renders the horizontal gallery strip on the canvas."""
        self.gallery_canvas.delete("all")

        # Layout constants
        ITEM_SIZE = 70
        PADDING = 6
        START_X = 4
        START_Y = 4

        for idx, item in enumerate(self.gallery_files):
            col = idx // 2  # Items stack vertically in pairs
            row = idx % 2

            x = START_X + col * (ITEM_SIZE + PADDING)
            y = START_Y + row * (ITEM_SIZE + PADDING)

            # 1. Outer Frame (Base border)
            self.gallery_canvas.create_rectangle(
                x, y, x + ITEM_SIZE, y + ITEM_SIZE,
                outline="#333333", width=1, fill="#151515",
                tags=(f"item_{idx}", "thumb_frame")
            )

            # 2. Image Content
            self.gallery_canvas.create_image(
                x + ITEM_SIZE // 2, y + ITEM_SIZE // 2, anchor="center",
                image=item['thumb'],
                tags=(f"item_{idx}", "thumb_img")
            )

            # 3. Selection Highlight
            is_selected = (idx == self.selected_gallery_index)
            outline_col = COLORS["accent"] if is_selected else "#444444"
            width = 2 if is_selected else 1

            self.gallery_canvas.create_rectangle(
                x + 1, y + 1, x + ITEM_SIZE - 1, y + ITEM_SIZE - 1,
                outline=outline_col, width=width,
                tags=(f"item_{idx}", "thumb_rect")
            )

        # Update horizontal scroll region
        total_cols = (len(self.gallery_files) + 1) // 2
        total_width = START_X + total_cols * (ITEM_SIZE + PADDING) + START_X
        self.gallery_canvas.configure(scrollregion=(0, 0, total_width, 160))

    def on_gallery_click(self, event):
        """Handles click to select/deselect a thumbnail."""
        x = self.gallery_canvas.canvasx(event.x)
        y = self.gallery_canvas.canvasy(event.y)

        clicked_index = None
        # Find the tag of the item clicked
        for item_id in self.gallery_canvas.find_overlapping(x, y, x + 1, y + 1):
            tags = self.gallery_canvas.gettags(item_id)
            for tag in tags:
                if tag.startswith("item_"):
                    clicked_index = int(tag.split("_")[1])
                    break
            if clicked_index is not None: break

        if clicked_index is not None:
            # Toggle selection
            self.selected_gallery_index = clicked_index if self.selected_gallery_index != clicked_index else None

        self.redraw_gallery()

    def on_gallery_hover_move(self, event):
        """Debounced tooltip logic: detects hover and schedules tooltip display."""
        if self.tooltip_after_id: self.root.after_cancel(self.tooltip_after_id)
        self.gallery_canvas.delete("tooltip")

        x_canvas = self.gallery_canvas.canvasx(event.x)
        y_canvas = self.gallery_canvas.canvasy(event.y)

        hover_index = None
        # Find item under cursor
        for i in self.gallery_canvas.find_overlapping(x_canvas, y_canvas, x_canvas + 1, y_canvas + 1):
            for tag in self.gallery_canvas.gettags(i):
                if tag.startswith("item_"):
                    hover_index = int(tag.split("_")[1])
                    break
            if hover_index is not None: break

        if hover_index is not None:
            # Schedule tooltip after 600ms dwell time
            self.tooltip_after_id = self.root.after(
                600,
                lambda: self.show_gallery_tooltip(hover_index, x_canvas, y_canvas)
            )

    def on_gallery_leave(self, event):
        """Cleans up tooltip on mouse exit."""
        if self.tooltip_after_id: self.root.after_cancel(self.tooltip_after_id)
        self.gallery_canvas.delete("tooltip")

    def show_gallery_tooltip(self, index, x, y):
        """Draws a custom tooltip for the hovered thumbnail."""
        if index >= len(self.gallery_files): return

        text = self.gallery_files[index]['name']

        # 1. Initial text placement (above mouse)
        text_y = y - 20
        text_id = self.gallery_canvas.create_text(
            x, text_y, text=text, fill="white",
            font=("Segoe UI", 8), anchor="s", tags="tooltip"
        )

        # 2. Calculate visible viewport limits
        view_left = self.gallery_canvas.canvasx(0)
        view_top = self.gallery_canvas.canvasy(0)
        view_width = self.gallery_canvas.winfo_width()
        view_right = view_left + view_width

        # 3. Calculate text bounding box
        bbox = self.gallery_canvas.bbox(text_id)
        dx = 0;
        dy = 0;
        padding = 6

        # 4. Clamping Logic (Keep tooltip inside visible viewport)

        # Horizontal clamping
        if bbox[2] > (view_right - padding):
            dx = (view_right - padding) - bbox[2]
        elif bbox[0] < (view_left + padding):
            dx = (view_left + padding) - bbox[0]

        # Vertical clamping (only upwards check needed)
        if bbox[1] < (view_top + padding):
            dy = (view_top + padding) - bbox[1]

        # 5. Apply movement and update bounding box
        if dx != 0 or dy != 0:
            self.gallery_canvas.move(text_id, dx, dy)
            bbox = self.gallery_canvas.bbox(text_id)

        # 6. Draw background rectangle
        bg_coords = (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2)
        bg_id = self.gallery_canvas.create_rectangle(
            bg_coords, fill="#2d2d30", outline=COLORS["border"], tags="tooltip"
        )

        # Ensure text is rendered on top of the background and thumbnails
        self.gallery_canvas.tag_raise(bg_id)
        self.gallery_canvas.tag_raise(text_id)

    def use_gallery_image(self):
        """Loads the selected image from the gallery into the main editor."""
        if self.selected_gallery_index is not None and self.selected_gallery_index < len(self.gallery_files):
            path = self.gallery_files[self.selected_gallery_index]['path']

            self.load_image_path(path)
            self.image_paths = [path]
            self.current_image_index = 0
            self.initialise_new_image()

            filename = os.path.basename(path)
            self.root.title(f"Background Remover - {filename}")
            self.status_label.config(text=f"Loaded from gallery: {filename}")

            # Deselect thumbnail
            self.selected_gallery_index = None
            self.redraw_gallery()

    def delete_gallery_image(self):
        """Deletes the selected image from the gallery list."""
        if self.selected_gallery_index is not None:
            del self.gallery_files[self.selected_gallery_index]
            self.selected_gallery_index = None
            self.redraw_gallery()

    def clean_gallery(self):
        """Clears all images from the gallery."""
        self.gallery_files = []
        self.selected_gallery_index = None
        self.redraw_gallery()


if __name__ == "__main__":
    files_to_process = sys.argv[1:]
    root = ROOT_CLASS()
    app = BackgroundRemoverGUI(root, files_to_process)
    root.mainloop()
