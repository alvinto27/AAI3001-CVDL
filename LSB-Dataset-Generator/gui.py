"""Native desktop interface; all image work stays in the shared generator."""
from __future__ import annotations

import bootstrap  # noqa: F401
import os
import queue
import threading
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from lsb_batch import Config, generate, read_config, write_config
from lsb_core import VERSION


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"LSB Dataset Generator · {VERSION}")
        self.geometry("1040x790")
        self.minsize(940, 620)
        self.configure(bg="#f3f5f8")
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.last_result: dict | None = None
        self.last_output: Path | None = None
        self.worker: threading.Thread | None = None
        self.closing = False
        self.controls: list[tk.Widget] = []
        self.vars = {
            "input_dir": tk.StringVar(), "output_dir": tk.StringVar(),
            "payload_rate": tk.StringVar(value="0.4"), "run_seed": tk.StringVar(value="42"),
            "payload_mode": tk.StringVar(value="fixed"),
            "payload_min": tk.StringVar(value="0.1"), "payload_max": tk.StringVar(value="0.9"),
            "recursive": tk.BooleanVar(value=True),
        }
        self.status = tk.StringVar(value="Ready to generate")
        self.detail = tk.StringVar(value="Choose your source PNG folder and an output location.")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f3f5f8")
        self.style.configure("Card.TFrame", background="#ffffff")
        self.style.configure("TLabel", background="#f3f5f8", foreground="#23324a", font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background="#ffffff")
        self.style.configure("Muted.TLabel", background="#ffffff", foreground="#556477", font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        self.style.configure("Section.TLabel", background="#ffffff", font=("Segoe UI", 12, "bold"))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        self.style.configure("Primary.TButton", background="#235bb9", foreground="white", font=("Segoe UI", 10, "bold"))
        self.style.map("Primary.TButton", background=[("active", "#17458e"), ("disabled", "#aab9d1")])
        self.style.configure("TEntry", padding=6, font=("Segoe UI", 10))
        self.style.configure("TRadiobutton", background="white", font=("Segoe UI", 10))
        self.style.configure("TCheckbutton", background="white", font=("Segoe UI", 10))
        self.style.configure("TProgressbar", background="#235bb9", troughcolor="#e7edf5", thickness=9)
        self.build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.after(70, self.poll)

    def button(self, parent: tk.Widget, text: str, command, **kwargs) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        self.controls.append(button)
        return button

    def entry(self, parent: tk.Widget, key: str, **kwargs) -> ttk.Entry:
        widget = ttk.Entry(parent, textvariable=self.vars[key], **kwargs)
        self.controls.append(widget)
        return widget

    def build(self) -> None:
        viewport = tk.Canvas(self, bg="#f3f5f8", highlightthickness=0)
        page_scroll = ttk.Scrollbar(self, orient="vertical", command=viewport.yview)
        viewport.configure(yscrollcommand=page_scroll.set)
        page_scroll.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)
        outer = ttk.Frame(viewport, padding=24)
        content_id = viewport.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", lambda event: viewport.configure(scrollregion=viewport.bbox("all")))
        def fit_page(event) -> None:
            viewport.itemconfigure(content_id, width=event.width)
            viewport.itemconfigure(content_id, height=max(event.height, outer.winfo_reqheight()))
        viewport.bind("<Configure>", fit_page)
        self.bind("<MouseWheel>", lambda event: viewport.yview_scroll(-int(event.delta / 120), "units") if event.widget is not self.log else None)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1, minsize=220)
        ttk.Label(outer, text="LSB Dataset Generator", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="Create clean / stego PNG pairs with exact modification masks and selected embedding locations.").grid(row=1, column=0, sticky="w", pady=(3, 18))

        body = ttk.Frame(outer)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        locations = ttk.Frame(body, style="Card.TFrame", padding=18)
        locations.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        locations.columnconfigure(0, weight=1)
        ttk.Label(locations, text="1   Choose folders", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(locations, text="Source images", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.entry(locations, "input_dir").grid(row=2, column=0, sticky="ew", pady=(5, 10))
        self.button(locations, "Browse…", self.pick_input).grid(row=2, column=1, padx=(8, 0))
        ttk.Label(locations, text="Run output folder", style="Card.TLabel").grid(row=3, column=0, sticky="w")
        self.entry(locations, "output_dir").grid(row=4, column=0, sticky="ew", pady=(5, 4))
        self.button(locations, "Choose…", self.pick_output).grid(row=4, column=1, padx=(8, 0))
        ttk.Label(locations, text="Choose a parent folder; a new run folder is suggested.", style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w")
        recursive = ttk.Checkbutton(locations, text="Include subfolders", variable=self.vars["recursive"])
        recursive.grid(row=6, column=0, sticky="w", pady=(14, 0))
        self.controls.append(recursive)

        settings = ttk.Frame(body, style="Card.TFrame", padding=18)
        settings.grid(row=0, column=1, sticky="nsew")
        settings.columnconfigure(0, weight=1)
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="2   Set embedding", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        for col, (value, label) in enumerate((("fixed", "Fixed rate"), ("range", "Random range"))):
            radio = ttk.Radiobutton(settings, text=label, value=value, variable=self.vars["payload_mode"], command=self.update_mode)
            radio.grid(row=1, column=col, sticky="w")
            self.controls.append(radio)
        self.rate_entry = self.entry(settings, "payload_rate", width=12)
        self.rate_entry.grid(row=2, column=0, sticky="ew", pady=(8, 4), padx=(0, 8))
        ttk.Label(settings, text="0.4 = 40% of RGB LSBs", style="Muted.TLabel").grid(row=2, column=1, sticky="w")
        range_frame = ttk.Frame(settings, style="Card.TFrame")
        range_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 9))
        ttk.Label(range_frame, text="Min", style="Muted.TLabel").pack(side="left")
        self.min_entry = self.entry(range_frame, "payload_min", width=8)
        self.min_entry.pack(side="left", padx=(6, 14))
        ttk.Label(range_frame, text="Max", style="Muted.TLabel").pack(side="left")
        self.max_entry = self.entry(range_frame, "payload_max", width=8)
        self.max_entry.pack(side="left", padx=6)
        ttk.Label(settings, text="Run seed", style="Card.TLabel").grid(row=4, column=0, sticky="w")
        self.entry(settings, "run_seed", width=15).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 10))

        actions = ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="ew", pady=14)
        self.generate_button = self.button(actions, "Generate dataset", self.start_generation, style="Primary.TButton")
        self.generate_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.button(actions, "Load config", self.load_settings).pack(side="right")
        self.button(actions, "Save config", self.save_settings).pack(side="right", padx=8)

        results = ttk.Frame(outer, style="Card.TFrame", padding=18)
        results.grid(row=4, column=0, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(3, weight=1)
        ttk.Label(results, textvariable=self.status, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(results, textvariable=self.detail, style="Muted.TLabel", wraplength=900).grid(row=1, column=0, sticky="w", pady=(4, 9))
        self.progress = ttk.Progressbar(results, mode="determinate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        log_frame = ttk.Frame(results, style="Card.TFrame")
        log_frame.grid(row=3, column=0, sticky="nsew")
        self.log = tk.Text(log_frame, height=5, wrap="word", bg="#f7f9fc", fg="#33435b", font=("Consolas", 9), relief="flat", padx=10, pady=8, state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        footer = ttk.Frame(results, style="Card.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.open_button = ttk.Button(footer, text="Open output folder", command=self.open_output, state="disabled")
        self.open_button.pack(side="left")
        ttk.Label(outer, text=f"v{VERSION}  •  Static 8-bit RGB PNG only  •  All RGB channels eligible  •  Original dimensions preserved", foreground="#556477", font=("Segoe UI", 9)).grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.update_mode()

    def update_mode(self) -> None:
        fixed = self.vars["payload_mode"].get() == "fixed"
        self.rate_entry.configure(state="normal" if fixed and not self.busy else "disabled")
        for widget in (self.min_entry, self.max_entry):
            widget.configure(state="normal" if not fixed and not self.busy else "disabled")

    def get_config(self) -> Config:
        values = {key: var.get() for key, var in self.vars.items()}
        for key in ("payload_rate", "payload_min", "payload_max"):
            values[key] = float(values[key])
        values["run_seed"] = int(values["run_seed"])
        config = Config(**values)
        config.validate()
        return config

    def apply_config(self, config: Config) -> None:
        for key, value in asdict(config).items():
            self.vars[key].set(value)
        self.update_mode()

    def pick_input(self) -> None:
        path = filedialog.askdirectory(title="Choose source image folder", parent=self)
        if path:
            self.vars["input_dir"].set(path)

    def pick_output(self) -> None:
        path = filedialog.askdirectory(title="Choose parent folder for a new run", parent=self)
        if path:
            name = datetime.now().strftime("lsb-run-%Y%m%d-%H%M%S-%f")
            self.vars["output_dir"].set(str(Path(path) / name))

    def save_settings(self) -> None:
        try:
            config = self.get_config()
            path = filedialog.asksaveasfilename(title="Save configuration", defaultextension=".yaml", filetypes=[("YAML", "*.yaml")], parent=self)
            if path:
                write_config(Path(path), config)
        except Exception as exc:
            self.show_error(exc)

    def load_settings(self) -> None:
        path = filedialog.askopenfilename(title="Load configuration", filetypes=[("YAML", "*.yaml *.yml")], parent=self)
        if path:
            try:
                self.apply_config(read_config(Path(path)))
            except Exception as exc:
                self.show_error(exc)

    def append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        # Bound UI memory; the complete generation log remains on disk.
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "101.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        for widget in self.controls:
            widget.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self.open_button.configure(state="normal" if self.last_output and self.last_output.exists() and not busy else "disabled")
        self.update_mode()

    def begin(self, operation, label: str) -> None:
        if self.busy:
            return
        self.last_result = None
        self.cancel_event.clear()
        self.set_busy(True)
        self.status.set(label)
        self.detail.set("Preparing…")
        self.progress.configure(value=0, maximum=1)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        def work() -> None:
            try:
                result = operation()
                self.events.put({"kind": "done", "result": result})
            except Exception as exc:
                self.events.put({"kind": "error", "message": str(exc)})
        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()

    def start_generation(self) -> None:
        try:
            config = self.get_config()
        except Exception as exc:
            self.show_error(exc)
            return
        self.last_output = Path(config.output_dir)
        self.begin(lambda: generate(config, self.events.put, self.cancel_event), "Generating dataset")

    def cancel(self) -> None:
        self.cancel_event.set()
        self.status.set("Stopping after the current image…")
        self.cancel_button.configure(state="disabled")

    def poll(self) -> None:
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event["kind"] == "progress":
                self.progress.configure(maximum=max(event["total"], 1), value=event["current"])
                self.detail.set(event["message"])
                self.append_log(event["message"])
            elif event["kind"] == "done":
                self.last_result = result = event["result"]
                self.set_busy(False)
                status = result["status"]
                self.status.set({"completed": "Dataset ready", "cancelled": "Cancelled — partial results retained", "failed": "Run failed — review the log"}.get(status, status))
                self.detail.set(f"{result['total_generated']} generated   •   {result['total_skipped']} skipped   •   of {result['total_discovered']} discovered")
                self.append_log(self.status.get() + ". " + self.detail.get())
            elif event["kind"] == "error":
                self.set_busy(False)
                self.status.set("Could not complete the operation")
                self.detail.set(event["message"])
                self.append_log(event["message"])
                if not self.closing:
                    self.show_error(event["message"])
        if self.closing and not self.busy:
            self.destroy()
            return
        self.poll_id = self.after(70, self.poll)

    def destroy(self) -> None:
        if getattr(self, "poll_id", None):
            self.after_cancel(self.poll_id)
            self.poll_id = None
        super().destroy()

    def show_error(self, error) -> None:
        messagebox.showerror("LSB Dataset Generator", str(error), parent=self)

    def open_output(self) -> None:
        if self.last_output and self.last_output.is_dir():
            try:
                os.startfile(str(self.last_output.resolve()))
            except (AttributeError, OSError) as exc:
                self.show_error(f"Output folder: {self.last_output}\n{exc}")

    def close(self) -> None:
        if self.busy:
            if not messagebox.askyesno("Stop and close?", "Stop after the current image, preserve completed results, then close?", parent=self):
                return
            self.closing = True
            self.cancel()
        else:
            self.destroy()


if __name__ == "__main__":
    App().mainloop()
