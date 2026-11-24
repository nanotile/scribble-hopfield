# -*- coding: utf-8 -*-
"""
GPU-Integrated AI-Enhanced Scribble Plotter - Gradio Web Interface
Module: scribble_plotter_web.py
Version: 4.0.0
Iteration: v12
Developer: Kent Benson
UV Environment: ./venv/bin/python scribble_plotter_web.py

Gradio web UI wrapping the existing processing pipeline.
Deploy behind Cloudflare tunnel: cf-tunnel add scribble <port>
"""

import matplotlib
matplotlib.use('Agg')

import os
import io
import shutil
import socket
import zipfile
import contextlib
import time
from pathlib import Path
from datetime import datetime, timedelta

import gradio as gr

from scribble_plotter_local import (
    setup_complete_system,
    CompleteConfiguration,
    CompleteProcessingSystem,
    hopfield_demo,
    gpu_status,
    show_directory_structure,
    create_sample_plt_files,
)

# ---------------------------------------------------------------------------
# Global system init
# ---------------------------------------------------------------------------
print("Initializing Scribble Plotter Web UI...")
SYSTEM_INFO = setup_complete_system()
CONFIG = CompleteConfiguration(SYSTEM_INFO)
PROCESSING_SYSTEM = CompleteProcessingSystem(CONFIG)

TEMP_BASE = Path(CONFIG.get('directories')['temp'])
TEMP_BASE.mkdir(parents=True, exist_ok=True)


def cleanup_old_sessions(max_age_hours: int = 24) -> None:
    """Delete web_session_* dirs older than max_age_hours."""
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    for entry in TEMP_BASE.iterdir():
        if entry.is_dir() and entry.name.startswith("web_session_"):
            try:
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(entry)
                    print(f"Cleaned up old session: {entry.name}")
            except OSError:
                pass


cleanup_old_sessions()


def find_free_port(start: int = 5000, end: int = 5100) -> int:
    """Find an available port in [start, end) per CLAUDE.md standard."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _filter_log(raw: str) -> str:
    """Keep only error/summary lines from the verbose processing output."""
    keep = []
    for line in raw.splitlines():
        stripped = line.strip()
        # Skip noisy per-file detail
        if stripped.startswith(("Loaded ", "AI Parameters:", "Hopfield Enhanced:",
                                "PNG saved:", "PDF saved:", "DXF saved:",
                                "SVG saved:", "Sequence encoding:", "Accession:",
                                "Generated ", "Stored pattern", "Processing:",
                                "Created:", "Memory full")):
            continue
        if stripped.startswith("Success:"):      # tqdm progress prefix
            continue
        if not stripped:
            continue
        keep.append(line)
    return "\n".join(keep)


def process_uploaded_files(
    files,
    examples: int,
    use_ai: bool,
    use_hopfield: bool,
    gen_png: bool,
    gen_pdf: bool,
    gen_dxf: bool,
    gen_svg: bool,
    organize_groups: bool,
    sequence_file,
    steps_override: int,
):
    """Process uploaded PLT files and return gallery images + zip + log."""
    if not files:
        return [], None, "No files uploaded."

    # Create session directory
    session_id = f"web_session_{int(time.time())}_{os.getpid()}"
    session_dir = TEMP_BASE / session_id
    session_input = session_dir / "input"
    session_output = session_dir / "output"
    session_input.mkdir(parents=True, exist_ok=True)
    session_output.mkdir(parents=True, exist_ok=True)

    # Copy uploaded files to session input
    plt_paths = []
    for f in files:
        src = Path(f)
        dst = session_input / src.name
        shutil.copy2(str(src), str(dst))
        plt_paths.append(dst)

    if not plt_paths:
        return [], None, "No valid PLT files found in upload."

    # Save original config values, then override for this session
    orig_output = CONFIG.config['directories']['output']
    orig_ai = CONFIG.get('use_ai_parameters')
    orig_hopfield = CONFIG.get('use_hopfield_memory')
    orig_png = CONFIG.get('generate_png')
    orig_pdf = CONFIG.get('generate_pdf')
    orig_dxf = CONFIG.get('generate_dxf')
    orig_svg = CONFIG.get('generate_svg')
    orig_groups = CONFIG.get('organize_groups')
    orig_steps = CONFIG.config.get('steps_override')
    orig_seq = PROCESSING_SYSTEM.sequence
    orig_seq_info = PROCESSING_SYSTEM.seq_info

    CONFIG.config['directories']['output'] = str(session_output)
    CONFIG.config['use_ai_parameters'] = use_ai
    CONFIG.config['use_hopfield_memory'] = use_hopfield
    CONFIG.config['generate_png'] = gen_png
    CONFIG.config['generate_pdf'] = gen_pdf
    CONFIG.config['generate_dxf'] = gen_dxf
    CONFIG.config['generate_svg'] = gen_svg
    CONFIG.config['organize_groups'] = organize_groups

    if int(steps_override) > 0:
        CONFIG.config['steps_override'] = int(steps_override)
    else:
        CONFIG.config.pop('steps_override', None)

    if sequence_file:
        PROCESSING_SYSTEM.set_sequence(str(sequence_file))
    else:
        PROCESSING_SYSTEM.sequence = None
        PROCESSING_SYSTEM.seq_info = None

    # Reset processing state
    PROCESSING_SYSTEM.processed_files = []
    PROCESSING_SYSTEM.error_files = []
    PROCESSING_SYSTEM.directory_manager.processed_files = []
    PROCESSING_SYSTEM.directory_manager.file_registry = {}

    # Capture stdout (suppresses verbose per-point output from pipeline)
    log_buffer = io.StringIO()
    errors = []
    start_time = time.time()
    total_ops = len(plt_paths) * int(examples)
    successful = 0

    try:
        with contextlib.redirect_stdout(log_buffer):
            for plt_file in plt_paths:
                for iteration in range(int(examples)):
                    ok = PROCESSING_SYSTEM.process_single_file(plt_file, iteration)
                    if ok:
                        successful += 1

            if organize_groups:
                PROCESSING_SYSTEM.directory_manager.organize_output_files()

    except Exception as exc:
        errors.append(str(exc))
    finally:
        # Restore original config
        CONFIG.config['directories']['output'] = orig_output
        CONFIG.config['use_ai_parameters'] = orig_ai
        CONFIG.config['use_hopfield_memory'] = orig_hopfield
        CONFIG.config['generate_png'] = orig_png
        CONFIG.config['generate_pdf'] = orig_pdf
        CONFIG.config['generate_dxf'] = orig_dxf
        CONFIG.config['generate_svg'] = orig_svg
        CONFIG.config['organize_groups'] = orig_groups
        if orig_steps is not None:
            CONFIG.config['steps_override'] = orig_steps
        else:
            CONFIG.config.pop('steps_override', None)
        PROCESSING_SYSTEM.sequence = orig_seq
        PROCESSING_SYSTEM.seq_info = orig_seq_info

    elapsed = time.time() - start_time

    # Collect gallery images (all PNGs in session output)
    gallery_images = sorted(session_output.rglob("*.png"))
    gallery_paths = [str(p) for p in gallery_images]
    pdf_count = len(list(session_output.rglob("*.pdf")))
    dxf_count = len(list(session_output.rglob("*.dxf")))
    svg_count = len(list(session_output.rglob("*.svg")))

    # Build zip of all outputs
    zip_path = None
    all_outputs = list(session_output.rglob("*"))
    output_files = [p for p in all_outputs if p.is_file()]
    if output_files:
        zip_file = session_dir / "scribble_output.zip"
        with zipfile.ZipFile(str(zip_file), 'w', zipfile.ZIP_DEFLATED) as zf:
            for fp in output_files:
                arcname = fp.relative_to(session_output)
                zf.write(str(fp), str(arcname))
        zip_path = str(zip_file)

    # Build concise summary instead of raw log dump
    summary_lines = [
        f"Processed {len(plt_paths)} file(s) x {examples} variation(s)  "
        f"({successful}/{total_ops} succeeded, {elapsed:.1f}s)",
        f"Output: {len(gallery_paths)} PNG, {pdf_count} PDF, {dxf_count} DXF, {svg_count} SVG",
    ]
    if PROCESSING_SYSTEM.error_files:
        summary_lines.append(
            f"Errors: {len(PROCESSING_SYSTEM.error_files)} file(s) failed"
        )
    if errors:
        summary_lines.append(f"Exception: {errors[0]}")

    # Append only error/summary lines from raw log (skip point counts etc.)
    filtered = _filter_log(log_buffer.getvalue())
    if filtered:
        summary_lines.append("")
        summary_lines.append(filtered)

    return gallery_paths, zip_path, "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Helper functions for other tabs
# ---------------------------------------------------------------------------

def run_hopfield_demo_web() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hopfield_demo(PROCESSING_SYSTEM)
    return buf.getvalue()


def get_system_status() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gpu_status()
        print()
        CONFIG.display_status()
        print()
        base_dir = CONFIG.get('directories')['base']
        show_directory_structure(base_dir)
    return buf.getvalue()


def create_samples_web() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        input_dir = CONFIG.get('directories')['input']
        create_sample_plt_files(input_dir)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Scribble Plotter",
    ) as app:
        gr.Markdown(
            "# GPU-Integrated AI-Enhanced Scribble Plotter\n"
            "*Honoring Kent Benson's 1983-1986 Hopfield Network Research*"
        )

        with gr.Tab("Process PLT Files"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload PLT Files",
                        file_count="multiple",
                        file_types=[".plt", ".PLT"],
                        type="filepath",
                    )
                    with gr.Accordion("Settings", open=True):
                        examples_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Variations per file",
                        )
                        use_ai_cb = gr.Checkbox(value=True, label="AI Parameter Prediction")
                        use_hopfield_cb = gr.Checkbox(value=True, label="Hopfield Memory")
                    with gr.Accordion("Output Formats", open=False):
                        gen_png_cb = gr.Checkbox(value=True, label="PNG")
                        gen_pdf_cb = gr.Checkbox(value=True, label="PDF")
                        gen_dxf_cb = gr.Checkbox(value=True, label="DXF")
                        gen_svg_cb = gr.Checkbox(value=True, label="SVG")
                        organize_cb = gr.Checkbox(value=True, label="GROUP directories")
                    with gr.Accordion("Sequence Encoding", open=False):
                        sequence_input = gr.File(
                            label="FASTA File (optional)",
                            file_count="single",
                            file_types=[".fasta", ".fa", ".fna", ".faa"],
                            type="filepath",
                        )
                        steps_slider = gr.Slider(
                            minimum=0, maximum=20, value=0, step=1,
                            label="Steps override (0 = auto)",
                        )
                    process_btn = gr.Button("Process", variant="primary")

                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Generated Artwork",
                        columns=3,
                        height="auto",
                    )
                    zip_download = gr.File(label="Download All Outputs (ZIP)")
                    log_box = gr.Textbox(
                        label="Processing Log",
                        lines=12,
                        max_lines=30,
                        interactive=False,
                    )

            process_btn.click(
                fn=process_uploaded_files,
                inputs=[
                    file_input,
                    examples_slider,
                    use_ai_cb,
                    use_hopfield_cb,
                    gen_png_cb,
                    gen_pdf_cb,
                    gen_dxf_cb,
                    gen_svg_cb,
                    organize_cb,
                    sequence_input,
                    steps_slider,
                ],
                outputs=[gallery, zip_download, log_box],
            )

        with gr.Tab("Hopfield Demo"):
            demo_btn = gr.Button("Run Hopfield Demo", variant="primary")
            demo_output = gr.Textbox(
                label="Demo Output",
                lines=15,
                max_lines=30,
                interactive=False,
            )
            demo_btn.click(fn=run_hopfield_demo_web, outputs=[demo_output])

        with gr.Tab("System Status"):
            with gr.Row():
                status_btn = gr.Button("Refresh Status", variant="primary")
                samples_btn = gr.Button("Create Sample PLT Files")
            status_output = gr.Textbox(
                label="System Status",
                lines=20,
                max_lines=40,
                interactive=False,
            )
            samples_output = gr.Textbox(
                label="Sample Creation Log",
                lines=5,
                max_lines=10,
                interactive=False,
            )
            status_btn.click(fn=get_system_status, outputs=[status_output])
            samples_btn.click(fn=create_samples_web, outputs=[samples_output])

    return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = find_free_port()
    print(f"\nStarting Scribble Plotter Web UI on port {port}")
    print(f"Local URL:  http://127.0.0.1:{port}")
    print("\nTo expose via Cloudflare tunnel:")
    print(f"  cf-tunnel add scribble {port}")
    print("  -> https://scribble.kentbenson.net")

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
    )
