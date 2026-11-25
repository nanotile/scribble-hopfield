# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-Integrated AI-Enhanced Scribble Plotter v4.1 - Converts PLT, DXF, DWG, and DWX vector graphics into artistic "scribble" renderings using GPU-accelerated AI and Hopfield neural networks, with optional biological sequence encoding that embeds DNA or protein data as vector geometry. Inspired by Kent Benson's 1983-1986 Hopfield Network research.

## Running Locally

**Setup:**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (torch is optional — numpy fallback works without it)
pip install -r requirements.txt
```

**Commands:**
```bash
# Quick system test
./venv/bin/python scribble_plotter_local.py --test

# GPU status check
./venv/bin/python scribble_plotter_local.py --gpu-status

# Hopfield network demo
./venv/bin/python scribble_plotter_local.py --demo

# Process input files (PLT, DXF, DWG, DWX — 3 variations each, outputs PNG/PDF/DXF/SVG)
./venv/bin/python scribble_plotter_local.py --input /path/to/files

# Process with custom settings
./venv/bin/python scribble_plotter_local.py --input ./files --examples 5 --output ./results

# Disable features
./venv/bin/python scribble_plotter_local.py --no-gpu --no-ai --no-hopfield --no-svg --no-smooth

# End-to-end demo (creates samples, processes, shows output)
./venv/bin/python scribble_plotter_local.py --full-demo

# Create sample PLT files for testing
./venv/bin/python scribble_plotter_local.py --create-samples

# Disable GROUP directory organization
./venv/bin/python scribble_plotter_local.py --input ./files --no-groups

# Sequence-driven wobble: encode a DNA/protein FASTA into the drawing
./venv/bin/python scribble_plotter_local.py --input ./files --sequence path/to/file.fasta --steps 8

# Decode the sequence back from an encoded SVG
./venv/bin/python decode_scribble.py output.svg
./venv/bin/python decode_scribble.py output.svg --verify original.fasta

# Run round-trip tests
./venv/bin/python test_roundtrip.py
```

**Input/Output:**
- Place `.plt`, `.dxf`, `.dwg`, or `.dwx` files in `ScribblePlotter_Output/input/`
- Output saved to `ScribblePlotter_Output/output/` (PNG, PDF, DXF, SVG)
- GROUP directories aggregate outputs: `GROUP_PDF/`, `GROUP_DXF/`, `GROUP_PNG/`, `GROUP_SVG/`

## Sequence Encoding

The `--sequence` flag replaces random wobble with deterministic perpendicular displacements driven by a DNA or protein FASTA sequence. Each base/residue is encoded as a quantized offset on the perpendicular axis of each line segment's interior steps.

- **Deterministic:** same drawing + same sequence = identical output every time.
- **Decodable:** `decode_scribble.py` reads the SVG and recovers the exact sequence.
- **Endpoints preserved:** only interior steps are displaced; segment endpoints stay on the original drawing so the picture remains recognisable.
- **Capacity:** each segment encodes `steps - 1` bases. If the sequence exceeds capacity, it is truncated and a warning is printed. Use `--steps N` to increase capacity.
- **Metadata:** the SVG contains a `<metadata>` block with sequence name, accession, encoding version, amplitude, steps, and alphabet. Each `<path>` has `data-seq-start` and `data-seq-end` attributes.

## Running in Google Colab

**Setup:**
1. Open `COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb` in Google Colab
2. Enable GPU: Runtime → Change Runtime Type → GPU
3. Run cells in order (first cell auto-installs dependencies)
4. Call `INTERFACE.display_interface()` for the interactive UI

## Architecture

The system consists of these main classes organized in a processing pipeline:

### Core Classes

| Class | Purpose |
|-------|---------|
| `CompleteConfiguration` | GPU detection, JSON config persistence (machine-specific keys never saved) |
| `PLTProcessor` | Unified HPGL parser — handles PU/PD/PA with integer, decimal, negative coords and multiple pairs per command |
| `CADFileProcessor` | Reads DXF/DWG/DWX via ezdxf, extracts LINE/POLYLINE/CIRCLE/ARC/ELLIPSE/SPLINE entities, auto-normalizes coordinates |
| `GPUAcceleratedAI` | Feature extraction (15-feature vectors), deterministic parameter prediction via continuous complexity mapping |
| `GPUHopfieldNetwork` | Stores artistic patterns (200 capacity), Hebbian learning with continuous tanh activation, multi-parameter recall |
| `ScribbleGenerator` | Generates scribbled geometry once from points + params (random or sequence-driven wobble), cubic spline smoothing |
| `ScribbleRenderer` | Draws pre-generated strokes via matplotlib, canvas sized from input bounding box |
| `SVGExporter` | Writes SVG directly as `<path>` elements with optional sequence metadata |
| `DirectoryManager` | Per-file working directories, GROUP aggregate directories (copy-based, non-destructive) |
| `CompleteProcessingSystem` | Batch orchestration, multi-format output (PNG/PDF/DXF/SVG) |
| `CompleteInterface` | IPyWidgets-based interactive UI (Colab only) |

### Data Flow

```
Input Files → PLTProcessor (PLT) or CADFileProcessor (DXF/DWG/DWX)
         → GPUAcceleratedAI (feature extraction)
         → GPUHopfieldNetwork (pattern recall)
         → ScribbleGenerator (wobble geometry, generated ONCE)
         → ScribbleRenderer (PNG/PDF) + SVGExporter (SVG) + DXF writer
         → all formats share identical geometry
```

### Key Design Patterns

- **Dual-Mode Computation:** All GPU operations have automatic CPU fallback; torch is optional
- **Single Geometry:** Scribbled paths are generated once and shared across all exporters (PNG, PDF, DXF, SVG)
- **Pattern Memory:** Hopfield network learns and stores artistic styles across processing runs
- **Spurious Memories:** Implementation of neural network "intuitive leaps" for creative variation
- **Adaptive Parameters:** AI maps drawing complexity/density/path-length to parameters via continuous functions (deterministic, no random)
- **Continuous Hopfield:** tanh activation preserves nuance; recalled patterns influence steps, scribble, and stroke weight
- **Spline Smoothing:** Cubic spline interpolation smooths scribble strokes (auto-disabled in sequence mode, `--no-smooth` to disable)
- **Adaptive Canvas:** Renderer derives figure dimensions from stroke bounding box instead of fixed size
- **Sequence Encoding:** Perpendicular displacement quantization embeds decodable biological data

## Running the Web UI (Gradio)

```bash
# Launch web interface (auto-detects free port in 5000-5100)
./venv/bin/python scribble_plotter_web.py

# Expose via Cloudflare tunnel
cf-tunnel add scribble <port>
# -> https://scribble.kentbenson.net
```

Three tabs: **Process PLT Files** (upload, configure, gallery + ZIP download), **Hopfield Demo**, **System Status**.

## File Structure

```
├── COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb  # Main Colab notebook
├── complete_gpu_integrated_ai_enhanced_scribble_plotter.py     # Colab Python module
├── scribble_plotter_local.py                                   # Local CLI runner (v4.0)
├── scribble_plotter_gui.py                                     # PyQt6 GUI
├── scribble_plotter_web.py                                     # Gradio web UI
├── decode_scribble.py                                          # Sequence decoder
├── test_roundtrip.py                                           # Encode/decode round-trip tests
├── test_data/                                                  # Test sequences (FASTA)
└── requirements.txt                                            # Python dependencies
```

Google Drive directories created at runtime:
- `ScribblePlotter_GPU_Complete/input` - PLT source files
- `ScribblePlotter_GPU_Complete/output` - Generated artwork
- `ScribblePlotter_GPU_Complete/models` - Saved Hopfield patterns
- `ScribblePlotter_GPU_Complete/config` - JSON configuration

## Key Configuration

Settings are persisted in JSON and accessible via `CONFIG.get()`/`CONFIG.set()`.
Machine-specific keys (`directories`, `device`, `gpu_enabled`, etc.) are computed at startup and never loaded from the saved config file.

- `use_ai_parameters` - Enable AI-based parameter prediction
- `use_hopfield_memory` - Enable pattern learning/recall
- `gpu_enabled` - GPU acceleration toggle (auto-detected)
- `total_examples` - Variations per input file (default: 3)
- `generate_pdf`/`generate_dxf`/`generate_png`/`generate_svg` - Output format toggles
- `organize_groups` - Copy outputs into GROUP aggregate directories (default: true)
- `scale_factor_x`/`scale_factor_y` - PLT coordinate scaling (default: 0.09)
