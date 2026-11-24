# Scribble Plotter

## What It Does

Scribble Plotter transforms CAD vector drawings into one-of-a-kind artistic "scribble" renderings. Feed it a PLT, DXF, DWG, or DWX file and it produces hand-drawn-looking artwork where every straight line becomes a lively, wobbly stroke — as if a skilled draughtsman traced the original with a slightly unsteady hand.

Each run generates multiple variations of the same drawing, so you get a family of related pieces from a single input file.

## How It Works

1. **Parse** — The system reads CAD geometry: pen-up/pen-down moves from HPGL plotter files, or LINE/POLYLINE/CIRCLE/ARC/ELLIPSE/SPLINE entities from DXF/DWG/DWX files.

2. **Analyze** — A GPU-accelerated AI extracts 15 geometric features from the drawing (density, curvature, symmetry, etc.) and predicts rendering parameters: how many wobble steps per segment, how much scribble variation, and stroke weight.

3. **Remember** — A Hopfield neural network (inspired by Kent Benson's 1983-1986 research) stores artistic styles as learned patterns. When processing new drawings, the network recalls the closest stored style and blends it with the AI's predictions, producing outputs that evolve with each batch.

4. **Generate** — The scribble engine displaces each line segment's interior points perpendicular to the stroke direction, creating the characteristic hand-drawn wobble. Geometry is generated once and shared across all output formats.

5. **Export** — Every variation is saved as PNG, PDF, DXF, and SVG. Output files are organized into per-file directories with GROUP aggregate folders for easy browsing.

## Biological Sequence Encoding

An optional mode replaces random wobble with deterministic displacements driven by a DNA or protein FASTA sequence. Each base or amino acid residue is encoded as a quantized perpendicular offset, embedding biological data directly into the drawing geometry. The encoded sequence can be decoded back from the SVG output — the drawing becomes a steganographic carrier for genetic information.

## Interfaces

- **Command Line** — `scribble_plotter_local.py` for batch processing, scripting, and automation.
- **Web UI** — `scribble_plotter_web.py` serves a Gradio interface at `scribble.kentbenson.net` with upload, preview gallery, and download.
- **Desktop GUI** — `scribble_plotter_gui.py` provides a PyQt6 native application with real-time preview.

## Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| HPGL Plotter | `.plt` | Native format — PU/PD/PA commands |
| AutoCAD DXF | `.dxf` | Full entity support via ezdxf |
| AutoCAD DWG | `.dwg` | Auto-converted via dwg2dxf (libredwg) |
| DWX Exchange | `.dwx` | Read as DXF-compatible |

## Output Formats

| Format | Use Case |
|--------|----------|
| PNG | Web display, preview |
| PDF | Print-ready, vector |
| DXF | CAD re-import, laser cutting |
| SVG | Web vector, sequence decoding |

## Origin

The project honors Kent Benson's 1983-1986 Hopfield Network research at university, applying neural network pattern memory to artistic rendering — the idea that a network trained on artistic styles can recall and apply those styles to new drawings, including the occasional "spurious memory" that produces unexpected creative variations.
