# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-Integrated AI-Enhanced Scribble Plotter v3.0 - A Google Colab-based system that converts PLT vector graphics into artistic "scribble" renderings using GPU-accelerated AI and Hopfield neural networks. Inspired by Kent Benson's 1983-1986 Hopfield Network research.

## Running Locally

**Setup:**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
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

# Process PLT files (default: 3 variations each)
./venv/bin/python scribble_plotter_local.py --input /path/to/plt/files

# Process with custom settings
./venv/bin/python scribble_plotter_local.py --input ./files --examples 5 --output ./results

# Disable features
./venv/bin/python scribble_plotter_local.py --no-gpu --no-ai --no-hopfield

# End-to-end demo (creates samples, processes, shows output)
./venv/bin/python scribble_plotter_local.py --full-demo

# Create sample PLT files for testing
./venv/bin/python scribble_plotter_local.py --create-samples

# Disable GROUP directory organization
./venv/bin/python scribble_plotter_local.py --input ./files --no-groups
```

**Input/Output:**
- Place `.plt` files in `ScribblePlotter_Output/input/`
- Output saved to `ScribblePlotter_Output/output/` (PNG, PDF, DXF)
- GROUP directories aggregate outputs: `GROUP_PDF/`, `GROUP_DXF/`, `GROUP_PNG/`

## Running in Google Colab

**Setup:**
1. Open `COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb` in Google Colab
2. Enable GPU: Runtime → Change Runtime Type → GPU
3. Run cells in order (first cell auto-installs dependencies)
4. Call `INTERFACE.display_interface()` for the interactive UI

## Architecture

The system consists of 8 main classes organized in a processing pipeline:

### Core Classes

| Class | Purpose |
|-------|---------|
| `CompleteConfiguration` | GPU detection, Google Drive integration, JSON config persistence |
| `PLTProcessor` | Parses PLT vector format (ACME Convert and ACME Trace formats), coordinate extraction |
| `GPUAcceleratedAI` | Feature extraction (15-feature vectors), parameter prediction with GPU/CPU fallback |
| `GPUHopfieldNetwork` | Stores artistic patterns (200 capacity), Hebbian learning, spurious memory detection |
| `ScribbleRenderer` | Converts coordinates to scribble artwork via matplotlib |
| `DirectoryManager` | Per-file working directories, GROUP aggregate directories (copy-based, non-destructive) |
| `CompleteProcessingSystem` | Batch orchestration, multi-format output (PNG/PDF/DXF) |
| `CompleteInterface` | IPyWidgets-based interactive UI |

### Data Flow

```
PLT Files → PLTProcessor → GPUAcceleratedAI (feature extraction)
         → GPUHopfieldNetwork (pattern recall) → ScribbleRenderer → PNG/PDF/DXF
```

### Key Design Patterns

- **Dual-Mode Computation:** All GPU operations have automatic CPU fallback
- **Pattern Memory:** Hopfield network learns and stores artistic styles across processing runs
- **Spurious Memories:** Implementation of neural network "intuitive leaps" for creative variation
- **Adaptive Parameters:** AI analyzes drawing complexity to select rendering parameters (steps, variation, stroke weight)

## File Structure

```
├── COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb  # Main Colab notebook
├── complete_gpu_integrated_ai_enhanced_scribble_plotter.py     # Colab Python module
├── scribble_plotter_local.py                                   # Local CLI runner
├── scribble_plotter_gui.py                                     # PyQt6 GUI
└── requirements.txt                                            # Python dependencies
```

Google Drive directories created at runtime:
- `ScribblePlotter_GPU_Complete/input` - PLT source files
- `ScribblePlotter_GPU_Complete/output` - Generated artwork
- `ScribblePlotter_GPU_Complete/models` - Saved Hopfield patterns
- `ScribblePlotter_GPU_Complete/config` - JSON configuration

## Key Configuration

Settings are persisted in JSON and accessible via `CONFIG.get()`/`CONFIG.set()`:

- `use_ai_parameters` - Enable AI-based parameter prediction
- `use_hopfield_memory` - Enable pattern learning/recall
- `gpu_enabled` - GPU acceleration toggle
- `total_examples` - Variations per input file (default: 3)
- `generate_pdf`/`generate_dxf`/`generate_png` - Output format toggles
- `organize_groups` - Copy outputs into GROUP aggregate directories (default: true)
- `scale_factor_x`/`scale_factor_y` - PLT coordinate scaling (default: 0.09)
