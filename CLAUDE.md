# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-Integrated AI-Enhanced Scribble Plotter v3.0 - A Google Colab-based system that converts PLT vector graphics into artistic "scribble" renderings using GPU-accelerated AI and Hopfield neural networks. Inspired by Kent Benson's 1983-1986 Hopfield Network research.

## Running the Project

This is a Google Colab notebook project. No traditional build system exists.

**Setup:**
1. Open `COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb` in Google Colab
2. Enable GPU: Runtime → Change Runtime Type → GPU
3. Run cells in order (first cell auto-installs dependencies)
4. Call `INTERFACE.display_interface()` for the interactive UI

**Quick Commands:**
- `quick_test()` - System health check
- `gpu_status()` - GPU availability and memory info
- `hopfield_demo()` - Neural network demonstration

**Dependencies (auto-installed):**
```
pip install matplotlib numpy pandas regex ipywidgets ezdxf reportlab tqdm scikit-learn torch torchvision cupy-cuda11x
```

## Architecture

The system consists of 7 main classes organized in a processing pipeline:

### Core Classes

| Class | Purpose |
|-------|---------|
| `CompleteConfiguration` | GPU detection, Google Drive integration, JSON config persistence |
| `PLTProcessor` | Parses PLT vector format (ACME Convert and ACME Trace formats), coordinate extraction |
| `GPUAcceleratedAI` | Feature extraction (15-feature vectors), parameter prediction with GPU/CPU fallback |
| `GPUHopfieldNetwork` | Stores artistic patterns (200 capacity), Hebbian learning, spurious memory detection |
| `ScribbleRenderer` | Converts coordinates to scribble artwork via matplotlib |
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
├── COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb  # Main notebook
└── complete_gpu_integrated_ai_enhanced_scribble_plotter.py     # Python module version
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
- `scale_factor_x`/`scale_factor_y` - PLT coordinate scaling (default: 0.09)
