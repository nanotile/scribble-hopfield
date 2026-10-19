# Scribble Plotter Rust Port — Design Spec

**Date:** 2026-10-19
**Author:** Kent Benson
**Status:** Draft

## Goal

Port the GPU-Integrated AI-Enhanced Scribble Plotter from Python to pure Rust.
Primary motivations: single-binary distribution (no Python/PyTorch/pip) and
raw performance (parallel batch processing, zero-copy PLT parsing).

## Scope

**In scope (this spec):**
- CLI binary with feature parity to `scribble_plotter_local.py`
- All 8 core classes ported as Rust modules in a reusable library crate
- PNG, PDF, and DXF output
- Hopfield network (CPU only — 15x15 is trivial)
- AI feature extraction (CPU only — ndarray + rayon)
- Parallel batch processing via rayon
- JSON config persistence
- Directory management and GROUP organization
- Sample PLT file creation
- All existing CLI flags

**Out of scope (future work):**
- Gradio web UI (future: axum + htmx or egui-web)
- PyQt6 desktop GUI (future: egui or iced)
- Google Colab notebook (stays Python)
- GPU compute shaders via wgpu (add if profiling justifies)
- IPyWidgets interactive interface

## Project Structure

```
scribble-hopfield-rs/
├── Cargo.toml                  # Workspace root
├── scribble-core/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # Public API re-exports
│       ├── config.rs           # Config system (JSON persistence)
│       ├── plt.rs              # PLT file parser
│       ├── features.rs         # Feature extraction (15-vector)
│       ├── hopfield.rs         # Hopfield network (store/recall/spurious)
│       ├── renderer.rs         # Scribble rendering → PNG/PDF
│       ├── dxf.rs              # DXF output writer
│       ├── dirs.rs             # Directory manager + GROUP organizer
│       └── pipeline.rs         # Batch processing orchestrator
└── scribble-cli/
    ├── Cargo.toml
    └── src/
        └── main.rs             # clap CLI (all --flags from Python version)
```

Location: `~/scribble_hopfield/scribble-hopfield-rs/` — alongside existing Python.

## Module Design

### config.rs — Config

Replaces `CompleteConfiguration`. Struct-based with serde.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub project_name: String,
    pub version: String,
    // Processing
    pub total_examples: u32,
    // Output toggles
    pub generate_png: bool,
    pub generate_pdf: bool,
    pub generate_dxf: bool,
    pub organize_groups: bool,
    pub output_dpi: u32,
    // Graphics
    pub page_width: u32,
    pub page_height: u32,
    pub scale_factor_x: f64,
    pub scale_factor_y: f64,
    pub scale_factor_z: f64,
    // AI
    pub use_ai_parameters: bool,
    // Hopfield
    pub use_hopfield_memory: bool,
    pub hopfield_memory_size: usize,
    pub hopfield_pattern_size: usize,
    // Scribble
    pub random_steps_upper_limit: u32,
    pub scribble_upper_limit: f64,
    pub steps_fixed: u32,
    pub scribble_fixed: f64,
    pub stroke_weight_value: f64,
    // Directories (runtime, not serialized)
    #[serde(skip)]
    pub directories: Directories,
}
```

`Config::load(path)` merges JSON on disk with compiled defaults (same
pattern as the Python version). `Config::save()` writes back to JSON.
`Directories` is a separate struct holding input/output/config/models/temp
paths, populated at startup based on CLI args or defaults.

### plt.rs — PltParser

Replaces `PLTProcessor`. Parses PLT files into `Vec<Point3>`.

```rust
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,  // 0.0 = pen down, 10.0 = pen up
}
```

Three format detectors (same as Python):
1. Raw coordinate triplets (`x,y,z;`)
2. ACME Convert (contains `PW0` in first 200 chars)
3. ACME Trace (default)

Regex patterns compiled once via `std::sync::OnceLock` (stable since Rust 1.80):
- `PD(\d+),(\d+)` and negative variants
- `PA(\d+)\.(\d+),(\d+)\.(\d+)`
- `PU(\d+)\.(\d+),(\d+)\.(\d+)`

`scale_points()` applies config scale factors. Returns owned `Vec<Point3>`.

### features.rs — FeatureExtractor

Replaces `GPUAcceleratedAI`. Extracts a 15-element feature vector from
a point cloud. Uses `ndarray` for vectorized math (replaces numpy).

Features computed (same as Python):
1. Line complexity (mean angle change / pi)
2. Total path length
3. Bounding box width
4. Bounding box height
5. Aspect ratio
6. Point density
7. Centroid X
8. Centroid Y
9. Spread (std dev of distances from centroid)
10-15. Padding (0.5)

`predict_parameters()` maps features to scribble parameters using the
same complexity-based heuristic as Python. Uses `rand` crate for
randomization (replaces `random` module).

Returns:
```rust
pub struct ScribbleParams {
    pub steps: u32,
    pub scribble: f64,
    pub stroke_weight: f64,
    pub color: (f64, f64, f64),
    pub complexity_score: f64,
    pub generation_method: String,
}
```

### hopfield.rs — HopfieldNetwork

Replaces `GPUHopfieldNetwork`. Pure CPU implementation using `ndarray`.

```rust
pub struct HopfieldNetwork {
    pattern_size: usize,
    memory_size: usize,
    weights: Array2<f64>,          // ndarray 2D
    stored_patterns: Vec<Array1<f64>>,
    pattern_labels: Vec<String>,
}
```

Methods:
- `store_pattern(pattern, label)` — Hebbian learning (outer product update)
- `recall_pattern(partial, max_iterations)` — synchronous update until convergence
- `find_spurious_memories(test_patterns)` — Kent's 1985 insight
- Internal: `normalize()` (z-score then sign), `denormalize()` ((x+1)/2)
- `patterns_similar()` — correlation > 0.9 threshold

The 15x15 weight matrix is trivially fast on CPU. No GPU needed.

### renderer.rs — Renderer

Replaces `ScribbleRenderer`. Uses the `plotters` crate to generate
PNG and PDF output.

The scribble algorithm is identical to Python:
- For each consecutive pen-down segment, subdivide into `steps` sub-segments
- Add uniform random noise (`-scribble..+scribble`) to intermediate points
- Final point snaps exactly to the endpoint
- Draw each sub-segment as a line with the given color/weight/alpha

`plotters` supports both `BitMapBackend` (PNG) and `SVGBackend`. For PDF,
use `plotters-cairo` or render to SVG and convert via `svg2pdf` crate.
If `plotters-cairo` is too heavy a dependency, render PNG at high DPI
and embed in a single-page PDF via `printpdf`.

**Decision: PNG via `plotters` BitMapBackend. PDF via `printpdf` embedding
the rendered PNG at 300 DPI.** This matches the Python behavior (matplotlib
savefig to PDF is also a raster embed at the configured DPI) and avoids
heavy Cairo dependencies.

Output includes `invert_yaxis` to match Processing coordinate system.

### dxf.rs — DxfWriter

Replaces the `save_dxf()` method. Writes DXF R2010 format.

Uses the `dxf` crate (pure Rust DXF reader/writer) to create a new
document, add LINE entities with the scribble algorithm applied, and
save to disk.

The scribble algorithm for DXF is identical to the renderer — same
subdivision + noise — but outputs `dxf::entities::Line` instead of
plot commands.

### dirs.rs — OutputOrganizer

Replaces `DirectoryManager`. Handles:

1. Per-file working directories: `{output}/{basename}_{iteration}/{PDF,DXF,PNG,TMP}/`
2. GROUP aggregate directories: `{output}/GROUP_PDF/`, `GROUP_DXF/`, `GROUP_PNG/`
3. File registration and non-destructive copy to GROUP dirs
4. Collision-safe naming: `{parent}_{filename}` prefix

Uses `std::fs` for all operations. `fs::copy` for GROUP organization
(same non-destructive copy semantics as Python's `shutil.copy2`).

### pipeline.rs — Pipeline

Replaces `CompleteProcessingSystem`. Orchestrates the full processing flow.

```rust
pub struct Pipeline {
    config: Config,
    parser: PltParser,
    features: FeatureExtractor,
    hopfield: HopfieldNetwork,
    renderer: Renderer,
    organizer: OutputOrganizer,
}
```

`process_batch(input_dir)`:
1. Glob `*.plt` + `*.PLT` files
2. For each file x each iteration (total_examples):
   - Parse PLT → points
   - Scale points
   - Extract features → predict params (if AI enabled)
   - Hopfield recall + blend (if Hopfield enabled)
   - Render PNG (if enabled)
   - Render PDF (if enabled)
   - Write DXF (if enabled)
   - Store pattern in Hopfield memory
3. Organize GROUP directories
4. Print summary

**Parallelism:** Process files sequentially (preserving Hopfield learning
order across files, matching Python behavior). Within each file,
parallelize the `total_examples` iterations via `rayon::par_iter` — each
iteration is independent (own random seed, own output). Hopfield pattern
storage happens after each file completes (sequential), so no
`Arc<Mutex<>>` needed. This gives meaningful speedup on multi-core
machines while keeping the Hopfield state simple and deterministic.

`process_single_file()` remains the unit of work, returning a result
struct with success/failure and output paths.

Progress reporting via `indicatif::ProgressBar`.

### main.rs — CLI

Replaces `main()` in `scribble_plotter_local.py`. Uses `clap` derive API.

```
scribble [OPTIONS]

Options:
  -i, --input <DIR>       Input directory containing PLT files
  -o, --output <DIR>      Base output directory
  -e, --examples <N>      Variations per file [default: 3]
      --no-ai             Disable AI parameter prediction
      --no-hopfield       Disable Hopfield memory
      --no-groups         Disable GROUP directory organization
      --test              Quick system test
      --demo              Hopfield network demo
      --full-demo         End-to-end demo with samples
      --create-samples    Create sample PLT files
  -h, --help              Print help
  -V, --version           Print version
```

Removes GPU-specific flags (`--no-gpu`, `--gpu-status`) since v1 is
CPU-only. These return as `--gpu` / `--gpu-status` if wgpu is added later.

## Dependencies (Cargo.toml)

### scribble-core
```toml
[dependencies]
ndarray = "0.16"
regex = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"
rayon = "1.10"
plotters = "0.3"
plotters-bitmap = "0.3"
printpdf = "0.7"
dxf = "0.6"
thiserror = "2"
```

### scribble-cli
```toml
[dependencies]
scribble-core = { path = "../scribble-core" }
clap = { version = "4", features = ["derive"] }
indicatif = "0.17"
anyhow = "1"
```

## Error Handling

- `scribble-core` defines `ScribbleError` via `thiserror` with variants
  for IO, parse, render, config, and DXF errors.
- `scribble-cli` uses `anyhow::Result` for top-level error reporting.
- No panics in library code. All fallible operations return `Result`.
- Matches Python behavior: individual file failures don't abort the batch.

## Testing Strategy

- Unit tests per module (`#[cfg(test)]` inline):
  - `plt.rs`: parse all three PLT formats, negative coordinates, empty files
  - `features.rs`: known point clouds → expected feature vectors
  - `hopfield.rs`: store/recall convergence, spurious memory detection
  - `renderer.rs`: render to in-memory buffer, verify non-empty
  - `dxf.rs`: write and re-read DXF, verify entity count
  - `config.rs`: load/save roundtrip, merge with defaults
- Integration test: end-to-end processing of sample PLT files
- The 3 sample PLT files from `create_sample_plt_files()` serve as test fixtures
- `cargo test` runs everything; no external dependencies needed

## Migration Path

1. Build `scribble-hopfield-rs/` alongside existing Python code
2. Verify output parity: process the same PLT files with both versions,
   compare that outputs exist in the same directory structure (exact pixel
   match is not expected due to random variation, but structure and formats
   must match)
3. Once validated, the Rust binary replaces CLI usage. Python Colab notebook
   remains as-is.
4. Future: add web UI (axum) and desktop GUI (egui) as additional binary
   crates in the workspace

## What This Doesn't Change

- The Colab notebook (`COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb`)
  stays Python — Colab requires it
- The Python module (`complete_gpu_integrated_ai_enhanced_scribble_plotter.py`)
  stays as the Colab backing module
- PLT file format and semantics are preserved exactly
- Config JSON format is compatible (Rust reads Python-generated configs)
- Output directory structure is identical
- All scribble algorithm parameters and heuristics are preserved
