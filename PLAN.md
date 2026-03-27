<!-- /autoplan restore point: /home/kent_benson/.gstack/projects/nanotile-scribble-hopfield/master-autoplan-restore-20260327-143831.md -->
# Plan: Integrate v11 Features into Scribble Hopfield

## Summary

Cherry-pick the genuinely new features from the v11 files into the existing codebase while preserving the original's architectural strengths. The v11 files themselves (4x duplicated code, breaking regressions) should NOT be committed as-is.

## Context

- **Branch:** master
- **Current state:** 3 committed files (notebook, .py module, local runner) + GUI
- **v11 files:** Untracked, contain ~1,300 lines of unique code buried in ~5,000 lines of duplication

## What We're Integrating (from v11)

### 1. DirectoryManager Class
**File:** `complete_gpu_integrated_ai_enhanced_scribble_plotter.py`
- New class that manages per-file working directories and GROUP aggregate directories
- Methods: `create_working_directories()`, `organize_output_files()`, `register_file()`
- GROUP directories: GROUP_PDF, GROUP_DXF, GROUP_PNG, GROUP_TIFF, GROUP_TRANSFORM
- Accepts `config` parameter (maintaining existing DI pattern, NOT v11's globals)

### 2. Sample PLT File Generator
**File:** `complete_gpu_integrated_ai_enhanced_scribble_plotter.py`
- `create_sample_plt_files(output_dir)` function
- Creates 3 demo PLT files: rectangle, circle, complex nested shapes
- Enables demo without uploading real PLT files

### 3. Complete Demo Runner
**File:** `complete_gpu_integrated_ai_enhanced_scribble_plotter.py`
- `run_complete_demo()` function for end-to-end demonstration
- Creates samples, runs Hopfield demo, processes test file, shows output structure

### 4. Directory Tree Viewer
**File:** `complete_gpu_integrated_ai_enhanced_scribble_plotter.py`
- `show_directory_structure(base_path, max_depth=3)` utility function
- Recursive tree display with box-drawing characters

### 5. GROUP Organization Config & UI
**Files:** `complete_gpu_integrated_ai_enhanced_scribble_plotter.py`, notebook
- New config key: `organize_groups` (default: True)
- New UI checkbox widget in the interface

## What We're NOT Integrating (v11 regressions)

- ~~Replace `CompleteConfiguration` with global dict~~ — Keep config class with JSON persistence
- ~~Remove `find_spurious_memories()`~~ — Keep Kent's 1985 insight implementation
- ~~Remove GPU feature extraction~~ — Keep 15-feature GPU tensor pipeline
- ~~Change class constructors to parameterless~~ — Keep dependency injection via config
- ~~Rename `GPUHopfieldNetwork` to `HopfieldNetwork`~~ — Keep original name
- ~~Remove `CompleteProcessingSystem` class~~ — Keep orchestration class
- ~~Remove `CompleteInterface` class~~ — Keep OOP interface

## Implementation Steps

### Step 1: Add DirectoryManager to the Python module
- Add `DirectoryManager` class to `complete_gpu_integrated_ai_enhanced_scribble_plotter.py`
- Constructor takes `config` parameter (matching existing pattern)
- Wire into `CompleteProcessingSystem` as `self.directory_manager`
- Add `organize_groups` to `CompleteConfiguration` defaults

### Step 2: Add utility functions to the Python module
- Add `create_sample_plt_files(output_dir)`
- Add `show_directory_structure(base_path, max_depth=3)`
- Add `run_complete_demo()` that uses existing class instances

### Step 3: Update CompleteProcessingSystem
- Use `DirectoryManager` for output directory creation in `process_single_file()`
- Call `organize_output_files()` at end of `process_batch()`
- Register output files via `register_file()` during processing

### Step 4: Update CompleteInterface (notebook)
- Add `organize_groups` checkbox widget
- Wire checkbox to config

### Step 5: Update scribble_plotter_local.py
- Add `--full-demo` flag for new end-to-end demo (`run_complete_demo()`)
- Keep existing `--demo` flag unchanged (runs `hopfield_demo()`)
- Add `--create-samples` flag to generate sample PLT files
- Integrate DirectoryManager into local processing pipeline

### Step 6: Update scribble_plotter_gui.py
- Add GROUP organization checkbox to GUI
- Wire to DirectoryManager

### Step 7: Update notebook
- Sync notebook cells with updated Python module
- Add new cells for demo and sample generation

### Step 8: Clean up v11 files
- Delete untracked v11 files (they're draft iterations, value has been extracted)
- Or: move to a `drafts/` directory if Kent wants to keep them for reference

### Step 9: Update CLAUDE.md
- Document new DirectoryManager, demo features, and GROUP organization
- Update command examples with new flags

## Files Modified

| File | Changes |
|------|---------|
| `complete_gpu_integrated_ai_enhanced_scribble_plotter.py` | Add DirectoryManager, utility functions, config key |
| `scribble_plotter_local.py` | Add --full-demo, --create-samples flags, DirectoryManager integration |
| `scribble_plotter_gui.py` | Add GROUP organization checkbox |
| `COMPLETE_GPU_INTEGRATED_AI_ENHANCED_SCRIBBLE_PLOTTER.ipynb` | Sync with module changes, add demo cell |
| `CLAUDE.md` | Document new features |

## Risk Assessment

- **Low risk:** Adding new class and functions to existing module (additive, no breaking changes)
- **Medium risk:** Modifying `CompleteProcessingSystem.process_single_file()` to use DirectoryManager (existing tests via --test should catch regressions)
- **GUI compatibility:** Maintained — no API changes to classes the GUI imports

## Test Plan

1. `./venv/bin/python scribble_plotter_local.py --test` — Existing system test passes
2. `./venv/bin/python scribble_plotter_local.py --demo` — Existing Hopfield demo unchanged
3. `./venv/bin/python scribble_plotter_local.py --full-demo` — New end-to-end demo runs
4. `./venv/bin/python scribble_plotter_local.py --create-samples` — Sample PLT files created
5. Process sample PLT files and verify GROUP directories are populated
6. Verify GUI launches and GROUP checkbox works

## NOT in Scope

These were raised during review and explicitly deferred:

- **Web-deployable demo** — Future enhancement, not part of v11 feature integration
- **SVG/PDF input support** — PLT-only is by design (the tool converts PLT vector art)
- **Interface consolidation** — Three interfaces (Colab, CLI, GUI) serve different use cases
- **Automated unit test suite** — Manual test plan is sufficient for a personal art project
- **Nobel Prize timing / Hopfield narrative** — Marketing concern, not a code plan item

## What Already Exists

| Asset | Location | Status |
|-------|----------|--------|
| Original codebase | tag `original` (commit 4700259) | Preserved on GitHub |
| v11 draft files | Committed at tag `original` | Source material for cherry-pick |
| PyQt6 GUI | `scribble_plotter_gui.py` | Working, must not break |
| Local CLI | `scribble_plotter_local.py` | Working, must not break |
| Colab notebook | `.ipynb` | Working, must not break |
| venv + requirements.txt | `./venv/` | Working Python environment |

## Error & Rescue Registry

| Error Scenario | Detection | Rescue |
|----------------|-----------|--------|
| DirectoryManager breaks existing output paths | `--test` fails | Revert DirectoryManager wiring in CompleteProcessingSystem, keep class as unused |
| GUI fails to import after module changes | GUI launch test | API is additive-only; revert if constructor signatures changed |
| Notebook cells out of sync | Run all cells in order | Copy class definitions from .py module verbatim |
| GROUP organize_output_files() corrupts originals | Manual inspection | Use COPY not MOVE — originals stay in per-file directories |
| matplotlib OOM from unclosed figures | Memory growth during batch | Add `plt.close(fig)` after every `savefig()` call |

## Failure Modes

1. **GROUP directory race condition** — If two files share the same base name, `organize_output_files()` could overwrite. Mitigation: include parent directory name in copied filename.
2. **Demo runner assumes writable CWD** — `run_complete_demo()` creates directories in CWD. Mitigation: use config's output_dir, not os.getcwd().
3. **Sample PLT files not valid HPGL** — If `create_sample_plt_files()` generates malformed PLT, the processor may crash. Mitigation: test with `--create-samples` then `--input` on the output.

## Review Findings Consensus

### Phase 1: CEO Review [subagent-only]

| # | Finding | Severity | Disposition |
|---|---------|----------|-------------|
| C1 | Plan solves code hygiene, not project impact | High | ACKNOWLEDGE — this IS the goal (fun project, clean integration) |
| C2 | PLT-only input is a limitation | Medium | OUT OF SCOPE — PLT is the domain |
| C3 | Three interfaces for one user | Medium | KEEP — Colab (demo), CLI (batch), GUI (interactive) serve different modes |
| C4 | No web-deployable option | High | DEFER — future enhancement, not v11 integration |
| C5 | Hopfield timing window (Nobel connection) | High | NOT ACTIONABLE in code plan |
| C6 | Missing competitive landscape analysis | Low | N/A — personal art tool |
| C7 | No usage metrics or telemetry | Low | OUT OF SCOPE |
| C8 | Demo should showcase artistic output quality | Medium | PARTIALLY ADDRESSED — demo runner shows full pipeline |
| C9 | Consider Streamlit/Gradio web interface | Medium | DEFER — future enhancement |

### Phase 2: Design Review

UI scope is minimal (one checkbox in GUI, one checkbox in notebook). No design system concerns.

| # | Finding | Severity | Disposition |
|---|---------|----------|-------------|
| D1 | GROUP checkbox placement in GUI | Low | Add to existing "Output Options" group box |
| D2 | IPyWidgets checkbox in notebook | Low | Add below existing format checkboxes |
| D3 | No visual feedback for GROUP organization | Medium | Print directory tree after organize_output_files() completes |

### Phase 3: Eng Review [subagent-only]

| # | Finding | Severity | Disposition |
|---|---------|----------|-------------|
| E1 | matplotlib figure leak — OOM on large batches | Critical | **ADD TO PLAN** — Add `plt.close(fig)` after every savefig() |
| E2 | Zero automated tests | Critical | DEFER — manual test plan sufficient for personal project |
| E3 | Four-way code duplication in v11 | High | THIS IS WHAT THE PLAN FIXES |
| E4 | GROUP copy-vs-move not specified | High | **ADD TO PLAN** — COPY, not move. Originals stay in per-file dirs |
| E5 | `--demo` flag collision with existing | High | ALREADY FIXED — using `--full-demo` |
| E6 | v11 DirectoryManager uses globals | High | ALREADY IN PLAN — refactor to config DI |
| E7 | Notebook DI pattern unclear | High | ALREADY FIXED in design doc — config wrapper |
| E8 | No validation of PLT content in sample generator | Medium | ADD — basic HPGL header/footer validation |
| E9 | `organize_output_files()` filename collision risk | Medium | ADD — prefix with parent dir name |
| E10 | `show_directory_structure()` unbounded recursion on symlinks | Medium | ADD — track visited inodes |
| E11 | Missing `__all__` export list | Medium | DEFER — module is imported whole |
| E12 | No type hints on new functions | Medium | DEFER — match existing code style (no type hints) |
| E13 | `run_complete_demo()` creates temp dirs but doesn't clean up | Medium | ADD — use tempfile or document cleanup |
| E14 | JSON config may not have `organize_groups` on upgrade | Low | HANDLED — CompleteConfiguration has defaults |
| E15 | GROUP_TRANSFORM directory purpose unclear | Low | DOCUMENT — for coordinate transform output files |

## Plan Amendments (from review)

### Amendment A: Fix matplotlib figure leak (from E1)
Add to **Step 3** (Update CompleteProcessingSystem):
- After every `savefig()` call in `ScribbleRenderer`, call `plt.close(fig)` to prevent OOM on large batches

### Amendment B: GROUP uses COPY not MOVE (from E4)
Clarify in **Step 1** (DirectoryManager):
- `organize_output_files()` COPIES files to GROUP directories; originals remain in per-file output directories

### Amendment C: Filename collision prevention (from E9)
Add to **Step 1** (DirectoryManager):
- When copying to GROUP dirs, prefix filename with source directory name to prevent collisions

### Amendment D: Directory tree symlink safety (from E10)
Add to **Step 2** (utility functions):
- `show_directory_structure()` must track visited inodes to prevent infinite recursion on symlinks

### Amendment E: Demo cleanup (from E13)
Add to **Step 2** (utility functions):
- `run_complete_demo()` should use a clearly named demo directory and document that cleanup is manual

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|
| 1 | CEO | Defer web-deployable option (C4) | Pragmatic | v11 integration scope, not greenfield | Web demo now |
| 2 | CEO | Defer interface consolidation (C3) | Explicit > Clever | Three interfaces serve different use cases | Merge to one |
| 3 | CEO | Defer Hopfield narrative (C5) | Bias toward action | Not actionable in code | Pause for marketing |
| 4 | Eng | Add plt.close() fix (E1) | Completeness | OOM is a real bug, low effort to fix | Defer to later |
| 5 | Eng | COPY not MOVE for GROUP (E4) | Explicit > Clever | Non-destructive default prevents data loss | Move for disk space |
| 6 | Eng | Defer automated tests (E2) | Pragmatic | Manual test plan covers the fun-project scope | Write pytest suite |
| 7 | Eng | Add symlink guard (E10) | Completeness | Easy to add, prevents real failure mode | Trust max_depth |
| 8 | Eng | Match existing style, no type hints (E12) | DRY | Consistency with existing 1,400-line module | Add types now |
| 9 | Design | Print tree after GROUP organize (D3) | Explicit > Clever | User sees what happened | Silent operation |
| 10 | CEO | Keep PLT-only scope (C2) | Pragmatic | PLT conversion IS the product | Add SVG/PDF input |
