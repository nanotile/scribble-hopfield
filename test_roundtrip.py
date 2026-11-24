#!/usr/bin/env python3
"""
Round-trip test: encode a DNA sequence into a scribble drawing SVG,
then decode it and verify the sequence matches exactly.

Module: test_roundtrip.py
Version: 1.0.0
Iteration: v1
Developer: Kent Benson
UV Environment: pip install -r requirements.txt

Usage:
    python test_roundtrip.py
"""

import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import scribble_plotter_local as spl
from decode_scribble import decode_svg


def make_config(tmpdir):
    dirs = {
        'base': str(tmpdir),
        'input': str(tmpdir / 'input'),
        'output': str(tmpdir / 'output'),
        'models': str(tmpdir / 'models'),
        'config': str(tmpdir / 'config'),
        'temp': str(tmpdir / 'temp'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    class FakeConfig:
        def __init__(self):
            self.data = {
                'directories': dirs,
                'device': 'cpu',
                'gpu_enabled': False,
                'scale_factor_x': 0.09,
                'scale_factor_y': 0.09,
                'scale_factor_z': 1.0,
                'page_width': 800,
                'page_height': 600,
                'organize_groups': False,
            }
            self.config = self.data

        def get(self, key, default=None):
            return self.data.get(key, default)

    return FakeConfig()


def test_roundtrip_dna():
    """Encode 300bp of phiX174 into SVG, decode, verify exact match."""
    project_dir = Path(__file__).parent
    fasta_path = project_dir / 'test_data' / 'phix174_300bp.fasta'
    plt_path = project_dir / 'ScribblePlotter_Output' / 'input' / 'sample_drawing.plt'

    if not fasta_path.exists():
        print(f"SKIP: test FASTA not found at {fasta_path}")
        return True
    if not plt_path.exists():
        print(f"SKIP: test PLT not found at {plt_path}")
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix='scribble_test_'))
    try:
        config = make_config(tmpdir)

        header, sequence = spl.read_fasta(str(fasta_path))
        name, accession = spl.parse_fasta_header(header)

        proc = spl.PLTProcessor(config)
        raw = proc.process_plt_file(str(plt_path))
        points = proc.scale_points(raw)

        steps = 8
        amplitude = 4.0
        params = {
            'steps': steps,
            'scribble': amplitude,
            'stroke_weight': 2.0,
            'color': (0, 0, 0),
        }
        seq_info = {
            'sequence_name': name,
            'accession': accession,
        }

        gen = spl.ScribbleGenerator(config)
        strokes, encoding_info = gen.generate(points, params, sequence, seq_info)

        assert strokes, "No strokes generated"
        assert encoding_info is not None, "No encoding info"
        assert encoding_info['embedded_count'] == len(sequence), (
            f"Expected {len(sequence)} embedded, got {encoding_info['embedded_count']}")

        svg_path = tmpdir / 'output' / 'test.svg'
        exporter = spl.SVGExporter()
        exporter.export(strokes, str(svg_path), params, encoding_info)

        assert svg_path.exists(), "SVG not written"

        decoded, metadata = decode_svg(str(svg_path))

        if decoded != sequence:
            mismatches = sum(1 for a, b in zip(decoded, sequence) if a != b)
            print(f"FAIL: {mismatches} mismatches in {len(sequence)} bases")
            for i, (a, b) in enumerate(zip(decoded, sequence)):
                if a != b:
                    print(f"  Position {i}: decoded={a} expected={b}")
                    if i > 10:
                        break
            return False

        print(f"PASS: round-trip test — {len(sequence)} bases encoded and decoded exactly")
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_roundtrip_synthetic():
    """Encode a synthetic random sequence, verify exact round-trip."""
    project_dir = Path(__file__).parent
    plt_path = project_dir / 'ScribblePlotter_Output' / 'input' / 'sample_drawing.plt'

    if not plt_path.exists():
        print(f"SKIP: test PLT not found at {plt_path}")
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix='scribble_test_'))
    try:
        config = make_config(tmpdir)

        rng = random.Random(12345)
        sequence = ''.join(rng.choice('ACGT') for _ in range(200))

        proc = spl.PLTProcessor(config)
        raw = proc.process_plt_file(str(plt_path))
        points = proc.scale_points(raw)

        for steps in [4, 6, 10, 15]:
            params = {
                'steps': steps,
                'scribble': 3.0,
                'stroke_weight': 2.0,
                'color': (0, 0, 0),
            }
            seq_info = {'sequence_name': 'synthetic', 'accession': 'SYNTH001'}

            gen = spl.ScribbleGenerator(config)
            strokes, enc_info = gen.generate(points, params, sequence, seq_info)

            svg_path = tmpdir / 'output' / f'test_steps{steps}.svg'
            exporter = spl.SVGExporter()
            exporter.export(strokes, str(svg_path), params, enc_info)

            decoded, _ = decode_svg(str(svg_path))

            if decoded != sequence:
                mismatches = sum(1 for a, b in zip(decoded, sequence) if a != b)
                print(f"FAIL (steps={steps}): {mismatches} mismatches")
                return False

        print("PASS: synthetic round-trip — 200 bases × 4 step configs all exact")
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_deterministic():
    """Same drawing + same sequence produces identical SVG twice."""
    project_dir = Path(__file__).parent
    plt_path = project_dir / 'ScribblePlotter_Output' / 'input' / 'sample_drawing.plt'

    if not plt_path.exists():
        print("SKIP: test PLT not found")
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix='scribble_test_'))
    try:
        config = make_config(tmpdir)
        sequence = 'ACGTACGTACGTACGT' * 10

        proc = spl.PLTProcessor(config)
        raw = proc.process_plt_file(str(plt_path))
        points = proc.scale_points(raw)

        params = {
            'steps': 6, 'scribble': 3.0,
            'stroke_weight': 2.0, 'color': (0, 0, 0),
        }
        seq_info = {'sequence_name': 'test', 'accession': 'TEST'}

        svgs = []
        for run in range(2):
            gen = spl.ScribbleGenerator(config)
            strokes, enc_info = gen.generate(points, params, sequence, seq_info)
            svg_path = tmpdir / 'output' / f'run{run}.svg'
            spl.SVGExporter().export(strokes, str(svg_path), params, enc_info)
            with open(svg_path) as f:
                svgs.append(f.read())

        if svgs[0] != svgs[1]:
            print("FAIL: two runs produced different SVGs")
            return False

        print("PASS: deterministic — two runs produce identical SVG")
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_truncation_report():
    """When sequence is longer than capacity, it's truncated and reported."""
    project_dir = Path(__file__).parent
    plt_path = project_dir / 'ScribblePlotter_Output' / 'input' / 'sample_drawing.plt'

    if not plt_path.exists():
        print("SKIP: test PLT not found")
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix='scribble_test_'))
    try:
        config = make_config(tmpdir)

        proc = spl.PLTProcessor(config)
        raw = proc.process_plt_file(str(plt_path))
        points = proc.scale_points(raw)

        long_sequence = 'ACGT' * 500

        params = {
            'steps': 4, 'scribble': 3.0,
            'stroke_weight': 2.0, 'color': (0, 0, 0),
        }
        seq_info = {'sequence_name': 'long', 'accession': 'LONG'}

        gen = spl.ScribbleGenerator(config)
        strokes, enc_info = gen.generate(points, params, long_sequence, seq_info)

        embedded = enc_info['embedded_count']
        capacity = enc_info['capacity']
        total = enc_info['total_bases']

        assert embedded <= capacity, f"Embedded ({embedded}) > capacity ({capacity})"
        assert embedded < total, f"Expected truncation but all {total} bases fit"

        svg_path = tmpdir / 'output' / 'truncated.svg'
        spl.SVGExporter().export(strokes, str(svg_path), params, enc_info)
        decoded, _ = decode_svg(str(svg_path))
        expected = long_sequence[:embedded]

        if decoded != expected:
            print("FAIL: truncated decode mismatch")
            return False

        print(f"PASS: truncation — {total} base sequence truncated to"
              f" {embedded}/{capacity} capacity, decoded correctly")
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    # Suppress banner output
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    # Force module reload to capture setup prints
    import importlib
    importlib.reload(spl)
    sys.stdout = old_stdout

    print("=" * 60)
    print("SCRIBBLE PLOTTER ROUND-TRIP TESTS")
    print("=" * 60)

    tests = [
        ("DNA round-trip (phiX174 300bp)", test_roundtrip_dna),
        ("Synthetic round-trip (4 step configs)", test_roundtrip_synthetic),
        ("Deterministic output", test_deterministic),
        ("Truncation reporting", test_truncation_report),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            passed = test_fn()
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            passed = False
        results.append((name, passed))

    print("\n" + "=" * 60)
    all_passed = all(r for _, r in results)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
