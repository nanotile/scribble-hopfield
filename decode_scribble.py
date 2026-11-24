#!/usr/bin/env python3
"""
Decode biological sequences from scribble-encoded SVG files.

Module: decode_scribble.py
Version: 1.0.0
Iteration: v1
Developer: Kent Benson
UV Environment: pip install -r requirements.txt

Reads an SVG produced by scribble_plotter_local.py with --sequence
and recovers the embedded DNA or protein sequence by measuring
perpendicular displacement of interior path points from their ideal
positions on each segment.

Usage:
    python decode_scribble.py path/to/encoded.svg
    python decode_scribble.py path/to/encoded.svg -o recovered.fasta
"""

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET


DNA_SYMBOLS = 'ACGT'
AMINO_ACID_SYMBOLS = 'ACDEFGHIKLMNPQRSTVWY'


def _build_decoding_table(alphabet_type):
    symbols = DNA_SYMBOLS if alphabet_type == 'dna' else AMINO_ACID_SYMBOLS
    n = len(symbols)
    levels = [(2 * k + 1 - n) / n for k in range(n)]
    return symbols, levels


def parse_svg_metadata(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = {'svg': 'http://www.w3.org/2000/svg'}
    meta_el = root.find('.//svg:metadata', ns)
    if meta_el is None:
        meta_el = root.find('.//metadata')
    if meta_el is None:
        raise ValueError("No <metadata> element found in SVG")

    enc_el = None
    for child in meta_el:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag == 'scribble-encoding':
            enc_el = child
            break

    if enc_el is None:
        raise ValueError("No <scribble-encoding> element in metadata")

    metadata = {
        'version': int(enc_el.get('version', '1')),
        'method': enc_el.get('method', 'perpendicular-displacement'),
        'alphabet': enc_el.get('alphabet', 'dna'),
        'amplitude': float(enc_el.get('amplitude')),
        'steps': int(enc_el.get('steps')),
    }

    for child in enc_el:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag == 'sequence':
            metadata['sequence_name'] = child.get('name', 'unknown')
            metadata['accession'] = child.get('accession', '')
            metadata['total_bases'] = int(child.get('length', '0'))
            metadata['embedded_count'] = int(child.get('embedded', '0'))
            break

    return metadata


def parse_path_d(d_string):
    """Parse SVG path d attribute into list of (x, y) points."""
    points = []
    tokens = re.findall(r'[ML](-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', d_string)
    for x_str, y_str in tokens:
        points.append((float(x_str), float(y_str)))
    return points


def collect_encoded_paths(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    paths = []

    for el in root.iter():
        tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        if tag == 'path':
            seq_start = el.get('data-seq-start')
            seq_end = el.get('data-seq-end')
            d = el.get('d', '')
            if seq_start is not None and seq_end is not None and d:
                paths.append({
                    'seq_start': int(seq_start),
                    'seq_end': int(seq_end),
                    'points': parse_path_d(d),
                })

    paths.sort(key=lambda p: p['seq_start'])
    return paths


def decode_path(points, steps, amplitude, symbols, levels):
    """Decode bases from one path's waypoints.

    The Y coordinates in the SVG were negated during export,
    so we negate them back before computing displacements.
    """
    restored = [(x, -y) for x, y in points]

    n_points = len(restored)
    if n_points < 2:
        return []

    n_segments = (n_points - 1) // steps
    if n_segments < 1:
        return []

    decoded = []

    for seg in range(n_segments):
        start_idx = seg * steps
        end_idx = (seg + 1) * steps

        if end_idx >= len(restored):
            break

        sx, sy = restored[start_idx]
        ex, ey = restored[end_idx]

        dx = ex - sx
        dy = ey - sy
        seg_len = math.hypot(dx, dy)

        if seg_len < 1e-10:
            continue

        perp_x = -dy / seg_len
        perp_y = dx / seg_len

        for j in range(1, steps):
            pt_idx = start_idx + j
            if pt_idx >= len(restored):
                break

            px, py = restored[pt_idx]

            t = j / steps
            ideal_x = sx + t * dx
            ideal_y = sy + t * dy

            disp_x = px - ideal_x
            disp_y = py - ideal_y

            perp_disp = disp_x * perp_x + disp_y * perp_y

            level = perp_disp / amplitude

            best_idx = 0
            best_dist = abs(level - levels[0])
            for k in range(1, len(levels)):
                dist = abs(level - levels[k])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = k

            decoded.append(symbols[best_idx])

    return decoded


def decode_svg(svg_path):
    metadata = parse_svg_metadata(svg_path)
    paths = collect_encoded_paths(svg_path)

    if not paths:
        raise ValueError("No encoded paths found in SVG")

    symbols, levels = _build_decoding_table(metadata['alphabet'])

    all_bases = []
    for path_info in paths:
        bases = decode_path(
            path_info['points'],
            metadata['steps'],
            metadata['amplitude'],
            symbols, levels)
        all_bases.extend(bases)

    embedded = metadata.get('embedded_count', len(all_bases))
    sequence = ''.join(all_bases[:embedded])

    return sequence, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Decode biological sequences from scribble-encoded SVG')
    parser.add_argument('svg', help='Path to encoded SVG file')
    parser.add_argument('-o', '--output', help='Output FASTA file (default: stdout)')
    parser.add_argument('-v', '--verify', help='FASTA file to verify against')

    args = parser.parse_args()

    sequence, metadata = decode_svg(args.svg)

    name = metadata.get('sequence_name', 'decoded')
    accession = metadata.get('accession', '')
    alphabet = metadata.get('alphabet', 'dna')
    embedded = metadata.get('embedded_count', len(sequence))

    header = f">{name}"
    if accession:
        header += f" accession={accession}"
    header += f" decoded_length={len(sequence)} alphabet={alphabet}"

    fasta_text = header + '\n'
    for i in range(0, len(sequence), 70):
        fasta_text += sequence[i:i+70] + '\n'

    if args.output:
        with open(args.output, 'w') as f:
            f.write(fasta_text)
        print(f"Decoded {len(sequence)} {'bases' if alphabet == 'dna' else 'residues'}"
              f" to {args.output}")
    else:
        sys.stdout.write(fasta_text)

    if args.verify:
        from scribble_plotter_local import read_fasta
        _, original = read_fasta(args.verify)
        original = original[:embedded]

        if sequence == original:
            print(f"PASS: decoded sequence matches original ({len(sequence)} bases)")
            return 0
        else:
            mismatches = sum(1 for a, b in zip(sequence, original) if a != b)
            print(f"FAIL: {mismatches} mismatches out of {len(sequence)} bases")
            for i, (a, b) in enumerate(zip(sequence, original)):
                if a != b:
                    print(f"  Position {i}: decoded={a} expected={b}")
                    if i > 5:
                        print(f"  ... and {mismatches - 6} more")
                        break
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
