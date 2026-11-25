# -*- coding: utf-8 -*-
"""
GPU-Integrated AI-Enhanced Scribble Plotter - Local Version
Module: scribble_plotter_local.py
Version: 4.1.0
Iteration: v2
Developer: Kent Benson
UV Environment: pip install -r requirements.txt
"""

import os
import json
import re
import random
import math
import hashlib
import shutil
import warnings
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore')

print("GPU-Integrated AI-Enhanced Scribble Plotter (Local Version)")
print("Honoring Kent Benson's 1983-1986 Hopfield Network Research")
print("=" * 70)


# ---------------------------------------------------------------------------
# Sequence encoding constants
# ---------------------------------------------------------------------------

DNA_SYMBOLS = 'ACGT'
AMINO_ACID_SYMBOLS = 'ACDEFGHIKLMNPQRSTVWY'
ENCODING_VERSION = 1


def _build_encoding_map(alphabet_type):
    symbols = DNA_SYMBOLS if alphabet_type == 'dna' else AMINO_ACID_SYMBOLS
    n = len(symbols)
    enc = {sym: (2 * k + 1 - n) / n for k, sym in enumerate(symbols)}
    if alphabet_type == 'dna':
        enc['U'] = enc['T']
    return enc


def _build_decoding_table(alphabet_type):
    symbols = DNA_SYMBOLS if alphabet_type == 'dna' else AMINO_ACID_SYMBOLS
    n = len(symbols)
    levels = [(2 * k + 1 - n) / n for k in range(n)]
    return symbols, levels


def detect_alphabet(sequence):
    unique = set(sequence.upper()) - {'N', '-', ' ', '\n', '\r'}
    if unique <= set('ACGTU'):
        return 'dna'
    return 'protein'


def read_fasta(path):
    header = ''
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
            elif line and not line.startswith('#'):
                seq_parts.append(line.upper())
    sequence = ''.join(seq_parts)
    # Strip characters that aren't part of the alphabet
    alpha = detect_alphabet(sequence)
    valid = set(DNA_SYMBOLS + 'U') if alpha == 'dna' else set(AMINO_ACID_SYMBOLS)
    cleaned = ''.join(c for c in sequence if c in valid)
    return header, cleaned


def parse_fasta_header(header):
    parts = header.split()
    name = parts[0] if parts else 'unknown'
    accession = None
    if '|' in name:
        segments = name.split('|')
        for seg in segments:
            if seg and seg[0].isalpha():
                accession = seg
                break
    elif '.' in name:
        accession = name.split('.')[0]
    else:
        accession = name
    return name, accession


# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------

def setup_complete_system(base_dir=None):
    print("Setting up complete system...")

    if base_dir is None:
        base_dir = Path.cwd() / "ScribblePlotter_Output"
    else:
        base_dir = Path(base_dir)

    gpu_available = False
    device_str = 'cpu'

    if TORCH_AVAILABLE:
        gpu_available = torch.cuda.is_available()
        print(f"GPU Available: {'YES' if gpu_available else 'NO'}")
        if gpu_available:
            device_str = 'cuda'
            print(f"GPU Type: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            print("Running on CPU (no GPU detected)")
    else:
        print("PyTorch not installed — running on CPU (numpy fallback)")

    directories = {
        'base': str(base_dir),
        'input': str(base_dir / 'input'),
        'output': str(base_dir / 'output'),
        'models': str(base_dir / 'models'),
        'config': str(base_dir / 'config'),
        'temp': str(base_dir / 'temp')
    }

    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)

    return {
        'gpu_available': gpu_available,
        'device': device_str,
        'directories': directories
    }


# ---------------------------------------------------------------------------
# Configuration — machine-specific keys are never loaded from disk
# ---------------------------------------------------------------------------

class CompleteConfiguration:
    MACHINE_SPECIFIC_KEYS = frozenset({
        'directories', 'device', 'gpu_enabled',
        'use_gpu_acceleration', 'use_gpu_ai', 'use_gpu_hopfield',
    })
    SESSION_ONLY_KEYS = frozenset({
        'steps_override', 'color_override', 'stroke_weight_override',
        'scribble_override', 'seed', 'smooth_strokes',
    })

    def __init__(self, system_info):
        self.system_info = system_info
        self.config_path = system_info['directories']['config']
        self.config_file = f"{self.config_path}/complete_config.json"

        self.defaults = {
            'project_name': 'GPU-Enhanced Scribble Plotter',
            'version': '4.0-local',
            'author': 'Kent Benson',
            'inspiration': '1983-1986 Hopfield Network Research',
            'gpu_enabled': system_info['gpu_available'],
            'device': system_info['device'],
            'directories': system_info['directories'],
            'total_examples': 3,
            'batch_size': 8,
            'use_gpu_acceleration': system_info['gpu_available'],
            'generate_pdf': True,
            'generate_dxf': True,
            'generate_png': True,
            'generate_svg': True,
            'organize_groups': True,
            'output_dpi': 300,
            'page_width': 800,
            'page_height': 600,
            'scale_factor_x': 0.09,
            'scale_factor_y': 0.09,
            'scale_factor_z': 1.0,
            'use_ai_parameters': True,
            'ai_confidence_threshold': 0.7,
            'use_gpu_ai': system_info['gpu_available'],
            'use_hopfield_memory': True,
            'hopfield_memory_size': 200,
            'hopfield_pattern_size': 15,
            'use_gpu_hopfield': system_info['gpu_available'],
            'create_random': True,
            'random_steps_upper_limit': 15,
            'scribble_upper_limit': 8,
            'steps_fixed': 5,
            'scribble_fixed': 3.0,
            'stroke_weight_value': 3.0,
            'gpu_memory_fraction': 0.8,
            'enable_mixed_precision': True,
            'optimize_memory': True
        }

        self.config = self.load_or_create_config()

    def load_or_create_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                for key in self.MACHINE_SPECIFIC_KEYS | self.SESSION_ONLY_KEYS:
                    loaded.pop(key, None)
                config = {**self.defaults, **loaded}
                print("Configuration loaded")
            else:
                config = self.defaults.copy()
                self.save_config(config)
                print("New configuration created")
            return config
        except Exception as e:
            print(f"Config error: {e}")
            return self.defaults.copy()

    def save_config(self, config=None):
        try:
            config_to_save = dict(config or self.config)
            for key in self.MACHINE_SPECIFIC_KEYS | self.SESSION_ONLY_KEYS:
                config_to_save.pop(key, None)
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Save config error: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def display_status(self):
        print("\nSYSTEM STATUS")
        print("=" * 40)
        print(f"Project: {self.get('project_name')}")
        print(f"Version: {self.get('version')}")
        print(f"GPU Enabled: {'YES' if self.get('gpu_enabled') else 'NO'}")
        print(f"Device: {self.get('device')}")
        print(f"AI Features: {'Enabled' if self.get('use_ai_parameters') else 'Disabled'}")
        print(f"Hopfield Networks: {'Enabled' if self.get('use_hopfield_memory') else 'Disabled'}")


# ---------------------------------------------------------------------------
# PLT Parser — unified HPGL parser
# ---------------------------------------------------------------------------

class PLTProcessor:
    _coord_re = re.compile(r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)')

    def __init__(self, config):
        self.config = config

    def process_plt_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            stripped = content.lstrip()
            if stripped and (stripped[0].isdigit() or stripped[0] == '-'):
                return self.process_raw_coordinates(content)
            else:
                return self.parse_hpgl(content)
        except Exception as e:
            print(f"PLT processing error: {e}")
            return []

    def process_raw_coordinates(self, content):
        coordinates = []
        content = content.replace('\n', '').replace('\r', '')
        for token in content.split(';'):
            token = token.strip()
            if not token:
                continue
            parts = token.split(',')
            if len(parts) >= 3:
                try:
                    coordinates.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
        return coordinates

    def parse_hpgl(self, content):
        coordinates = []
        pen_down = False

        content = content.replace('\n', '').replace('\r', '')
        commands = content.split(';')

        for cmd in commands:
            cmd = cmd.strip()
            if not cmd:
                continue

            if cmd.startswith('PU'):
                pen_down = False
                for x_str, y_str in self._coord_re.findall(cmd[2:]):
                    coordinates.append((float(x_str), float(y_str), 10.0))

            elif cmd.startswith('PD'):
                pen_down = True
                for x_str, y_str in self._coord_re.findall(cmd[2:]):
                    coordinates.append((float(x_str), float(y_str), 0.0))

            elif cmd.startswith('PA'):
                z_val = 0.0 if pen_down else 10.0
                for x_str, y_str in self._coord_re.findall(cmd[2:]):
                    coordinates.append((float(x_str), float(y_str), z_val))

        return coordinates

    def scale_points(self, coordinates):
        if not coordinates:
            return []

        scale_x = self.config.get('scale_factor_x', 0.09)
        scale_y = self.config.get('scale_factor_y', 0.09)
        scale_z = self.config.get('scale_factor_z', 1.0)

        return [(x * scale_x, y * scale_y, z * scale_z) for x, y, z in coordinates]


# ---------------------------------------------------------------------------
# CAD File Processor — reads DXF, DWG, DWX via ezdxf
# ---------------------------------------------------------------------------

class CADFileProcessor:
    SUPPORTED_EXTENSIONS = frozenset({'.dxf', '.dwg', '.dwx'})

    def __init__(self, config):
        self.config = config

    def process_cad_file(self, file_path):
        import ezdxf

        path = Path(file_path)
        ext = path.suffix.lower()

        try:
            if ext == '.dwg':
                doc = self._read_dwg(file_path)
            else:
                doc = ezdxf.readfile(file_path)
        except Exception as e:
            print(f"  CAD file error ({ext}): {e}")
            if ext == '.dwg':
                print("  Note: DWG conversion requires dwg2dxf (libredwg).")
                print("  Install: git clone https://github.com/LibreDWG/libredwg && cd libredwg && mkdir build && cd build && cmake .. && make dwg2dxf")
            return []

        return self._extract_coordinates(doc)

    def _read_dwg(self, file_path):
        import ezdxf
        import subprocess

        dwg2dxf_bin = shutil.which('dwg2dxf')
        if not dwg2dxf_bin:
            local_bin = Path.home() / '.local' / 'bin' / 'dwg2dxf'
            if local_bin.exists():
                dwg2dxf_bin = str(local_bin)

        if dwg2dxf_bin:
            tmp_path = Path(file_path).with_suffix('.___tmp_dxf___')
            env = os.environ.copy()
            lib_dir = str(Path.home() / '.local' / 'lib')
            env['LD_LIBRARY_PATH'] = lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
            try:
                result = subprocess.run(
                    [dwg2dxf_bin, '-o', str(tmp_path), str(file_path)],
                    capture_output=True, text=True, timeout=60, env=env)
                if tmp_path.exists():
                    doc = ezdxf.readfile(str(tmp_path))
                    tmp_path.unlink(missing_ok=True)
                    return doc
                print(f"  dwg2dxf conversion failed: {result.stderr.strip()}")
            except Exception as e:
                print(f"  dwg2dxf error: {e}")
                tmp_path.unlink(missing_ok=True)

        try:
            from ezdxf.addons import odafc
            tmp_path = Path(file_path).with_suffix('.___tmp_dxf___')
            odafc.convert(file_path, str(tmp_path))
            doc = ezdxf.readfile(str(tmp_path))
            tmp_path.unlink(missing_ok=True)
            return doc
        except (ImportError, Exception):
            pass

        raise RuntimeError("No DWG converter found (dwg2dxf or ODA File Converter)")

    def _extract_coordinates(self, doc):
        coordinates = []
        msp = doc.modelspace()

        for entity in msp:
            pts = self._entity_to_coords(entity)
            if not pts or len(pts) < 1:
                continue
            x0, y0 = pts[0]
            coordinates.append((x0, y0, 10.0))
            for x, y in pts:
                coordinates.append((x, y, 0.0))
            xn, yn = pts[-1]
            coordinates.append((xn, yn, 10.0))

        return coordinates

    def _entity_to_coords(self, entity):
        dxftype = entity.dxftype()

        if dxftype == 'LINE':
            s = entity.dxf.start
            e = entity.dxf.end
            return [(s.x, s.y), (e.x, e.y)]

        if dxftype == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in entity.get_points(format='xy')]
            if entity.closed and len(pts) > 1:
                pts.append(pts[0])
            return pts if len(pts) >= 2 else None

        if dxftype == 'POLYLINE':
            pts = [(v.dxf.location.x, v.dxf.location.y)
                   for v in entity.vertices]
            if entity.is_closed and len(pts) > 1:
                pts.append(pts[0])
            return pts if len(pts) >= 2 else None

        if dxftype == 'CIRCLE':
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r = entity.dxf.radius
            steps = max(36, int(r * 2))
            return [(cx + r * math.cos(2 * math.pi * i / steps),
                     cy + r * math.sin(2 * math.pi * i / steps))
                    for i in range(steps + 1)]

        if dxftype == 'ARC':
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r = entity.dxf.radius
            sa = math.radians(entity.dxf.start_angle)
            ea = math.radians(entity.dxf.end_angle)
            if ea <= sa:
                ea += 2 * math.pi
            steps = max(18, int((ea - sa) / (2 * math.pi) * 72))
            return [(cx + r * math.cos(sa + (ea - sa) * i / steps),
                     cy + r * math.sin(sa + (ea - sa) * i / steps))
                    for i in range(steps + 1)]

        if dxftype == 'ELLIPSE':
            try:
                cx, cy = entity.dxf.center.x, entity.dxf.center.y
                mx, my = entity.dxf.major_axis.x, entity.dxf.major_axis.y
                ratio = entity.dxf.ratio
                sp, ep = entity.dxf.start_param, entity.dxf.end_param
                if ep <= sp:
                    ep += 2 * math.pi
                steps = 72
                pts = []
                for i in range(steps + 1):
                    t = sp + (ep - sp) * i / steps
                    x = cx + mx * math.cos(t) - my * ratio * math.sin(t)
                    y = cy + my * math.cos(t) + mx * ratio * math.sin(t)
                    pts.append((x, y))
                return pts
            except Exception:
                return None

        if dxftype == 'SPLINE':
            try:
                pts = [(p.x, p.y) for p in entity.flattening(0.5)]
                return pts if len(pts) >= 2 else None
            except Exception:
                return None

        return None

    def scale_points(self, coordinates):
        if not coordinates:
            return []

        xs = [c[0] for c in coordinates]
        ys = [c[1] for c in coordinates]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        max_range = max(x_range, y_range, 1.0)

        norm = 10000.0 / max_range
        x_min, y_min = min(xs), min(ys)

        scale_x = self.config.get('scale_factor_x', 0.09)
        scale_y = self.config.get('scale_factor_y', 0.09)
        scale_z = self.config.get('scale_factor_z', 1.0)

        return [((x - x_min) * norm * scale_x,
                 (y - y_min) * norm * scale_y,
                 z * scale_z)
                for x, y, z in coordinates]


# ---------------------------------------------------------------------------
# GPU-Accelerated AI (unchanged, torch-optional)
# ---------------------------------------------------------------------------

class GPUAcceleratedAI:
    def __init__(self, config):
        self.config = config
        self.gpu_available = config.get('gpu_enabled', False) and TORCH_AVAILABLE

        if self.gpu_available:
            self.device = torch.device(config.get('device', 'cpu'))
            print("Initializing GPU-accelerated AI...")
            torch.cuda.empty_cache()
        else:
            self.device = None
            print("Initializing CPU AI...")

    def extract_features_gpu(self, points):
        try:
            if not points:
                return np.zeros(15)

            coords = np.array([(p[0], p[1]) for p in points if p[2] == 0])
            if len(coords) < 2:
                return np.zeros(15)

            if self.gpu_available:
                coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
                features = self._gpu_feature_calculation(coords_tensor)
                return features.cpu().numpy()
            else:
                return self._cpu_feature_calculation(coords)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(15)

    def _gpu_feature_calculation(self, coords_tensor):
        features = []

        if len(coords_tensor) > 2:
            directions = coords_tensor[1:] - coords_tensor[:-1]
            angles = torch.atan2(directions[:, 1], directions[:, 0])
            angle_changes = torch.abs(angles[1:] - angles[:-1])
            complexity = torch.mean(angle_changes) / math.pi
        else:
            complexity = torch.tensor(0.0, device=self.device)
        features.append(complexity)

        if len(coords_tensor) > 1:
            diffs = coords_tensor[1:] - coords_tensor[:-1]
            distances = torch.sqrt(torch.sum(diffs**2, dim=1))
            total_length = torch.sum(distances)
        else:
            total_length = torch.tensor(0.0, device=self.device)
        features.append(total_length)

        if len(coords_tensor) > 0:
            min_coords = torch.min(coords_tensor, dim=0)[0]
            max_coords = torch.max(coords_tensor, dim=0)[0]
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            aspect_ratio = width / torch.clamp(height, min=1e-6)
            features.extend([width, height, aspect_ratio])
        else:
            features.extend([torch.tensor(0.0, device=self.device)] * 3)

        area = features[2] * features[3] if len(features) > 3 else torch.tensor(1.0, device=self.device)
        density = len(coords_tensor) / torch.clamp(area, min=1e-6)
        features.append(density)

        if len(coords_tensor) > 0:
            centroid = torch.mean(coords_tensor, dim=0)
            distances = torch.sqrt(torch.sum((coords_tensor - centroid)**2, dim=1))
            spread = torch.std(distances)
            features.extend([centroid[0], centroid[1], spread])
        else:
            features.extend([torch.tensor(0.0, device=self.device)] * 3)

        while len(features) < 15:
            features.append(torch.tensor(0.5, device=self.device))

        return torch.stack(features[:15])

    def _cpu_feature_calculation(self, coords):
        features = [0.5] * 15

        if len(coords) > 2:
            diffs = np.diff(coords, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_changes = np.abs(np.diff(angles))
            features[0] = float(np.mean(angle_changes) / math.pi)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            features[1] = float(np.sum(distances))
        elif len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            features[1] = float(np.sum(distances))

        if len(coords) > 1:
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            features[2] = float(max_coords[0] - min_coords[0])
            features[3] = float(max_coords[1] - min_coords[1])
            features[4] = features[2] / max(features[3], 1e-6)

        if len(coords) > 0:
            area = max(features[2] * features[3], 1e-6)
            features[5] = len(coords) / area
            centroid = np.mean(coords, axis=0)
            features[6] = float(centroid[0])
            features[7] = float(centroid[1])
            if len(coords) > 1:
                dists = np.sqrt(np.sum((coords - centroid)**2, axis=1))
                features[8] = float(np.std(dists))

        return np.array(features)

    def predict_parameters(self, points):
        try:
            features = self.extract_features_gpu(points)

            complexity = float(features[0])
            path_length = float(features[1])
            aspect_ratio = float(features[4]) if len(features) > 4 else 1.0
            density = float(features[5]) if len(features) > 5 else 0.0

            steps = int(3 + 12 * complexity)
            scribble = 1.0 + 6.0 * (1.0 - complexity)

            length_factor = min(path_length / 1000.0, 3.0)
            scribble *= 0.8 + 0.4 * min(length_factor, 1.0)
            steps += int(2 * max(length_factor - 1.0, 0.0))

            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                scribble *= 1.1

            if density > 0.01:
                scribble *= max(0.6, 1.0 - density * 0.3)

            stroke_weight = self.config.get('stroke_weight_value', 3.0)
            color = (0.0, 0.0, 0.0)

            return {
                'steps': max(2, min(20, steps)),
                'scribble': max(0.1, min(10.0, scribble)),
                'stroke_weight': max(0.5, min(8.0, stroke_weight)),
                'color': color,
                'complexity_score': complexity,
                'generation_method': 'gpu_ai' if self.gpu_available else 'cpu_ai'
            }
        except Exception as e:
            print(f"Parameter prediction error: {e}")
            return {
                'steps': 5,
                'scribble': 4.0,
                'stroke_weight': self.config.get('stroke_weight_value', 3.0),
                'color': (0.0, 0.0, 0.0),
                'complexity_score': 0.5,
                'generation_method': 'fallback'
            }


# ---------------------------------------------------------------------------
# GPU Hopfield Network (unchanged, torch-optional)
# ---------------------------------------------------------------------------

class GPUHopfieldNetwork:
    def __init__(self, config):
        self.config = config
        self.gpu_available = config.get('gpu_enabled', False) and TORCH_AVAILABLE

        self.pattern_size = config.get('hopfield_pattern_size', 15)
        self.memory_size = config.get('hopfield_memory_size', 200)

        if self.gpu_available:
            self.device = torch.device(config.get('device', 'cpu'))
            self.weights = torch.zeros(self.pattern_size, self.pattern_size, device=self.device)
            print("GPU Hopfield network initialized")
        else:
            self.device = None
            self.weights = np.zeros((self.pattern_size, self.pattern_size))
            print("CPU Hopfield network initialized")

        self.stored_patterns = []
        self.pattern_labels = []

    def store_pattern(self, pattern, label=""):
        try:
            if len(self.stored_patterns) >= self.memory_size:
                print(f"Memory full ({self.memory_size} patterns)")
                return False

            if self.gpu_available:
                norm_pattern = self._normalize_pattern_gpu(pattern)
                self.stored_patterns.append(norm_pattern.cpu().numpy())
                self.pattern_labels.append(label)
                self.weights += torch.outer(norm_pattern, norm_pattern)
            else:
                norm_pattern = self._normalize_pattern_cpu(pattern)
                self.stored_patterns.append(norm_pattern)
                self.pattern_labels.append(label)
                self.weights += np.outer(norm_pattern, norm_pattern)

            print(f"Stored pattern '{label}' ({len(self.stored_patterns)}/{self.memory_size})")
            return True
        except Exception as e:
            print(f"Pattern storage error: {e}")
            return False

    def recall_pattern(self, partial_pattern, max_iterations=50):
        try:
            if self.gpu_available:
                return self._gpu_recall(partial_pattern, max_iterations)
            else:
                return self._cpu_recall(partial_pattern, max_iterations)
        except Exception as e:
            print(f"Pattern recall error: {e}")
            return partial_pattern

    def _gpu_recall(self, partial_pattern, max_iterations):
        current = self._normalize_pattern_gpu(partial_pattern)
        for _ in range(max_iterations):
            previous = current.clone()
            activations = torch.mv(self.weights, current)
            current = torch.tanh(activations)
            if torch.allclose(current, previous, atol=1e-6):
                break
        return self._denormalize_pattern_gpu(current)

    def _cpu_recall(self, partial_pattern, max_iterations):
        current = self._normalize_pattern_cpu(partial_pattern)
        for _ in range(max_iterations):
            previous = current.copy()
            activations = self.weights @ current
            current = np.tanh(activations)
            if np.allclose(current, previous, atol=1e-6):
                break
        return self._denormalize_pattern_cpu(current)

    def find_spurious_memories(self, test_patterns):
        spurious_memories = []
        for test_pattern in test_patterns:
            recalled = self.recall_pattern(test_pattern)
            is_stored = any(self._patterns_similar(recalled, s) for s in self.stored_patterns)
            if not is_stored:
                spurious_memories.append(recalled)
        return spurious_memories

    def _patterns_similar(self, pattern1, pattern2, threshold=0.9):
        if len(pattern1) != len(pattern2):
            return False
        p1 = pattern1.cpu().numpy() if hasattr(pattern1, 'cpu') else np.asarray(pattern1)
        p2 = pattern2.cpu().numpy() if hasattr(pattern2, 'cpu') else np.asarray(pattern2)
        correlation = np.corrcoef(p1, p2)[0, 1]
        return abs(correlation) > threshold

    def _normalize_pattern_gpu(self, pattern):
        pt = torch.tensor(pattern[:self.pattern_size], dtype=torch.float32, device=self.device)
        if len(pt) < self.pattern_size:
            pt = torch.cat([pt, torch.zeros(self.pattern_size - len(pt), device=self.device)])
        if torch.std(pt) > 0:
            pt = (pt - torch.mean(pt)) / torch.std(pt)
        return pt

    def _denormalize_pattern_gpu(self, pattern):
        return ((pattern + 1) / 2).cpu().numpy()

    def _normalize_pattern_cpu(self, pattern):
        p = np.array(pattern[:self.pattern_size], dtype=float)
        if len(p) < self.pattern_size:
            p = np.pad(p, (0, self.pattern_size - len(p)))
        if np.std(p) > 0:
            p = (p - np.mean(p)) / np.std(p)
        return p

    def _denormalize_pattern_cpu(self, pattern):
        return (pattern + 1) / 2


# ---------------------------------------------------------------------------
# Scribble geometry generator — produces wobbled paths once for all exporters
# ---------------------------------------------------------------------------

class ScribbleGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, points, params, sequence=None, seq_info=None):
        """Generate scribbled strokes from point data.

        Returns (strokes, encoding_info).
        strokes: list of strokes, each a list of (x, y) waypoints.
        encoding_info: dict with sequence metadata, or None.
        """
        steps = params['steps']
        amplitude = params['scribble']

        encoding_map = None
        alphabet_type = None
        if sequence:
            alphabet_type = detect_alphabet(sequence)
            encoding_map = _build_encoding_map(alphabet_type)
            fallback_rng = random.Random(
                int.from_bytes(hashlib.sha256(sequence.encode()).digest()[:8], 'big'))
        else:
            fallback_rng = None

        strokes = []
        current_stroke = []
        seq_idx = 0
        stroke_seq_ranges = []
        stroke_seq_start = 0

        for i in range(len(points) - 1):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[i + 1]

            if z2 == 0.0:
                if not current_stroke:
                    current_stroke = [(x1, y1)]
                    stroke_seq_start = seq_idx

                waypoints, seq_idx = self._scribble_segment(
                    x1, y1, x2, y2, steps, amplitude,
                    sequence, seq_idx, encoding_map, fallback_rng)

                current_stroke.extend(waypoints[1:])
            else:
                if current_stroke:
                    strokes.append(current_stroke)
                    stroke_seq_ranges.append((stroke_seq_start, seq_idx))
                    current_stroke = []

        if current_stroke:
            strokes.append(current_stroke)
            stroke_seq_ranges.append((stroke_seq_start, seq_idx))

        if not sequence and self.config.get('smooth_strokes', True):
            strokes = [self._smooth_stroke(s) for s in strokes]

        encoding_info = None
        if sequence:
            capacity = self._count_capacity(points, steps)
            embedded = min(seq_idx, len(sequence))
            encoding_info = {
                'embedded_count': embedded,
                'total_bases': len(sequence),
                'capacity': capacity,
                'stroke_seq_ranges': stroke_seq_ranges,
                'alphabet': alphabet_type,
                'steps': steps,
                'amplitude': amplitude,
                'version': ENCODING_VERSION,
            }
            if seq_info:
                encoding_info.update(seq_info)

        return strokes, encoding_info

    def _scribble_segment(self, x1, y1, x2, y2, steps, amplitude,
                          sequence, seq_idx, encoding_map, fallback_rng):
        waypoints = [(x1, y1)]

        if steps < 1:
            waypoints.append((x2, y2))
            return waypoints, seq_idx

        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)

        if seg_len < 1e-10:
            return waypoints, seq_idx

        if sequence is not None and encoding_map is not None:
            perp_x = -dy / seg_len
            perp_y = dx / seg_len

            for step_num in range(1, steps):
                t = step_num / steps
                ideal_x = x1 + t * dx
                ideal_y = y1 + t * dy

                if seq_idx < len(sequence):
                    base = sequence[seq_idx]
                    level = encoding_map.get(base, 0.0)
                    point_x = ideal_x + level * amplitude * perp_x
                    point_y = ideal_y + level * amplitude * perp_y
                    seq_idx += 1
                else:
                    offset = fallback_rng.uniform(-amplitude, amplitude)
                    point_x = ideal_x + offset * perp_x
                    point_y = ideal_y + offset * perp_y

                waypoints.append((point_x, point_y))

            waypoints.append((x2, y2))
        else:
            perp_x = -dy / seg_len
            perp_y = dx / seg_len

            for step_num in range(1, steps):
                t = step_num / steps
                ideal_x = x1 + t * dx
                ideal_y = y1 + t * dy
                offset = random.uniform(-amplitude, amplitude)
                point_x = ideal_x + offset * perp_x
                point_y = ideal_y + offset * perp_y
                waypoints.append((point_x, point_y))

            waypoints.append((x2, y2))

        return waypoints, seq_idx

    def _smooth_stroke(self, waypoints, density=3):
        if len(waypoints) < 4 or not SCIPY_AVAILABLE:
            return waypoints

        xs = np.array([p[0] for p in waypoints])
        ys = np.array([p[1] for p in waypoints])

        diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        t = np.concatenate([[0], np.cumsum(diffs)])
        total = t[-1]
        if total < 1e-10:
            return waypoints

        t /= total

        cs_x = CubicSpline(t, xs, bc_type='clamped')
        cs_y = CubicSpline(t, ys, bc_type='clamped')

        n_out = max(len(waypoints) * density, 8)
        t_new = np.linspace(0, 1, n_out)
        smoothed = list(zip(cs_x(t_new).tolist(), cs_y(t_new).tolist()))
        smoothed[0] = waypoints[0]
        smoothed[-1] = waypoints[-1]
        return smoothed

    def _count_capacity(self, points, steps):
        count = 0
        for i in range(len(points) - 1):
            _, _, z2 = points[i + 1]
            if z2 == 0.0:
                x1, y1, _ = points[i]
                x2, y2, _ = points[i + 1]
                if math.hypot(x2 - x1, y2 - y1) >= 1e-10:
                    count += steps - 1
        return count


# ---------------------------------------------------------------------------
# Scribble renderer — draws pre-generated strokes via matplotlib
# ---------------------------------------------------------------------------

class ScribbleRenderer:
    def __init__(self, config):
        self.config = config

    def render_from_strokes(self, strokes, params):
        if not strokes:
            return None

        all_x = [p[0] for s in strokes for p in s]
        all_y = [p[1] for s in strokes for p in s]
        if not all_x:
            return None

        draw_w = max(all_x) - min(all_x)
        draw_h = max(all_y) - min(all_y)
        max_inches = 10.0
        if draw_w >= draw_h:
            fig_width = max_inches
            fig_height = max(max_inches * draw_h / max(draw_w, 1e-6), 2.0)
        else:
            fig_height = max_inches
            fig_width = max(max_inches * draw_w / max(draw_h, 1e-6), 2.0)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        ax.axis('off')

        color = params.get('color', (0, 0, 0))
        stroke_weight = params.get('stroke_weight', 3.0)
        lw = stroke_weight / 5

        for stroke in strokes:
            xs = [p[0] for p in stroke]
            ys = [p[1] for p in stroke]
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.8)

        pad = max(draw_w, draw_h) * 0.03
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(max(all_y) + pad, min(all_y) - pad)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# SVG exporter — writes paths directly, embeds sequence metadata
# ---------------------------------------------------------------------------

class SVGExporter:
    def export(self, strokes, output_path, params, encoding_info=None):
        all_x = [p[0] for s in strokes for p in s]
        all_y = [p[1] for s in strokes for p in s]
        if not all_x:
            return False

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        pad = max(max_x - min_x, max_y - min_y) * 0.05
        vb_x = min_x - pad
        vb_w = (max_x - min_x) + 2 * pad
        vb_h = (max_y - min_y) + 2 * pad

        color = params.get('color', (0, 0, 0))
        stroke_weight = params.get('stroke_weight', 3.0)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        lw = stroke_weight / 5

        lines = []
        lines.append('<?xml version="1.0" encoding="utf-8"?>')
        lines.append('<svg xmlns="http://www.w3.org/2000/svg"')
        lines.append(f'     viewBox="{vb_x:.6f} {-(max_y + pad):.6f} {vb_w:.6f} {vb_h:.6f}"')
        lines.append(f'     width="{vb_w:.1f}" height="{vb_h:.1f}">')

        if encoding_info:
            lines.append('  <metadata>')
            ei = encoding_info
            lines.append(f'    <scribble-encoding xmlns="urn:scribble-plotter:encoding:{ei["version"]}"')
            lines.append(f'        version="{ei["version"]}"')
            lines.append('        method="perpendicular-displacement"')
            lines.append(f'        alphabet="{ei["alphabet"]}"')
            lines.append(f'        amplitude="{ei["amplitude"]:.10f}"')
            lines.append(f'        steps="{ei["steps"]}">')
            seq_name = ei.get('sequence_name', 'unknown')
            accession = ei.get('accession', '')
            acc_attr = f' accession="{accession}"' if accession else ''
            lines.append(f'      <sequence name="{seq_name}"{acc_attr}'
                         f' length="{ei["total_bases"]}"'
                         f' embedded="{ei["embedded_count"]}" />')
            lines.append('    </scribble-encoding>')
            lines.append('  </metadata>')

        lines.append(f'  <g stroke="{hex_color}" stroke-width="{lw:.3f}"'
                     f' fill="none" stroke-linecap="round" stroke-linejoin="round">')

        stroke_ranges = encoding_info['stroke_seq_ranges'] if encoding_info else []

        for s_idx, stroke in enumerate(strokes):
            path_parts = []
            for j, (px, py) in enumerate(stroke):
                cmd = 'M' if j == 0 else 'L'
                path_parts.append(f'{cmd}{px:.6f},{-py:.6f}')

            data_attrs = ''
            if encoding_info and s_idx < len(stroke_ranges):
                s_start, s_end = stroke_ranges[s_idx]
                if s_end > s_start:
                    data_attrs = (f' data-seq-start="{s_start}"'
                                  f' data-seq-end="{s_end - 1}"')

            lines.append(f'    <path d="{" ".join(path_parts)}"{data_attrs} />')

        lines.append('  </g>')
        lines.append('</svg>')

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        return True


# ---------------------------------------------------------------------------
# Directory manager (unchanged)
# ---------------------------------------------------------------------------

class DirectoryManager:
    GROUP_DIRS = ['GROUP_PDF', 'GROUP_DXF', 'GROUP_PNG', 'GROUP_SVG',
                  'GROUP_TIFF', 'GROUP_TRANSFORM']

    def __init__(self, config):
        self.config = config
        self.processed_files = []
        self.file_registry = {}
        print("Directory manager initialized")

    def create_working_directories(self, base_dir, file_base_name, file_index):
        try:
            working_dir = Path(base_dir) / f"{file_base_name}_{file_index}"

            directories = {
                'pdf': working_dir / 'PDF',
                'dxf': working_dir / 'DXF',
                'png': working_dir / 'PNG',
                'svg': working_dir / 'SVG',
                'tmp': working_dir / 'TMP',
            }

            for gdir in self.GROUP_DIRS:
                directories[gdir.lower()] = Path(base_dir) / gdir

            for name, path in directories.items():
                path.mkdir(parents=True, exist_ok=True)

            return {name: str(path) for name, path in directories.items()}
        except Exception as e:
            print(f"Directory creation error: {e}")
            return {}

    def register_file(self, file_path, file_type):
        self.processed_files.append(file_path)
        self.file_registry[file_path] = {
            'type': file_type,
            'timestamp': datetime.now()
        }

    def organize_output_files(self):
        if not self.config.get('organize_groups', True):
            return

        try:
            base_output = Path(self.config.get('directories')['output'])
            copied_count = 0

            for file_path in self.processed_files:
                source = Path(file_path)
                if not source.exists():
                    continue

                ext = source.suffix.lower()
                group_map = {
                    '.pdf': 'GROUP_PDF',
                    '.dxf': 'GROUP_DXF',
                    '.png': 'GROUP_PNG',
                    '.svg': 'GROUP_SVG',
                    '.tiff': 'GROUP_TIFF',
                    '.tif': 'GROUP_TIFF',
                }

                group_name = group_map.get(ext)
                if not group_name:
                    continue

                group_dir = base_output / group_name
                group_dir.mkdir(exist_ok=True)

                parent_name = source.parent.parent.name
                dest_name = f"{parent_name}_{source.name}"
                destination = group_dir / dest_name

                shutil.copy2(str(source), str(destination))
                copied_count += 1

            print(f"Organized {copied_count} files into GROUP directories")
        except Exception as e:
            print(f"Error organizing files: {e}")


# ---------------------------------------------------------------------------
# Processing system — generates geometry once, exports to all formats
# ---------------------------------------------------------------------------

class CompleteProcessingSystem:
    def __init__(self, config):
        self.config = config
        self.plt_processor = PLTProcessor(config)
        self.cad_processor = CADFileProcessor(config)
        self.gpu_ai = GPUAcceleratedAI(config)
        self.hopfield_network = GPUHopfieldNetwork(config)
        self.renderer = ScribbleRenderer(config)
        self.generator = ScribbleGenerator(config)
        self.svg_exporter = SVGExporter()
        self.directory_manager = DirectoryManager(config)

        self.processed_files = []
        self.error_files = []
        self.sequence = None
        self.seq_info = None

        print("Complete processing system initialized")

    def set_sequence(self, fasta_path):
        header, sequence = read_fasta(fasta_path)
        name, accession = parse_fasta_header(header)
        self.sequence = sequence
        self.seq_info = {
            'sequence_name': name,
            'accession': accession,
            'fasta_header': header,
        }
        alpha = detect_alphabet(sequence)
        print(f"  Loaded sequence: {name} ({len(sequence)} {'bases' if alpha == 'dna' else 'residues'})")
        if accession:
            print(f"  Accession: {accession}")

    def get_input_files(self, directory):
        try:
            path = Path(directory)
            if not path.exists():
                print(f"Directory not found: {directory}")
                return []

            exts = ['plt', 'dxf', 'dwg', 'dwx']
            found = []
            for ext in exts:
                found.extend(path.glob(f"*.{ext}"))
                found.extend(path.glob(f"*.{ext.upper()}"))
            found = sorted(set(found))

            by_ext = {}
            for f in found:
                e = f.suffix.lower()
                by_ext[e] = by_ext.get(e, 0) + 1
            breakdown = ', '.join(f"{v} {k}" for k, v in sorted(by_ext.items()))
            print(f"Found {len(found)} input files ({breakdown})")
            return found
        except Exception as e:
            print(f"Error scanning input files: {e}")
            return []

    def get_plt_files(self, directory):
        return self.get_input_files(directory)

    def process_single_file(self, plt_file, iteration=0):
        try:
            print(f"\nProcessing: {plt_file.name} (iteration {iteration + 1})")

            ext = Path(plt_file).suffix.lower()
            if ext in CADFileProcessor.SUPPORTED_EXTENSIONS:
                raw_coords = self.cad_processor.process_cad_file(str(plt_file))
                processor = self.cad_processor
            else:
                raw_coords = self.plt_processor.process_plt_file(str(plt_file))
                processor = self.plt_processor

            if not raw_coords:
                print(f"  Failed to process {ext} file")
                return False

            points = processor.scale_points(raw_coords)
            pu_count = sum(1 for _, _, z in points if z != 0)
            pd_count = sum(1 for _, _, z in points if z == 0)
            print(f"  Loaded {len(points)} points (PD={pd_count}, PU={pu_count})")

            if self.config.get('use_ai_parameters'):
                params = self.gpu_ai.predict_parameters(points)
                print(f"  AI Parameters: steps={params['steps']}, scribble={params['scribble']:.1f}")

                if self.config.get('use_hopfield_memory'):
                    features = self.gpu_ai.extract_features_gpu(points)
                    recalled = self.hopfield_network.recall_pattern(features)

                    blend = 0.7
                    r_complexity = max(0.0, min(1.0, float(recalled[0])))
                    mem_steps = 3 + 12 * r_complexity
                    mem_scribble = 1.0 + 6.0 * (1.0 - r_complexity)
                    params['steps'] = max(2, min(20, int(
                        blend * params['steps'] + (1 - blend) * mem_steps)))
                    params['scribble'] = max(0.1, min(10.0,
                        blend * params['scribble'] + (1 - blend) * mem_scribble))
                    r_spread = max(0.0, min(1.0, float(recalled[8]))) if len(recalled) > 8 else 0.5
                    mem_weight = 1.5 + 4.0 * r_spread
                    params['stroke_weight'] = max(0.5, min(8.0,
                        blend * params['stroke_weight'] + (1 - blend) * mem_weight))
                    print(f"  Hopfield Enhanced: steps={params['steps']},"
                          f" scribble={params['scribble']:.1f},"
                          f" weight={params['stroke_weight']:.1f}")
            else:
                params = {
                    'steps': self.config.get('steps_fixed', 5),
                    'scribble': self.config.get('scribble_fixed', 3.0),
                    'stroke_weight': self.config.get('stroke_weight_value', 3.0),
                    'color': (0.0, 0.0, 0.0),
                    'generation_method': 'traditional'
                }

            # CLI overrides apply on top of any parameter source
            if self.config.get('steps_override'):
                params['steps'] = self.config.get('steps_override')
            color_ovr = self.config.get('color_override')
            if color_ovr is not None:
                params['color'] = color_ovr
            scribble_ovr = self.config.get('scribble_override')
            if scribble_ovr is not None:
                params['scribble'] = scribble_ovr
            sw_ovr = self.config.get('stroke_weight_override')
            if sw_ovr is not None:
                params['stroke_weight'] = sw_ovr

            # Generate scribbled geometry ONCE
            strokes, encoding_info = self.generator.generate(
                points, params, self.sequence, self.seq_info)

            if encoding_info:
                print(f"  Sequence encoding: {encoding_info['embedded_count']}"
                      f"/{encoding_info['total_bases']} bases embedded"
                      f" (capacity: {encoding_info['capacity']})")
                if encoding_info['embedded_count'] < encoding_info['total_bases']:
                    print("  Warning: sequence truncated — drawing has fewer"
                          " encodable positions than the sequence length."
                          " Increase --steps to fit more.")
                elif encoding_info['embedded_count'] < encoding_info['capacity']:
                    print("  Note: sequence shorter than capacity — remaining"
                          f" {encoding_info['capacity'] - encoding_info['embedded_count']}"
                          " positions use random wobble.")

            if not strokes:
                print("  No drawable strokes found")
                return False

            base_name = plt_file.stem
            output_base = Path(self.config.get('directories')['output'])

            dirs = self.directory_manager.create_working_directories(
                str(output_base), base_name, iteration)

            output_dirs = {
                'pdf': Path(dirs.get('pdf', output_base / f"{base_name}_{iteration}" / 'PDF')),
                'dxf': Path(dirs.get('dxf', output_base / f"{base_name}_{iteration}" / 'DXF')),
                'png': Path(dirs.get('png', output_base / f"{base_name}_{iteration}" / 'PNG')),
                'svg': Path(dirs.get('svg', output_base / f"{base_name}_{iteration}" / 'SVG')),
            }

            method_suffix = params.get('generation_method', 'ai')
            output_name = (f"{base_name}_{method_suffix}_steps{params['steps']}"
                           f"_scribble{int(params['scribble'])}_{iteration}")

            success_count = 0

            # Render figure from pre-generated strokes
            figure = self.renderer.render_from_strokes(strokes, params)
            if not figure:
                print("  Failed to render artwork")
                return False

            if self.config.get('generate_png'):
                png_path = output_dirs['png'] / f"{output_name}.png"
                try:
                    figure.savefig(str(png_path), format='png',
                                   dpi=self.config.get('output_dpi', 150),
                                   bbox_inches='tight', facecolor='white')
                    self.directory_manager.register_file(str(png_path), 'png')
                    success_count += 1
                    print(f"  PNG saved: {png_path}")
                except Exception as e:
                    print(f"  PNG error: {e}")

            if self.config.get('generate_pdf'):
                pdf_path = output_dirs['pdf'] / f"{output_name}.pdf"
                try:
                    figure.savefig(str(pdf_path), format='pdf',
                                   dpi=self.config.get('output_dpi', 300),
                                   bbox_inches='tight', facecolor='white')
                    self.directory_manager.register_file(str(pdf_path), 'pdf')
                    success_count += 1
                    print(f"  PDF saved: {pdf_path}")
                except Exception as e:
                    print(f"  PDF error: {e}")

            plt.close(figure)

            if self.config.get('generate_dxf'):
                dxf_path = output_dirs['dxf'] / f"{output_name}.dxf"
                try:
                    self.save_dxf(strokes, str(dxf_path))
                    self.directory_manager.register_file(str(dxf_path), 'dxf')
                    success_count += 1
                    print(f"  DXF saved: {dxf_path}")
                except Exception as e:
                    print(f"  DXF error: {e}")

            if self.config.get('generate_svg'):
                svg_path = output_dirs['svg'] / f"{output_name}.svg"
                try:
                    self.svg_exporter.export(strokes, str(svg_path), params, encoding_info)
                    self.directory_manager.register_file(str(svg_path), 'svg')
                    success_count += 1
                    print(f"  SVG saved: {svg_path}")
                except Exception as e:
                    print(f"  SVG error: {e}")

            if success_count > 0 and self.config.get('use_hopfield_memory'):
                features = self.gpu_ai.extract_features_gpu(points)
                style_name = f"{base_name}_{iteration}"
                self.hopfield_network.store_pattern(features, style_name)

            if success_count > 0:
                self.processed_files.append(output_name)
                print(f"  Generated {success_count} format(s) successfully")
                return True
            else:
                print("  Failed to save any formats")
                return False

        except Exception as e:
            print(f"  Processing error: {e}")
            import traceback
            traceback.print_exc()
            self.error_files.append(str(plt_file))
            return False

    def save_dxf(self, strokes, output_path):
        import ezdxf
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x1, y1 = stroke[i]
                x2, y2 = stroke[i + 1]
                msp.add_line((x1, y1), (x2, y2))

        doc.saveas(output_path)
        return True

    def process_batch(self, input_directory):
        print("BATCH PROCESSING")
        print("=" * 50)

        plt_files = self.get_input_files(input_directory)
        if not plt_files:
            return {'success': False, 'message': 'No input files found (PLT/DXF/DWG/DWX)'}

        total_examples = self.config.get('total_examples', 3)
        total_operations = len(plt_files) * total_examples

        print(f"Processing {len(plt_files)} files x {total_examples}"
              f" examples = {total_operations} operations")

        if self.config.get('gpu_enabled'):
            print("Using GPU acceleration!")

        successful_operations = 0

        with tqdm(total=total_operations, desc="Processing") as pbar:
            for plt_file in plt_files:
                for iteration in range(total_examples):
                    success = self.process_single_file(plt_file, iteration)
                    if success:
                        successful_operations += 1
                    pbar.update(1)
                    pbar.set_description(f"Success: {successful_operations}/{total_operations}")

        summary = {
            'success': True,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': f"{successful_operations}/{total_operations}",
            'processed_files': len(self.processed_files),
            'error_files': len(self.error_files),
            'hopfield_patterns': len(self.hopfield_network.stored_patterns),
            'gpu_accelerated': self.config.get('gpu_enabled'),
            'output_directory': self.config.get('directories')['output']
        }

        self.directory_manager.organize_output_files()

        if self.config.get('organize_groups', True):
            output_dir = self.config.get('directories')['output']
            show_directory_structure(output_dir)

        self.print_summary(summary)
        return summary

    def print_summary(self, summary):
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total operations: {summary['total_operations']}")
        print(f"Successful: {summary['successful_operations']}")
        print(f"Success rate: {summary['success_rate']}")
        print(f"Hopfield patterns learned: {summary['hopfield_patterns']}")
        print(f"GPU accelerated: {'YES' if summary['gpu_accelerated'] else 'NO'}")
        print(f"Output directory: {summary['output_directory']}")
        if summary['error_files'] > 0:
            print(f"\nFiles with errors: {summary['error_files']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def quick_test(config, processing_system):
    print("\nQUICK SYSTEM TEST")
    print("=" * 30)

    print(f"1. Configuration: {'OK' if config else 'FAILED'}")
    print(f"2. GPU Available: {'YES' if config.get('gpu_enabled') else 'NO'}")
    print(f"3. PyTorch: {'Available' if TORCH_AVAILABLE else 'Not installed (numpy fallback)'}")
    print(f"4. Processing System: {'OK' if processing_system else 'FAILED'}")

    input_dir = Path(config.get('directories')['input'])
    print(f"5. Input Directory: {'EXISTS' if input_dir.exists() else 'MISSING'}")

    if input_dir.exists():
        input_files = processing_system.get_input_files(str(input_dir))
        print(f"6. Input Files (PLT/DXF/DWG/DWX): {len(input_files) if input_files else 'NONE'}")


def gpu_status():
    if not TORCH_AVAILABLE:
        print("\nGPU STATUS: PyTorch not installed")
        print("Install with: pip install torch")
        return

    if torch.cuda.is_available():
        print("\nGPU STATUS: AVAILABLE")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Used: {allocated:.2f} GB ({allocated / total * 100:.1f}%)")
    else:
        print("\nGPU STATUS: NOT AVAILABLE")
        print("Running on CPU")


def hopfield_demo(processing_system):
    print("\nHOPFIELD NETWORK DEMO")
    print("Kent Benson's 1986 vision of neural network art control")
    print("=" * 50)

    demo_net = processing_system.hopfield_network
    artistic_pattern = [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2]
    demo_net.store_pattern(artistic_pattern, "demo_style")
    partial = [0.8, 0.2, 0.0, 0.0, 0.7, 0.0, 0.6, 0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]
    recalled = demo_net.recall_pattern(partial)
    print(f"Partial input: {[f'{x:.1f}' for x in partial[:8]]}")
    print(f"Network recall: {[f'{x:.1f}' for x in recalled[:8]]}")
    print("\nThis shows how the network completes partial artistic patterns!")


def show_directory_structure(base_path, max_depth=3):
    base = Path(base_path)
    if not base.exists():
        print(f"Directory not found: {base_path}")
        return

    print("\nDIRECTORY STRUCTURE")
    print("=" * 30)
    print(f"{base.name}/")

    visited_inodes = set()

    def print_tree(path, prefix="", current_depth=0):
        if current_depth >= max_depth:
            return
        try:
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in visited_inodes:
                return
            visited_inodes.add(inode)
        except OSError:
            return
        try:
            items = sorted(path.iterdir())
        except PermissionError:
            return
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{item.name}")
            if item.is_dir() and current_depth + 1 < max_depth:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, current_depth + 1)

    print_tree(base)


def create_sample_plt_files(output_dir):
    input_dir = Path(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    plt_content_1 = """IN;
SP1;
PA0,0;
PD;
PA100,0;
PA100,100;
PA0,100;
PA0,0;
PU;
SP0;"""

    plt_content_2 = """IN;
SP1;
PA50,0;
PD;
PA70,14;
PA85,35;
PA92,57;
PA85,79;
PA70,100;
PA50,107;
PA30,100;
PA15,79;
PA8,57;
PA15,35;
PA30,14;
PA50,0;
PU;
SP0;"""

    plt_content_3 = """IN;
SP1;
PA20,20;
PD;
PA80,20;
PA80,80;
PA20,80;
PA20,20;
PU;
PA30,30;
PD;
PA70,30;
PA70,70;
PA30,70;
PA30,30;
PU;
PA40,40;
PD;
PA60,60;
PU;
PA60,40;
PD;
PA40,60;
PU;
SP0;"""

    sample_files = [
        ('sample_rectangle.plt', plt_content_1),
        ('sample_circle.plt', plt_content_2),
        ('sample_complex.plt', plt_content_3),
    ]

    created_count = 0
    for filename, content in sample_files:
        file_path = input_dir / filename
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  Created: {filename}")
            created_count += 1
        except Exception as e:
            print(f"  Failed to create {filename}: {e}")

    print(f"Created {created_count} sample PLT files in {input_dir}")
    return created_count > 0


def run_complete_demo(config, processing_system):
    print("\nRUNNING COMPLETE DEMONSTRATION")
    print("=" * 50)

    print("\n1. Creating sample PLT files...")
    input_dir = config.get('directories')['input']
    if not create_sample_plt_files(input_dir):
        print("Failed to create sample files")
        return

    print("\n2. Demonstrating Hopfield networks...")
    hopfield_demo(processing_system)

    print("\n3. Processing test file...")
    plt_files = processing_system.get_plt_files(input_dir)
    if plt_files:
        success = processing_system.process_single_file(plt_files[0], 0)
        if success:
            print("Test processing successful!")
        else:
            print("Test processing failed")
        processing_system.directory_manager.organize_output_files()

    print("\n4. Output directory structure:")
    output_dir = config.get('directories')['output']
    show_directory_structure(output_dir)

    output_base = Path(output_dir)
    for subdir in ['GROUP_PDF', 'GROUP_DXF', 'GROUP_PNG', 'GROUP_SVG']:
        group_dir = output_base / subdir
        if group_dir.exists():
            files = list(group_dir.glob('*'))
            print(f"  {subdir}: {len(files)} files")
        else:
            print(f"  {subdir}: Not created yet")

    print("\nDEMO COMPLETE!")
    print(f"Output: {output_dir}")


def _parse_color(color_str):
    """Parse color string to (r, g, b) tuple with values 0.0-1.0."""
    named = {
        'black': (0.0, 0.0, 0.0), 'white': (1.0, 1.0, 1.0),
        'red': (1.0, 0.0, 0.0), 'green': (0.0, 0.5, 0.0),
        'blue': (0.0, 0.0, 1.0), 'gray': (0.5, 0.5, 0.5),
        'grey': (0.5, 0.5, 0.5), 'navy': (0.0, 0.0, 0.5),
        'brown': (0.4, 0.2, 0.0), 'sepia': (0.44, 0.26, 0.08),
    }
    s = color_str.strip().lower()
    if s in named:
        return named[s]
    s = s.lstrip('#')
    if len(s) == 6:
        return (int(s[0:2], 16) / 255, int(s[2:4], 16) / 255, int(s[4:6], 16) / 255)
    raise ValueError(f"Unknown color: {color_str!r}  (use a name or #rrggbb hex)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GPU-Integrated AI-Enhanced Scribble Plotter')
    parser.add_argument('--input', '-i', type=str,
                        help='Input directory containing PLT, DXF, DWG, or DWX files')
    parser.add_argument('--output', '-o', type=str,
                        help='Base output directory')
    parser.add_argument('--examples', '-e', type=int, default=3,
                        help='Number of examples per file')
    parser.add_argument('--steps', type=int, default=None,
                        help='Override steps per segment (more steps = more encoding capacity)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Path to FASTA file for sequence-driven wobble')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible output')
    parser.add_argument('--color', type=str, default=None,
                        help='Stroke color: "black", "red", or hex "#2a4b7c"')
    parser.add_argument('--stroke-weight', type=float, default=None,
                        help='Stroke weight (default: 3.0)')
    parser.add_argument('--scribble', type=float, default=None,
                        help='Scribble wobble amplitude (default: 3.0)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-ai', action='store_true',
                        help='Disable AI parameter prediction')
    parser.add_argument('--no-hopfield', action='store_true',
                        help='Disable Hopfield memory')
    parser.add_argument('--no-svg', action='store_true',
                        help='Disable SVG output')
    parser.add_argument('--test', action='store_true',
                        help='Run quick system test')
    parser.add_argument('--demo', action='store_true',
                        help='Run Hopfield demo')
    parser.add_argument('--full-demo', action='store_true',
                        help='Run complete end-to-end demo')
    parser.add_argument('--create-samples', action='store_true',
                        help='Create sample PLT files')
    parser.add_argument('--no-groups', action='store_true',
                        help='Disable GROUP directory organization')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable cubic spline stroke smoothing')
    parser.add_argument('--gpu-status', action='store_true',
                        help='Show GPU status')

    args = parser.parse_args()

    base_dir = args.output if args.output else None
    SYSTEM_INFO = setup_complete_system(base_dir)

    CONFIG = CompleteConfiguration(SYSTEM_INFO)

    if args.no_gpu:
        CONFIG.set('gpu_enabled', False)
        CONFIG.set('use_gpu_acceleration', False)
    if args.no_ai:
        CONFIG.set('use_ai_parameters', False)
    if args.no_hopfield:
        CONFIG.set('use_hopfield_memory', False)
    if args.examples:
        CONFIG.set('total_examples', args.examples)
    if args.no_svg:
        CONFIG.config['generate_svg'] = False
    if args.steps:
        CONFIG.config['steps_override'] = args.steps
    if args.seed is not None:
        random.seed(args.seed)
        CONFIG.config['seed'] = args.seed
    if args.color:
        CONFIG.config['color_override'] = _parse_color(args.color)
    if args.stroke_weight is not None:
        CONFIG.config['stroke_weight_override'] = args.stroke_weight
    if args.scribble is not None:
        CONFIG.config['scribble_override'] = args.scribble
    if args.no_smooth:
        CONFIG.config['smooth_strokes'] = False

    CONFIG.display_status()

    PROCESSING_SYSTEM = CompleteProcessingSystem(CONFIG)

    if args.gpu_status:
        gpu_status()
        return

    if args.test:
        quick_test(CONFIG, PROCESSING_SYSTEM)
        return

    if args.no_groups:
        CONFIG.set('organize_groups', False)

    if args.demo:
        hopfield_demo(PROCESSING_SYSTEM)
        return

    if args.full_demo:
        run_complete_demo(CONFIG, PROCESSING_SYSTEM)
        return

    if args.create_samples:
        input_dir = CONFIG.get('directories')['input']
        create_sample_plt_files(input_dir)
        return

    if args.sequence:
        PROCESSING_SYSTEM.set_sequence(args.sequence)

    if args.input:
        input_dir = args.input
    else:
        input_dir = CONFIG.get('directories')['input']

    plt_files = PROCESSING_SYSTEM.get_input_files(input_dir)

    if not plt_files:
        print(f"\nNo input files found in: {input_dir}")
        print("Place .plt, .dxf, .dwg, or .dwx files in the input directory and run again.")
        print("\nAvailable commands:")
        print("  --test            Run system test")
        print("  --demo            Run Hopfield demo")
        print("  --full-demo       Run complete end-to-end demo")
        print("  --create-samples  Create sample PLT files")
        print("  --gpu-status      Show GPU status")
        print("  --sequence FILE   Use FASTA sequence for deterministic wobble")
        print("  --steps N         Override steps per segment")
        print("  --help            Show all options")
        return

    PROCESSING_SYSTEM.process_batch(input_dir)


if __name__ == '__main__':
    main()
