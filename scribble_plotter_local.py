# -*- coding: utf-8 -*-
"""
GPU-Integrated AI-Enhanced Scribble Plotter - Local Version
Adapted from Google Colab notebook for local execution
"""

import os
import sys
import json
import re
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from datetime import datetime
import shutil
import warnings
import argparse

warnings.filterwarnings('ignore')

# GPU imports
import torch

print("GPU-Integrated AI-Enhanced Scribble Plotter (Local Version)")
print("Honoring Kent Benson's 1983-1986 Hopfield Network Research")
print("=" * 70)


def setup_complete_system(base_dir: str = None):
    """Setup GPU acceleration and project structure"""

    print("Setting up complete system with GPU acceleration...")

    # Use provided base_dir or default to current directory
    if base_dir is None:
        base_dir = Path.cwd() / "ScribblePlotter_Output"
    else:
        base_dir = Path(base_dir)

    # GPU Detection and Setup
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {'YES' if gpu_available else 'NO'}")

    if gpu_available:
        device = torch.device('cuda')
        print(f"GPU Type: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

        # Test GPU performance
        print("GPU Performance Test...")
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)

        import time
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"GPU Matrix Test: {gpu_time:.3f} seconds")
    else:
        device = torch.device('cpu')
        print("Running on CPU (no GPU detected)")

    # Create project structure
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
        print(f"Created: {name} -> {path}")

    return {
        'gpu_available': gpu_available,
        'device': device,
        'directories': directories
    }


class CompleteConfiguration:
    """Complete configuration system with GPU settings"""

    def __init__(self, system_info):
        self.system_info = system_info
        self.config_path = system_info['directories']['config']
        self.config_file = f"{self.config_path}/complete_config.json"

        self.defaults = {
            'project_name': 'GPU-Enhanced Scribble Plotter',
            'version': '3.0-local',
            'author': 'Kent Benson',
            'inspiration': '1983-1986 Hopfield Network Research',
            'gpu_enabled': system_info['gpu_available'],
            'device': str(system_info['device']),
            'directories': system_info['directories'],
            'total_examples': 3,
            'batch_size': 8,
            'use_gpu_acceleration': system_info['gpu_available'],
            'generate_pdf': True,
            'generate_dxf': True,
            'generate_png': True,
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
            config_to_save = config or self.config
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


class PLTProcessor:
    """Core PLT file processing with GPU optimization"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.setup_patterns()

    def setup_patterns(self):
        self.patterns = {
            'PD': re.compile(r'PD(\d+),(\d+)'),
            'PD_neg_x': re.compile(r'PD-(\d+),(\d+)'),
            'PD_neg_y': re.compile(r'PD(\d+),-(\d+)'),
            'PD_neg_both': re.compile(r'PD-(\d+),-(\d+)'),
            'PA': re.compile(r'PA(\d+)\.(\d+),(\d+)\.(\d+)'),
            'PU': re.compile(r'PU(\d+)\.(\d+),(\d+)\.(\d+)'),
        }

    def process_plt_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            if 'PW0' in content[:200]:
                return self.process_acme_convert(content)
            else:
                return self.process_acme_trace(content)

        except Exception as e:
            print(f"PLT processing error: {e}")
            return []

    def process_acme_convert(self, content):
        content = content.replace('\n', '').replace('\r', '')
        lines = content.split(';')
        return self.extract_coordinates(lines)

    def process_acme_trace(self, content):
        lines = content.split(';')
        return self.extract_coordinates(lines)

    def extract_coordinates(self, lines):
        coordinates = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern_name, pattern in self.patterns.items():
                match = pattern.match(line)
                if match:
                    coord = self.process_match(pattern_name, match)
                    if coord:
                        coordinates.append(coord)
                    break

        coordinates.reverse()
        return coordinates

    def process_match(self, pattern_name, match):
        groups = match.groups()

        if pattern_name == 'PD':
            return (float(groups[0]), float(groups[1]), 0.0)
        elif pattern_name == 'PD_neg_x':
            return (-float(groups[0]), float(groups[1]), 0.0)
        elif pattern_name == 'PD_neg_y':
            return (float(groups[0]), -float(groups[1]), 0.0)
        elif pattern_name == 'PD_neg_both':
            return (-float(groups[0]), -float(groups[1]), 0.0)
        elif pattern_name == 'PA':
            return (float(groups[0]), float(groups[2]), 0.0)
        elif pattern_name == 'PU':
            return (float(groups[0]), float(groups[2]), 10.0)

        return None

    def scale_points(self, coordinates):
        if not coordinates:
            return []

        scale_x = self.config.get('scale_factor_x', 0.09)
        scale_y = self.config.get('scale_factor_y', 0.09)
        scale_z = self.config.get('scale_factor_z', 1.0)

        scaled = []
        for x, y, z in coordinates:
            scaled.append((x * scale_x, y * scale_y, z * scale_z))

        return scaled


class GPUAcceleratedAI:
    """Complete GPU-accelerated AI system"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.gpu_available = config.get('gpu_enabled', False)

        if self.gpu_available:
            print("Initializing GPU-accelerated AI...")
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(config.get('gpu_memory_fraction', 0.8))
        else:
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

        if len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            features[1] = np.sum(distances)

            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            features[2] = max_coords[0] - min_coords[0]
            features[3] = max_coords[1] - min_coords[1]
            features[4] = features[2] / max(features[3], 1e-6)

        return np.array(features)

    def predict_parameters(self, points):
        try:
            features = self.extract_features_gpu(points)

            complexity = features[0]
            path_length = features[1]
            aspect_ratio = features[4] if len(features) > 4 else 1.0

            if complexity > 0.7:
                steps = random.randint(8, 15)
                scribble = random.uniform(1.0, 3.0)
            elif complexity > 0.4:
                steps = random.randint(4, 10)
                scribble = random.uniform(2.0, 5.0)
            else:
                steps = random.randint(2, 7)
                scribble = random.uniform(3.0, 7.0)

            if path_length > 2000:
                scribble *= 1.2
                steps = min(steps + 2, 15)
            elif path_length < 500:
                scribble *= 0.8
                steps = max(steps - 1, 2)

            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                scribble *= 1.1

            stroke_weight = random.uniform(1.0, 4.0)
            color = (random.random(), random.random(), random.random())

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
                'steps': random.randint(3, 8),
                'scribble': random.uniform(2.0, 6.0),
                'stroke_weight': random.uniform(1.0, 4.0),
                'color': (random.random(), random.random(), random.random()),
                'complexity_score': 0.5,
                'generation_method': 'fallback'
            }


class GPUHopfieldNetwork:
    """GPU-accelerated Hopfield network for style memory"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.gpu_available = config.get('gpu_enabled', False)

        self.pattern_size = config.get('hopfield_pattern_size', 15)
        self.memory_size = config.get('hopfield_memory_size', 200)

        if self.gpu_available:
            self.weights = torch.zeros(self.pattern_size, self.pattern_size, device=self.device)
            print("GPU Hopfield network initialized")
        else:
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

        for iteration in range(max_iterations):
            previous = current.clone()
            activations = torch.mv(self.weights, current)
            current = torch.sign(activations)

            if torch.equal(current, previous):
                break

        return self._denormalize_pattern_gpu(current)

    def _cpu_recall(self, partial_pattern, max_iterations):
        current = self._normalize_pattern_cpu(partial_pattern)

        for iteration in range(max_iterations):
            previous = current.copy()

            for i in range(len(current)):
                activation = np.dot(self.weights[i], current)
                current[i] = 1 if activation > 0 else -1

            if np.array_equal(current, previous):
                break

        return self._denormalize_pattern_cpu(current)

    def find_spurious_memories(self, test_patterns):
        spurious_memories = []

        print("Searching for spurious memories...")
        print("Exploring 'illogical associations' that lead to plausible new states")

        for i, test_pattern in enumerate(test_patterns):
            recalled = self.recall_pattern(test_pattern)

            is_stored = False
            for stored in self.stored_patterns:
                if self._patterns_similar(recalled, stored):
                    is_stored = True
                    break

            if not is_stored:
                spurious_memories.append(recalled)
                print(f"Found spurious memory {len(spurious_memories)}")

        return spurious_memories

    def _patterns_similar(self, pattern1, pattern2, threshold=0.9):
        if len(pattern1) != len(pattern2):
            return False

        if self.gpu_available and hasattr(pattern1, 'device'):
            pattern1 = pattern1.cpu().numpy()
        if self.gpu_available and hasattr(pattern2, 'device'):
            pattern2 = pattern2.cpu().numpy()

        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        return abs(correlation) > threshold

    def _normalize_pattern_gpu(self, pattern):
        pattern_tensor = torch.tensor(pattern[:self.pattern_size],
                                      dtype=torch.float32, device=self.device)

        if len(pattern_tensor) < self.pattern_size:
            padding = torch.zeros(self.pattern_size - len(pattern_tensor), device=self.device)
            pattern_tensor = torch.cat([pattern_tensor, padding])

        if torch.std(pattern_tensor) > 0:
            pattern_tensor = (pattern_tensor - torch.mean(pattern_tensor)) / torch.std(pattern_tensor)

        return torch.sign(pattern_tensor)

    def _denormalize_pattern_gpu(self, pattern):
        return ((pattern + 1) / 2).cpu().numpy()

    def _normalize_pattern_cpu(self, pattern):
        pattern = np.array(pattern[:self.pattern_size])
        if len(pattern) < self.pattern_size:
            pattern = np.pad(pattern, (0, self.pattern_size - len(pattern)))

        if np.std(pattern) > 0:
            pattern = (pattern - np.mean(pattern)) / np.std(pattern)

        return np.sign(pattern)

    def _denormalize_pattern_cpu(self, pattern):
        return (pattern + 1) / 2


class ScribbleRenderer:
    """Enhanced scribble renderer"""

    def __init__(self, config):
        self.config = config

    def render_artwork(self, points, params):
        try:
            if not points:
                return None

            fig_width = self.config.get('page_width', 800) / 100
            fig_height = self.config.get('page_height', 600) / 100

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.set_aspect('equal')
            ax.set_facecolor('white')
            ax.axis('off')

            steps = params['steps']
            scrib_val = params['scribble']
            stroke_weight = params['stroke_weight']
            color = params['color']

            for i in range(len(points) - 1):
                x1, y1, z1 = points[i]

                if z1 == 0:
                    x2, y2, z2 = points[i + 1] if i + 1 < len(points) else points[i]

                    x_step = (x2 - x1) / steps
                    y_step = (y2 - y1) / steps

                    current_x, current_y = x1, y1

                    for step in range(steps):
                        if step < steps - 1:
                            next_x = current_x + x_step + random.uniform(-scrib_val, scrib_val)
                            next_y = current_y + y_step + random.uniform(-scrib_val, scrib_val)
                        else:
                            next_x, next_y = x2, y2

                        ax.plot([current_x, next_x], [current_y, next_y],
                                color=color, linewidth=stroke_weight / 5, alpha=0.8)

                        current_x, current_y = next_x, next_y

            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)
            ax.invert_yaxis()
            plt.tight_layout()

            return fig

        except Exception as e:
            print(f"Rendering error: {e}")
            # Close any partially-created figure to prevent memory leak
            plt.close('all')
            return None


class DirectoryManager:
    """Manages per-file working directories and GROUP aggregate directories"""

    GROUP_DIRS = ['GROUP_PDF', 'GROUP_DXF', 'GROUP_PNG', 'GROUP_TIFF', 'GROUP_TRANSFORM']

    def __init__(self, config):
        self.config = config
        self.processed_files = []
        self.file_registry = {}
        print("Directory manager initialized")

    def create_working_directories(self, base_dir, file_base_name, file_index):
        """Create per-file working directories and GROUP aggregate directories"""
        try:
            working_dir = Path(base_dir) / f"{file_base_name}_{file_index}"

            directories = {
                'pdf': working_dir / 'PDF',
                'dxf': working_dir / 'DXF',
                'png': working_dir / 'PNG',
                'tmp': working_dir / 'TMP',
            }

            # GROUP directories at the base output level
            for gdir in self.GROUP_DIRS:
                directories[gdir.lower()] = Path(base_dir) / gdir

            for name, path in directories.items():
                path.mkdir(parents=True, exist_ok=True)

            return {name: str(path) for name, path in directories.items()}

        except Exception as e:
            print(f"Directory creation error: {e}")
            return {}

    def register_file(self, file_path, file_type):
        """Register a processed output file for later GROUP organization"""
        self.processed_files.append(file_path)
        self.file_registry[file_path] = {
            'type': file_type,
            'timestamp': datetime.now()
        }

    def organize_output_files(self):
        """Copy all registered output files to GROUP directories (non-destructive)"""
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
                    '.tiff': 'GROUP_TIFF',
                    '.tif': 'GROUP_TIFF',
                }

                group_name = group_map.get(ext)
                if not group_name:
                    continue

                group_dir = base_output / group_name
                group_dir.mkdir(exist_ok=True)

                # Prefix with parent dir name to prevent filename collisions
                parent_name = source.parent.parent.name  # e.g. "rectangle_0"
                dest_name = f"{parent_name}_{source.name}"
                destination = group_dir / dest_name

                shutil.copy2(str(source), str(destination))
                copied_count += 1

            print(f"Organized {copied_count} files into GROUP directories")

        except Exception as e:
            print(f"Error organizing files: {e}")


class CompleteProcessingSystem:
    """Complete processing system"""

    def __init__(self, config):
        self.config = config
        self.plt_processor = PLTProcessor(config)
        self.gpu_ai = GPUAcceleratedAI(config)
        self.hopfield_network = GPUHopfieldNetwork(config)
        self.renderer = ScribbleRenderer(config)
        self.directory_manager = DirectoryManager(config)

        self.processed_files = []
        self.error_files = []

        print("Complete processing system initialized")

    def get_plt_files(self, directory):
        try:
            path = Path(directory)
            if not path.exists():
                print(f"Directory not found: {directory}")
                return []

            plt_files = list(path.glob("*.plt")) + list(path.glob("*.PLT"))
            print(f"Found {len(plt_files)} PLT files")
            return plt_files
        except Exception as e:
            print(f"Error getting PLT files: {e}")
            return []

    def process_single_file(self, plt_file, iteration=0):
        try:
            print(f"\nProcessing: {plt_file.name} (iteration {iteration + 1})")

            raw_coords = self.plt_processor.process_plt_file(str(plt_file))
            if not raw_coords:
                print(f"  Failed to process PLT file")
                return False

            points = self.plt_processor.scale_points(raw_coords)
            print(f"  Loaded {len(points)} points")

            if self.config.get('use_ai_parameters'):
                params = self.gpu_ai.predict_parameters(points)
                print(f"  AI Parameters: steps={params['steps']}, scribble={params['scribble']:.1f}")

                if self.config.get('use_hopfield_memory'):
                    features = self.gpu_ai.extract_features_gpu(points)
                    evolved_features = self.hopfield_network.recall_pattern(features)

                    blend_factor = 0.7
                    params['scribble'] = (blend_factor * params['scribble'] +
                                          (1 - blend_factor) * abs(evolved_features[0]) * 8)
                    params['scribble'] = max(0.1, min(10.0, params['scribble']))

                    print(f"  Hopfield Enhanced: scribble={params['scribble']:.1f}")
            else:
                params = {
                    'steps': random.randint(3, 10),
                    'scribble': random.uniform(2.0, 6.0),
                    'stroke_weight': random.uniform(1.0, 4.0),
                    'color': (random.random(), random.random(), random.random()),
                    'generation_method': 'traditional'
                }

            base_name = plt_file.stem
            output_base = Path(self.config.get('directories')['output'])

            # Use DirectoryManager for directory creation
            dirs = self.directory_manager.create_working_directories(
                str(output_base), base_name, iteration)

            output_dirs = {
                'pdf': Path(dirs.get('pdf', output_base / f"{base_name}_{iteration}" / 'PDF')),
                'dxf': Path(dirs.get('dxf', output_base / f"{base_name}_{iteration}" / 'DXF')),
                'png': Path(dirs.get('png', output_base / f"{base_name}_{iteration}" / 'PNG')),
            }

            figure = self.renderer.render_artwork(points, params)
            if not figure:
                print(f"  Failed to render artwork")
                return False

            method_suffix = params.get('generation_method', 'ai')
            output_name = f"{base_name}_{method_suffix}_steps{params['steps']}_scribble{int(params['scribble'])}_{iteration}"

            success_count = 0

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

            if self.config.get('generate_dxf'):
                dxf_path = output_dirs['dxf'] / f"{output_name}.dxf"
                try:
                    self.save_dxf(points, str(dxf_path), params)
                    self.directory_manager.register_file(str(dxf_path), 'dxf')
                    success_count += 1
                    print(f"  DXF saved: {dxf_path}")
                except Exception as e:
                    print(f"  DXF error: {e}")

            plt.close(figure)

            if success_count > 0 and self.config.get('use_hopfield_memory'):
                features = self.gpu_ai.extract_features_gpu(points)
                style_name = f"{base_name}_{iteration}"
                self.hopfield_network.store_pattern(features, style_name)

            if success_count > 0:
                self.processed_files.append(output_name)
                print(f"  Generated {success_count} format(s) successfully")
                return True
            else:
                print(f"  Failed to save any formats")
                return False

        except Exception as e:
            print(f"  Processing error: {e}")
            self.error_files.append(str(plt_file))
            return False

    def save_dxf(self, points, output_path, params):
        try:
            import ezdxf
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()

            steps = params['steps']
            scrib_val = params['scribble']

            for i in range(len(points) - 1):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[i + 1]

                if z1 == 0:
                    x_step = (x2 - x1) / steps
                    y_step = (y2 - y1) / steps

                    prev_x, prev_y = x1, y1

                    for step in range(steps):
                        if step < steps - 1:
                            curr_x = prev_x + x_step + random.uniform(-scrib_val, scrib_val)
                            curr_y = prev_y + y_step + random.uniform(-scrib_val, scrib_val)
                        else:
                            curr_x, curr_y = x2, y2

                        msp.add_line((prev_x, prev_y), (curr_x, curr_y))
                        prev_x, prev_y = curr_x, curr_y

            doc.saveas(output_path)
            return True
        except Exception as e:
            print(f"DXF save error: {e}")
            return False

    def process_batch(self, input_directory):
        print("BATCH PROCESSING")
        print("=" * 50)

        plt_files = self.get_plt_files(input_directory)
        if not plt_files:
            return {'success': False, 'message': 'No PLT files found'}

        total_examples = self.config.get('total_examples', 3)
        total_operations = len(plt_files) * total_examples

        print(f"Processing {len(plt_files)} files x {total_examples} examples = {total_operations} operations")

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

        # Organize output files into GROUP directories
        self.directory_manager.organize_output_files()

        # Show directory structure after organizing
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


def quick_test(config, processing_system):
    """Quick system test"""
    print("\nQUICK SYSTEM TEST")
    print("=" * 30)

    print(f"1. Configuration: {'OK' if config else 'FAILED'}")
    print(f"2. GPU Available: {'YES' if config.get('gpu_enabled') else 'NO'}")
    print(f"3. Processing System: {'OK' if processing_system else 'FAILED'}")

    input_dir = Path(config.get('directories')['input'])
    print(f"4. Input Directory: {'EXISTS' if input_dir.exists() else 'MISSING'}")

    if input_dir.exists():
        plt_files = processing_system.get_plt_files(str(input_dir))
        print(f"5. PLT Files: {len(plt_files) if plt_files else 'NONE'}")


def gpu_status():
    """Check GPU status"""
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
    """Quick Hopfield demonstration"""
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
    """Show directory tree with box-drawing characters"""
    base = Path(base_path)
    if not base.exists():
        print(f"Directory not found: {base_path}")
        return

    print(f"\nDIRECTORY STRUCTURE")
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
    """Create 3 sample PLT files for demonstration"""
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
    """Run end-to-end demonstration: create samples, process, show output"""
    print("\nRUNNING COMPLETE DEMONSTRATION")
    print("=" * 50)

    # 1. Create sample files
    print("\n1. Creating sample PLT files...")
    input_dir = config.get('directories')['input']
    if not create_sample_plt_files(input_dir):
        print("Failed to create sample files")
        return

    # 2. Hopfield demo
    print("\n2. Demonstrating Hopfield networks...")
    hopfield_demo(processing_system)

    # 3. Process one sample file
    print("\n3. Processing test file...")
    plt_files = processing_system.get_plt_files(input_dir)
    if plt_files:
        success = processing_system.process_single_file(plt_files[0], 0)
        if success:
            print("Test processing successful!")
        else:
            print("Test processing failed")

        # Organize GROUP directories
        processing_system.directory_manager.organize_output_files()

    # 4. Show output structure
    print("\n4. Output directory structure:")
    output_dir = config.get('directories')['output']
    show_directory_structure(output_dir)

    output_base = Path(output_dir)
    for subdir in ['GROUP_PDF', 'GROUP_DXF', 'GROUP_PNG']:
        group_dir = output_base / subdir
        if group_dir.exists():
            files = list(group_dir.glob('*'))
            print(f"  {subdir}: {len(files)} files")
        else:
            print(f"  {subdir}: Not created yet")

    print("\nDEMO COMPLETE!")
    print(f"Output: {output_dir}")
    print("Note: Demo output persists in the output directory. Delete manually if not needed.")


def main():
    parser = argparse.ArgumentParser(description='GPU-Integrated AI-Enhanced Scribble Plotter')
    parser.add_argument('--input', '-i', type=str, help='Input directory containing PLT files')
    parser.add_argument('--output', '-o', type=str, help='Base output directory')
    parser.add_argument('--examples', '-e', type=int, default=3, help='Number of examples per file')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI parameter prediction')
    parser.add_argument('--no-hopfield', action='store_true', help='Disable Hopfield memory')
    parser.add_argument('--test', action='store_true', help='Run quick system test')
    parser.add_argument('--demo', action='store_true', help='Run Hopfield demo')
    parser.add_argument('--full-demo', action='store_true', help='Run complete end-to-end demo')
    parser.add_argument('--create-samples', action='store_true', help='Create sample PLT files')
    parser.add_argument('--no-groups', action='store_true', help='Disable GROUP directory organization')
    parser.add_argument('--gpu-status', action='store_true', help='Show GPU status')

    args = parser.parse_args()

    # Setup system
    base_dir = args.output if args.output else None
    SYSTEM_INFO = setup_complete_system(base_dir)

    # Initialize configuration
    CONFIG = CompleteConfiguration(SYSTEM_INFO)

    # Apply command-line overrides
    if args.no_gpu:
        CONFIG.set('gpu_enabled', False)
        CONFIG.set('use_gpu_acceleration', False)
    if args.no_ai:
        CONFIG.set('use_ai_parameters', False)
    if args.no_hopfield:
        CONFIG.set('use_hopfield_memory', False)
    if args.examples:
        CONFIG.set('total_examples', args.examples)

    CONFIG.display_status()

    # Initialize processing system
    PROCESSING_SYSTEM = CompleteProcessingSystem(CONFIG)

    # Handle commands
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

    # Process files
    if args.input:
        input_dir = args.input
    else:
        input_dir = CONFIG.get('directories')['input']

    plt_files = PROCESSING_SYSTEM.get_plt_files(input_dir)

    if not plt_files:
        print(f"\nNo PLT files found in: {input_dir}")
        print(f"Place your .plt files in the input directory and run again.")
        print(f"\nAvailable commands:")
        print(f"  --test            Run system test")
        print(f"  --demo            Run Hopfield demo")
        print(f"  --full-demo       Run complete end-to-end demo")
        print(f"  --create-samples  Create sample PLT files")
        print(f"  --gpu-status      Show GPU status")
        print(f"  --help            Show all options")
        return

    # Process batch
    PROCESSING_SYSTEM.process_batch(input_dir)


if __name__ == '__main__':
    main()
