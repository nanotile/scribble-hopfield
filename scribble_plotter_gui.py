# -*- coding: utf-8 -*-
"""
GPU-Integrated AI-Enhanced Scribble Plotter - PyQt6 GUI
A local graphical interface for the Scribble Plotter system
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QScrollArea, QFileDialog, QMessageBox,
    QSplitter, QGridLayout, QSizePolicy, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QAction, QFont

# Import from the local scribble plotter module
from scribble_plotter_local import (
    setup_complete_system,
    CompleteConfiguration,
    CompleteProcessingSystem,
)
import torch


class ProcessingWorker(QThread):
    """Worker thread for batch processing to prevent UI freeze"""

    progress = pyqtSignal(int, int)  # current, total
    log = pyqtSignal(str)
    image_ready = pyqtSignal(str)  # path to PNG
    finished = pyqtSignal(dict)  # summary

    def __init__(self, processing_system, input_dir, parent=None):
        super().__init__(parent)
        self.processing_system = processing_system
        self.input_dir = input_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            config = self.processing_system.config
            plt_files = self.processing_system.get_plt_files(self.input_dir)

            if not plt_files:
                self.log.emit("No PLT files found in input directory")
                self.finished.emit({'success': False, 'message': 'No PLT files found'})
                return

            total_examples = config.get('total_examples', 3)
            total_operations = len(plt_files) * total_examples

            self.log.emit(f"Processing {len(plt_files)} files x {total_examples} examples = {total_operations} operations")

            if config.get('gpu_enabled'):
                self.log.emit("Using GPU acceleration!")

            successful_operations = 0
            current_operation = 0

            for plt_file in plt_files:
                if self._is_cancelled:
                    self.log.emit("Processing cancelled by user")
                    break

                for iteration in range(total_examples):
                    if self._is_cancelled:
                        break

                    current_operation += 1
                    self.progress.emit(current_operation, total_operations)
                    self.log.emit(f"Processing: {plt_file.name} (iteration {iteration + 1})")

                    success = self.processing_system.process_single_file(plt_file, iteration)

                    if success:
                        successful_operations += 1
                        # Find and emit the generated PNG path
                        base_name = plt_file.stem
                        output_base = Path(config.get('directories')['output'])
                        work_dir = output_base / f"{base_name}_{iteration}" / 'PNG'

                        if work_dir.exists():
                            png_files = list(work_dir.glob("*.png"))
                            if png_files:
                                self.image_ready.emit(str(png_files[0]))

                    self.log.emit(f"  {'Success' if success else 'Failed'}")

            # Organize output files into GROUP directories
            self.processing_system.directory_manager.organize_output_files()
            if config.get('organize_groups', True):
                self.log.emit("Output files organized into GROUP directories")

            summary = {
                'success': True,
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': f"{successful_operations}/{total_operations}",
                'processed_files': len(self.processing_system.processed_files),
                'error_files': len(self.processing_system.error_files),
                'hopfield_patterns': len(self.processing_system.hopfield_network.stored_patterns),
                'gpu_accelerated': config.get('gpu_enabled'),
                'output_directory': config.get('directories')['output']
            }

            self.finished.emit(summary)

        except Exception as e:
            self.log.emit(f"Processing error: {str(e)}")
            self.finished.emit({'success': False, 'message': str(e)})


class SingleFileWorker(QThread):
    """Worker thread for processing a single file"""

    log = pyqtSignal(str)
    image_ready = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, processing_system, plt_file, parent=None):
        super().__init__(parent)
        self.processing_system = processing_system
        self.plt_file = plt_file

    def run(self):
        try:
            config = self.processing_system.config
            self.log.emit(f"Processing single file: {self.plt_file.name}")

            success = self.processing_system.process_single_file(self.plt_file, 0)

            if success:
                base_name = self.plt_file.stem
                output_base = Path(config.get('directories')['output'])
                work_dir = output_base / f"{base_name}_0" / 'PNG'

                if work_dir.exists():
                    png_files = list(work_dir.glob("*.png"))
                    if png_files:
                        self.image_ready.emit(str(png_files[0]))

                self.log.emit("Single file processing complete")
            else:
                self.log.emit("Single file processing failed")

            self.finished.emit(success)

        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            self.finished.emit(False)


class HopfieldDemoWorker(QThread):
    """Worker thread for Hopfield demo"""

    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, hopfield_network, parent=None):
        super().__init__(parent)
        self.hopfield_network = hopfield_network

    def run(self):
        try:
            self.log.emit("HOPFIELD NETWORK DEMO")
            self.log.emit("Kent Benson's 1986 vision of neural network art control")
            self.log.emit("=" * 50)

            artistic_pattern = [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2]
            self.hopfield_network.store_pattern(artistic_pattern, "demo_style")
            self.log.emit("Stored artistic pattern: demo_style")

            partial = [0.8, 0.2, 0.0, 0.0, 0.7, 0.0, 0.6, 0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]
            recalled = self.hopfield_network.recall_pattern(partial)

            self.log.emit(f"Partial input: {[f'{x:.1f}' for x in partial[:8]]}")
            self.log.emit(f"Network recall: {[f'{x:.1f}' for x in recalled[:8]]}")
            self.log.emit("")
            self.log.emit("This shows how the network completes partial artistic patterns!")
            self.log.emit("The Hopfield network 'remembers' stored patterns and can")
            self.log.emit("reconstruct them from partial or noisy inputs.")

            self.finished.emit()

        except Exception as e:
            self.log.emit(f"Demo error: {str(e)}")
            self.finished.emit()


class ScribblePlotterGUI(QMainWindow):
    """Main GUI window for the Scribble Plotter application"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GPU-Integrated AI-Enhanced Scribble Plotter")
        self.setMinimumSize(900, 700)

        # Initialize the processing system
        self.init_processing_system()

        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()

        # Worker threads
        self.processing_worker = None
        self.single_file_worker = None
        self.demo_worker = None

    def init_processing_system(self):
        """Initialize the backend processing system"""
        base_dir = Path.cwd() / "ScribblePlotter_Output"
        self.system_info = setup_complete_system(str(base_dir))
        self.config = CompleteConfiguration(self.system_info)
        self.processing_system = CompleteProcessingSystem(self.config)

        self.input_dir = self.config.get('directories')['input']
        self.output_dir = self.config.get('directories')['output']

    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Top section: Settings
        settings_layout = QHBoxLayout()

        # Left column: Input and AI settings
        left_column = QVBoxLayout()
        left_column.addWidget(self.create_input_settings_group())
        left_column.addWidget(self.create_ai_settings_group())
        left_column.addStretch()

        # Right column: Output settings and controls
        right_column = QVBoxLayout()
        right_column.addWidget(self.create_output_settings_group())
        right_column.addWidget(self.create_controls_group())
        right_column.addStretch()

        settings_layout.addLayout(left_column)
        settings_layout.addLayout(right_column)

        main_layout.addLayout(settings_layout)

        # Middle section: Preview and Log (splitter for resizing)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.create_preview_panel())
        splitter.addWidget(self.create_log_panel())
        splitter.setSizes([400, 400])

        main_layout.addWidget(splitter, stretch=1)

        # Bottom section: Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m - %p%")
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

    def create_input_settings_group(self):
        """Create the input settings group"""
        group = QGroupBox("Input Settings")
        layout = QGridLayout()

        # Input directory
        layout.addWidget(QLabel("Input Directory:"), 0, 0)
        self.input_dir_edit = QLineEdit(self.input_dir)
        self.input_dir_edit.setReadOnly(True)
        layout.addWidget(self.input_dir_edit, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.select_input_dir)
        layout.addWidget(browse_btn, 0, 2)

        # Number of examples
        layout.addWidget(QLabel("Examples per file:"), 1, 0)
        self.examples_spinbox = QSpinBox()
        self.examples_spinbox.setRange(1, 10)
        self.examples_spinbox.setValue(self.config.get('total_examples', 3))
        self.examples_spinbox.valueChanged.connect(self.on_examples_changed)
        layout.addWidget(self.examples_spinbox, 1, 1)

        group.setLayout(layout)
        return group

    def create_ai_settings_group(self):
        """Create the AI settings group"""
        group = QGroupBox("AI Settings")
        layout = QVBoxLayout()

        # AI Parameters checkbox
        self.ai_params_checkbox = QCheckBox("Use AI Parameters")
        self.ai_params_checkbox.setChecked(self.config.get('use_ai_parameters', True))
        self.ai_params_checkbox.stateChanged.connect(self.on_ai_params_changed)
        layout.addWidget(self.ai_params_checkbox)

        # Hopfield Memory checkbox
        self.hopfield_checkbox = QCheckBox("Use Hopfield Memory")
        self.hopfield_checkbox.setChecked(self.config.get('use_hopfield_memory', True))
        self.hopfield_checkbox.stateChanged.connect(self.on_hopfield_changed)
        layout.addWidget(self.hopfield_checkbox)

        # GPU Acceleration checkbox
        self.gpu_checkbox = QCheckBox("GPU Acceleration")
        gpu_available = self.config.get('gpu_enabled', False)
        self.gpu_checkbox.setChecked(gpu_available)
        self.gpu_checkbox.setEnabled(torch.cuda.is_available())
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_changed)

        if not torch.cuda.is_available():
            self.gpu_checkbox.setToolTip("GPU not available on this system")
        layout.addWidget(self.gpu_checkbox)

        group.setLayout(layout)
        return group

    def create_output_settings_group(self):
        """Create the output settings group"""
        group = QGroupBox("Output Settings")
        layout = QVBoxLayout()

        # PNG checkbox
        self.png_checkbox = QCheckBox("Generate PNG")
        self.png_checkbox.setChecked(self.config.get('generate_png', True))
        self.png_checkbox.stateChanged.connect(lambda state: self.config.set('generate_png', state == Qt.CheckState.Checked.value))
        layout.addWidget(self.png_checkbox)

        # PDF checkbox
        self.pdf_checkbox = QCheckBox("Generate PDF")
        self.pdf_checkbox.setChecked(self.config.get('generate_pdf', True))
        self.pdf_checkbox.stateChanged.connect(lambda state: self.config.set('generate_pdf', state == Qt.CheckState.Checked.value))
        layout.addWidget(self.pdf_checkbox)

        # DXF checkbox
        self.dxf_checkbox = QCheckBox("Generate DXF")
        self.dxf_checkbox.setChecked(self.config.get('generate_dxf', True))
        self.dxf_checkbox.stateChanged.connect(lambda state: self.config.set('generate_dxf', state == Qt.CheckState.Checked.value))
        layout.addWidget(self.dxf_checkbox)

        # GROUP organization checkbox
        self.groups_checkbox = QCheckBox("Organize GROUP directories")
        self.groups_checkbox.setChecked(self.config.get('organize_groups', True))
        self.groups_checkbox.stateChanged.connect(lambda state: self.config.set('organize_groups', state == Qt.CheckState.Checked.value))
        self.groups_checkbox.setToolTip("Copy output files into GROUP_PDF, GROUP_DXF, GROUP_PNG directories")
        layout.addWidget(self.groups_checkbox)

        group.setLayout(layout)
        return group

    def create_controls_group(self):
        """Create the controls group"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()

        # Test Single File button
        self.test_btn = QPushButton("Test Single File")
        self.test_btn.clicked.connect(self.test_single_file)
        layout.addWidget(self.test_btn)

        # Process All Files button
        self.process_btn = QPushButton("Process All Files")
        self.process_btn.clicked.connect(self.process_all_files)
        layout.addWidget(self.process_btn)

        # Cancel button (hidden initially)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.hide()
        layout.addWidget(self.cancel_btn)

        # Hopfield Demo button
        self.demo_btn = QPushButton("Hopfield Demo")
        self.demo_btn.clicked.connect(self.hopfield_demo)
        layout.addWidget(self.demo_btn)

        group.setLayout(layout)
        return group

    def create_preview_panel(self):
        """Create the preview panel"""
        group = QGroupBox("Preview")
        layout = QVBoxLayout()

        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(350, 300)

        # Image label
        self.preview_label = QLabel("No preview available")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("QLabel { background-color: white; border: 1px solid #ccc; }")
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scroll_area.setWidget(self.preview_label)
        layout.addWidget(scroll_area)

        group.setLayout(layout)
        return group

    def create_log_panel(self):
        """Create the status log panel"""
        group = QGroupBox("Status Log")
        layout = QVBoxLayout()

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumSize(350, 300)
        font = QFont("Courier New", 9)
        self.log_text.setFont(font)

        layout.addWidget(self.log_text)

        # Clear log button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    def setup_menu(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Input Directory...", self)
        open_action.triggered.connect(self.select_input_dir)
        file_menu.addAction(open_action)

        open_output_action = QAction("Open Output Directory", self)
        open_output_action.triggered.connect(self.open_output_dir)
        file_menu.addAction(open_output_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_statusbar(self):
        """Setup the status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # GPU status indicator
        gpu_status = "GPU: " + ("Available" if torch.cuda.is_available() else "Not Available")
        if torch.cuda.is_available():
            gpu_status += f" ({torch.cuda.get_device_name(0)})"

        self.gpu_label = QLabel(gpu_status)
        self.statusbar.addPermanentWidget(self.gpu_label)

        self.statusbar.showMessage("Ready")

    def log(self, message):
        """Add a message to the log"""
        self.log_text.appendPlainText(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def select_input_dir(self):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            self.input_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self.input_dir = directory
            self.input_dir_edit.setText(directory)
            self.log(f"Input directory set to: {directory}")

            # Count PLT files
            plt_files = list(Path(directory).glob("*.plt")) + list(Path(directory).glob("*.PLT"))
            self.log(f"Found {len(plt_files)} PLT files")

    def open_output_dir(self):
        """Open the output directory in file manager"""
        import subprocess
        output_path = self.config.get('directories')['output']

        if sys.platform == 'win32':
            os.startfile(output_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', output_path])
        else:
            subprocess.run(['xdg-open', output_path])

    def on_examples_changed(self, value):
        """Handle examples spinbox change"""
        self.config.set('total_examples', value)

    def on_ai_params_changed(self, state):
        """Handle AI parameters checkbox change"""
        self.config.set('use_ai_parameters', state == Qt.CheckState.Checked.value)

    def on_hopfield_changed(self, state):
        """Handle Hopfield memory checkbox change"""
        self.config.set('use_hopfield_memory', state == Qt.CheckState.Checked.value)

    def on_gpu_changed(self, state):
        """Handle GPU acceleration checkbox change"""
        enabled = state == Qt.CheckState.Checked.value
        self.config.set('gpu_enabled', enabled)
        self.config.set('use_gpu_acceleration', enabled)

    def test_single_file(self):
        """Process a single PLT file"""
        plt_files = list(Path(self.input_dir).glob("*.plt")) + list(Path(self.input_dir).glob("*.PLT"))

        if not plt_files:
            QMessageBox.warning(self, "No Files", "No PLT files found in the input directory.")
            return

        self.set_processing_state(True)
        self.log("Starting single file test...")

        self.single_file_worker = SingleFileWorker(self.processing_system, plt_files[0])
        self.single_file_worker.log.connect(self.log)
        self.single_file_worker.image_ready.connect(self.update_preview)
        self.single_file_worker.finished.connect(self.on_single_file_finished)
        self.single_file_worker.start()

    def on_single_file_finished(self, success):
        """Handle single file processing completion"""
        self.set_processing_state(False)
        if success:
            self.statusbar.showMessage("Single file processing complete")
        else:
            self.statusbar.showMessage("Single file processing failed")

    def process_all_files(self):
        """Process all PLT files in the input directory"""
        plt_files = list(Path(self.input_dir).glob("*.plt")) + list(Path(self.input_dir).glob("*.PLT"))

        if not plt_files:
            QMessageBox.warning(self, "No Files", "No PLT files found in the input directory.")
            return

        total_examples = self.config.get('total_examples', 3)
        total_operations = len(plt_files) * total_examples

        self.set_processing_state(True)
        self.progress_bar.setRange(0, total_operations)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.cancel_btn.show()

        self.log(f"Starting batch processing: {len(plt_files)} files x {total_examples} examples")

        self.processing_worker = ProcessingWorker(self.processing_system, self.input_dir)
        self.processing_worker.progress.connect(self.update_progress)
        self.processing_worker.log.connect(self.log)
        self.processing_worker.image_ready.connect(self.update_preview)
        self.processing_worker.finished.connect(self.on_processing_finished)
        self.processing_worker.start()

    def cancel_processing(self):
        """Cancel the current processing operation"""
        if self.processing_worker and self.processing_worker.isRunning():
            self.processing_worker.cancel()
            self.log("Cancelling processing...")

    def update_progress(self, current, total):
        """Update the progress bar"""
        self.progress_bar.setValue(current)
        self.statusbar.showMessage(f"Processing... {current}/{total}")

    def on_processing_finished(self, summary):
        """Handle batch processing completion"""
        self.set_processing_state(False)
        self.progress_bar.hide()
        self.cancel_btn.hide()

        self.log("")
        self.log("=" * 50)
        self.log("PROCESSING COMPLETE")
        self.log("=" * 50)
        self.log(f"Total operations: {summary.get('total_operations', 0)}")
        self.log(f"Successful: {summary.get('successful_operations', 0)}")
        self.log(f"Success rate: {summary.get('success_rate', 'N/A')}")
        self.log(f"Hopfield patterns learned: {summary.get('hopfield_patterns', 0)}")
        self.log(f"GPU accelerated: {'YES' if summary.get('gpu_accelerated') else 'NO'}")
        self.log(f"Output directory: {summary.get('output_directory', '')}")
        self.log("=" * 50)

        self.statusbar.showMessage("Processing complete")

        if summary.get('success'):
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Successfully processed {summary.get('successful_operations', 0)} of {summary.get('total_operations', 0)} operations.\n\n"
                f"Output saved to:\n{summary.get('output_directory', '')}"
            )

    def hopfield_demo(self):
        """Run the Hopfield network demonstration"""
        self.set_processing_state(True)

        self.demo_worker = HopfieldDemoWorker(self.processing_system.hopfield_network)
        self.demo_worker.log.connect(self.log)
        self.demo_worker.finished.connect(self.on_demo_finished)
        self.demo_worker.start()

    def on_demo_finished(self):
        """Handle demo completion"""
        self.set_processing_state(False)
        self.statusbar.showMessage("Hopfield demo complete")

    def update_preview(self, image_path):
        """Update the preview panel with an image"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale to fit while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    QSize(500, 400),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
                self.log(f"Preview updated: {os.path.basename(image_path)}")

    def set_processing_state(self, processing):
        """Enable/disable UI elements during processing"""
        self.test_btn.setEnabled(not processing)
        self.process_btn.setEnabled(not processing)
        self.demo_btn.setEnabled(not processing)
        self.examples_spinbox.setEnabled(not processing)
        self.ai_params_checkbox.setEnabled(not processing)
        self.hopfield_checkbox.setEnabled(not processing)
        self.gpu_checkbox.setEnabled(not processing and torch.cuda.is_available())
        self.png_checkbox.setEnabled(not processing)
        self.pdf_checkbox.setEnabled(not processing)
        self.dxf_checkbox.setEnabled(not processing)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Scribble Plotter",
            "GPU-Integrated AI-Enhanced Scribble Plotter v3.0\n\n"
            "A system that converts PLT vector graphics into artistic\n"
            "'scribble' renderings using GPU-accelerated AI and\n"
            "Hopfield neural networks.\n\n"
            "Honoring Kent Benson's 1983-1986 Hopfield Network Research"
        )

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any running workers
        if self.processing_worker and self.processing_worker.isRunning():
            self.processing_worker.cancel()
            self.processing_worker.wait()

        if self.single_file_worker and self.single_file_worker.isRunning():
            self.single_file_worker.wait()

        if self.demo_worker and self.demo_worker.isRunning():
            self.demo_worker.wait()

        event.accept()


def main():
    # Suppress matplotlib backend warning in GUI mode
    import matplotlib
    matplotlib.use('Agg')

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = ScribblePlotterGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
