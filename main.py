"""
Mass Photo Converter
A high-performance batch conversion tool for PNG to AVIF format.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QSlider, QProgressBar,
    QFileDialog, QMessageBox, QFrame, QGroupBox, QComboBox, QSizePolicy,
    QProgressDialog, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QCursor, QPen, QColor

from io import BytesIO
from PIL import Image

# Try to register optional format plugins
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False

try:
    import pillow_jxl  # noqa: F401 - registers on import
    JXL_AVAILABLE = True
except ImportError:
    JXL_AVAILABLE = False


# -----------------------------------------------------------------------------
# Visual Theme
# -----------------------------------------------------------------------------

DARK_THEME_STYLESHEET = """
/* Global Reset */
QWidget {
    color: #E0E0E0;
    background-color: #1E1E1E;
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}

/* Main Window & Panels */
QMainWindow {
    background-color: #121212;
}

QGroupBox {
    border: 1px solid #3A3A3A;
    border-radius: 8px;
    margin-top: 1.2em;
    font-weight: bold;
    background-color: #252526; /* Distinct section background */
    padding-top: 25px; /* Added breathing room inside groups */
    padding-bottom: 15px;
    padding-left: 15px;
    padding-right: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: transparent;
    color: #64B5F6; /* Accent color for titles */
    font-size: 15pt; /* Bigger section headers */
}

/* Buttons */
QPushButton {
    background-color: #3C3C3C;
    border: 1px solid #505050;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 80px;
}
QPushButton:hover {
    background-color: #4A4A4A;
    border-color: #606060;
}
QPushButton:pressed {
    background-color: #323232;
    border-color: #404040;
}
QPushButton:disabled {
    background-color: #252525;
    color: #606060;
    border-color: #303030;
}

/* Primary Action Buttons */
QPushButton[class="primary"] {
    background-color: #0D47A1;
    border: 1px solid #1565C0;
    color: #FFFFFF;
    font-weight: bold;
}
QPushButton[class="primary"]:hover {
    background-color: #1565C0;
    border-color: #1976D2;
}
QPushButton[class="primary"]:pressed {
    background-color: #0D47A1;
}

/* Input Fields */
QLineEdit, QComboBox {
    background-color: #2D2D2D;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
    selection-background-color: #0D47A1;
    color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #64B5F6;
    background-color: #333333;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox QAbstractItemView {
    background-color: #2D2D2D;
    border: 1px solid #404040;
    color: #E0E0E0;
    outline: none;
}

/* Checkbox */
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #505050;
    background-color: #2D2D2D;
}
QCheckBox::indicator:checked {
    background-color: #1976D2;
    border-color: #1976D2;
    image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz48L3N2Zz4=);
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #404040;
    border-radius: 6px;
    text-align: center;
    background-color: #202020;
    height: 20px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1565C0, stop:1 #2196F3);
    border-radius: 5px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #1E1E1E;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background: #606060;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


@dataclass
class OutputFormat:
    """Configuration for an output format."""
    name: str
    extension: str
    pillow_format: str
    default_quality: int = 80
    extra_params: dict = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


def _build_output_formats() -> dict[str, OutputFormat]:
    """Build available output formats based on installed plugins."""
    formats = {
        'WebP': OutputFormat('WebP', '.webp', 'WebP', default_quality=85),
        'AVIF': OutputFormat('AVIF', '.avif', 'AVIF', default_quality=80, extra_params={'speed': 6}),
        'JPEG': OutputFormat('JPEG', '.jpg', 'JPEG', default_quality=80, extra_params={'optimize': True}),
        'PNG': OutputFormat('PNG', '.png', 'PNG', default_quality=100, extra_params={'optimize': True}),
        'TIFF': OutputFormat('TIFF', '.tiff', 'TIFF', default_quality=80, extra_params={'compression': 'tiff_lzw'}),
        'BMP': OutputFormat('BMP', '.bmp', 'BMP', default_quality=100),
    }
    if HEIC_AVAILABLE:
        formats['HEIC'] = OutputFormat('HEIC', '.heic', 'HEIF', default_quality=80)
    if JXL_AVAILABLE:
        formats['JPEG XL'] = OutputFormat('JPEG XL', '.jxl', 'JXL', default_quality=80)
    return formats


OUTPUT_FORMATS = _build_output_formats()
INPUT_EXTENSIONS = {'.png', '.tiff', '.tif', '.jpg', '.jpeg', '.bmp', '.webp', '.avif', '.heic', '.heif'}


@dataclass
class ConversionResult:
    """Result of a single file conversion."""
    source_path: Path
    dest_path: Path
    original_size: int
    converted_size: int
    success: bool
    error: Optional[str] = None


def prepare_image_for_save(img: Image.Image, format_name: str) -> Image.Image:
    """Prepare image for saving (handle alpha channel for non-supporting formats)."""
    # If the format doesn't support transparency, composite over white
    if format_name in ('JPEG', 'BMP'):
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3])
            return background
        elif img.mode != 'RGB':
            return img.convert('RGB')
        return img
        
    # For other formats, ensure valid mode (usually RGB or RGBA)
    # Most support RGBA, but if it's CMYK or something else, we might want to convert
    if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
        return img.convert('RGBA')
    
    return img


class ConversionWorker(QThread):
    """Background thread for batch image conversion."""
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, current_file
    conversion_complete = pyqtSignal(list)  # list of ConversionResult
    error_occurred = pyqtSignal(str)
    
    def __init__(
        self,
        source_dir: Path,
        dest_dir: Path,
        include_subfolders: bool,
        quality: int,
        output_format: OutputFormat,
        allowed_extensions: Optional[set] = None,
        parent=None
    ):
        super().__init__(parent)
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.include_subfolders = include_subfolders
        self.quality = quality
        self.output_format = output_format
        self.allowed_extensions = allowed_extensions or INPUT_EXTENSIONS
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        """Request cancellation of the conversion process."""
        with self._lock:
            self._cancelled = True
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        with self._lock:
            return self._cancelled
    
    def find_image_files(self) -> list[Path]:
        """Find all supported image files in the source directory."""
        image_files = []
        
        if self.include_subfolders:
            for root, _, files in os.walk(self.source_dir):
                for file in files:
                    if Path(file).suffix.lower() in self.allowed_extensions:
                        image_files.append(Path(root) / file)
        else:
            for file in self.source_dir.iterdir():
                if file.is_file() and file.suffix.lower() in self.allowed_extensions:
                    image_files.append(file)
        
        return image_files
    
    def get_dest_path(self, source_file: Path) -> Path:
        """Calculate destination path, preserving folder structure."""
        relative_path = source_file.relative_to(self.source_dir)
        dest_path = self.dest_dir / relative_path.with_suffix(self.output_format.extension)
        return dest_path
    
    def convert_single_file(self, source_file: Path) -> ConversionResult:
        """Convert a single image file to the selected output format."""
        dest_path = self.get_dest_path(source_file)
        original_size = source_file.stat().st_size
        
        try:
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open and convert image
            with Image.open(source_file) as img:
                # Prepare image (handle alpha etc)
                img = prepare_image_for_save(img, self.output_format.name)
                
                # Build save parameters from format config
                save_params = {
                    'format': self.output_format.pillow_format,
                    'quality': self.quality,
                    **self.output_format.extra_params
                }
                img.save(dest_path, **save_params)
            
            converted_size = dest_path.stat().st_size
            
            return ConversionResult(
                source_path=source_file,
                dest_path=dest_path,
                original_size=original_size,
                converted_size=converted_size,
                success=True
            )
            
        except Exception as e:
            return ConversionResult(
                source_path=source_file,
                dest_path=dest_path,
                original_size=original_size,
                converted_size=0,
                success=False,
                error=str(e)
            )
    
    def run(self):
        """Execute the batch conversion process."""
        try:
            image_files = self.find_image_files()
            total_files = len(image_files)
            
            if total_files == 0:
                self.error_occurred.emit("No image files found in the selected directory.")
                return
            
            results = []
            completed = 0
            
            # Use all available CPU cores
            max_workers = os.cpu_count() or 4
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all conversion tasks
                future_to_file = {
                    executor.submit(self.convert_single_file, f): f 
                    for f in image_files
                }
                
                # Process completed conversions
                for future in as_completed(future_to_file):
                    if self.is_cancelled():
                        executor.shutdown(wait=False, cancel_futures=True)
                        return
                    
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    self.progress_updated.emit(
                        completed,
                        total_files,
                        result.source_path.name
                    )
            
            self.conversion_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"Conversion failed: {str(e)}")


class SizeEstimator:
    """Estimates file size reduction for various output formats."""
    
    # Compression ratios by format and quality level
    FORMAT_RATIOS = {
        'WebP': {10: 0.06, 20: 0.10, 30: 0.14, 40: 0.20, 50: 0.26, 60: 0.34, 70: 0.44, 80: 0.56, 90: 0.74, 100: 0.92},
        'AVIF': {10: 0.05, 20: 0.08, 30: 0.12, 40: 0.18, 50: 0.25, 60: 0.32, 70: 0.42, 80: 0.55, 90: 0.72, 100: 0.90},
        'HEIC': {10: 0.06, 20: 0.09, 30: 0.13, 40: 0.19, 50: 0.26, 60: 0.35, 70: 0.46, 80: 0.58, 90: 0.75, 100: 0.92},
        'JPEG XL': {10: 0.04, 20: 0.07, 30: 0.11, 40: 0.16, 50: 0.22, 60: 0.30, 70: 0.40, 80: 0.52, 90: 0.70, 100: 0.88},
        'JPEG': {10: 0.08, 20: 0.13, 30: 0.18, 40: 0.24, 50: 0.33, 60: 0.42, 70: 0.55, 80: 0.68, 90: 0.85, 100: 1.05},
        'PNG': {10: 0.80, 20: 0.85, 30: 0.90, 40: 0.95, 50: 1.0, 60: 1.0, 70: 1.0, 80: 1.0, 90: 1.0, 100: 1.0}, # PNG is lossless
        'TIFF': {10: 0.90, 50: 1.0, 100: 1.2},    # TIFF usually large
        'BMP': {10: 1.0, 50: 1.0, 100: 1.0},      # BMP is uncompressed
    }
    
    @classmethod
    def estimate_ratio(cls, quality: int, format_name: str = 'AVIF') -> float:
        """Estimate compression ratio for a given quality level and format."""
        ratios = cls.FORMAT_RATIOS.get(format_name, cls.FORMAT_RATIOS['AVIF'])
        
        # Find the two nearest quality levels and interpolate
        lower_q = (quality // 10) * 10
        upper_q = min(lower_q + 10, 100)
        
        if lower_q == 0:
            lower_q = 10
        
        lower_ratio = ratios.get(lower_q, 0.25)
        upper_ratio = ratios.get(upper_q, 0.25)
        
        # Linear interpolation
        if upper_q == lower_q:
            return lower_ratio
        
        t = (quality - lower_q) / (upper_q - lower_q)
        return lower_ratio + t * (upper_ratio - lower_ratio)
    
    @classmethod
    def format_size(cls, size_bytes: int) -> str:
        """Format byte size to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


class AnalysisWorker(QThread):
    """Worker thread for analyzing quality vs file size."""
    
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # {format_name: [(quality, size), ...]}
    error = pyqtSignal(str)
    
    def __init__(self, pil_image, formats_to_analyze: list, original_size: int, quality_levels: list):
        super().__init__()
        self.pil_image = pil_image
        self.formats_to_analyze = formats_to_analyze  # List of format names
        self.original_size = original_size
        self.quality_levels = quality_levels
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def _convert_single(self, format_name: str, quality: int) -> tuple:
        """Convert at a single quality level, return (format, quality, size)."""
        fmt = OUTPUT_FORMATS.get(format_name)
        if not fmt:
            return (format_name, quality, 0)
        
        # Copy image for thread safety - PIL images are not thread-safe
        img_copy = self.pil_image.copy()
        
        # Prepare for save (handle alpha for JPEG etc)
        img_copy = prepare_image_for_save(img_copy, fmt.name)
        
        buffer = BytesIO()
        save_params = {
            'format': fmt.pillow_format,
            'quality': quality,
            **fmt.extra_params
        }
        img_copy.save(buffer, **save_params)
        return (format_name, quality, buffer.tell())
    
    def run(self):
        try:
            # Build list of all tasks
            tasks = []
            for format_name in self.formats_to_analyze:
                fmt = OUTPUT_FORMATS.get(format_name)
                if fmt:
                    for quality in self.quality_levels:
                        tasks.append((format_name, quality))
            
            total_steps = len(tasks)
            results = {fmt: [] for fmt in self.formats_to_analyze}
            completed = 0
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._convert_single, fmt, q): (fmt, q)
                    for fmt, q in tasks
                }
                
                for future in as_completed(futures):
                    if self._cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return
                    
                    format_name, quality, size = future.result()
                    results[format_name].append((quality, size))
                    
                    completed += 1
                    self.progress.emit(completed, total_steps, f"Analyzing ({completed}/{total_steps})...")
            
            # Sort results by quality within each format
            sorted_results = {}
            for format_name in results:
                sorted_results[format_name] = sorted(results[format_name], key=lambda x: x[0])
            
            self.finished.emit(sorted_results)
            
        except Exception as e:
            self.error.emit(str(e))


class QualityChartDialog(QDialog):
    """Dialog showing quality vs file size chart."""
    
    FORMAT_COLORS = {
        'WebP': QColor(76, 175, 80),      # Green
        'AVIF': QColor(33, 150, 243),     # Blue
        'HEIC': QColor(255, 152, 0),      # Orange
        'JPEG XL': QColor(156, 39, 176),  # Purple
    }
    
    def __init__(self, results: dict, original_size: int, parent=None):
        super().__init__(parent)
        self.results = results  # {format_name: [(quality, size), ...]}
        self.original_size = original_size
        self.setWindowTitle("Quality vs File Size Analysis")
        self.setMinimumSize(600, 400)
        self.resize(900, 600)
        self.setSizeGripEnabled(True)  # Show resize grip
        
        # Store data point positions for hover detection
        self.data_point_positions = []  # [(x, y, format_name, quality, size), ...]
        self.hovered_point = None
        
        layout = QVBoxLayout(self)
        
        # Chart area - use custom widget for mouse tracking
        self.chart_widget = QWidget()
        self.chart_widget.setMinimumHeight(350)
        self.chart_widget.paintEvent = self.paint_chart
        self.chart_widget.mouseMoveEvent = self.on_chart_mouse_move
        self.chart_widget.setMouseTracking(True)
        layout.addWidget(self.chart_widget, 1)
        
        # Tooltip label (below chart)
        self.tooltip_label = QLabel("")
        self.tooltip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tooltip_label.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        layout.addWidget(self.tooltip_label)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        for format_name in results.keys():
            color = self.FORMAT_COLORS.get(format_name, QColor(128, 128, 128))
            color_box = QLabel()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(f"background-color: {color.name()}; border-radius: 3px;")
            legend_layout.addWidget(color_box)
            legend_layout.addWidget(QLabel(format_name))
            legend_layout.addSpacing(20)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
    
    def paint_chart(self, event):
        """Custom paint for the chart."""
        # Clear old point positions
        self.data_point_positions.clear()
        
        painter = QPainter(self.chart_widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.chart_widget.rect()
        margin_left = 80   # More room for Y-axis label
        margin_right = 30
        margin_top = 20
        margin_bottom = 60  # More room for X-axis label
        
        chart_left = margin_left
        chart_right = rect.width() - margin_right
        chart_top = margin_top
        chart_bottom = rect.height() - margin_bottom
        chart_width = chart_right - chart_left
        chart_height = chart_bottom - chart_top
        
        # Background
        painter.fillRect(rect, QColor(45, 45, 45))
        
        # Chart area background
        painter.fillRect(chart_left, chart_top, chart_width, chart_height, QColor(35, 35, 35))
        
        # Calculate max percentage from actual data (+ 10% buffer)
        max_pct = 0
        for data_points in self.results.values():
            for quality, size in data_points:
                size_pct = (size / self.original_size) * 100
                if size_pct > max_pct:
                    max_pct = size_pct
        max_pct = max(5, max_pct * 1.15)  # Add 15% buffer, minimum 5%
        
        # Calculate nice grid steps based on actual max
        step_size = max_pct / 5
        # Round step to nice number
        if step_size < 2:
            step_size = 1
        elif step_size < 5:
            step_size = 2
        elif step_size < 10:
            step_size = 5
        else:
            step_size = int((step_size + 4) // 5) * 5
        
        grid_steps = [int(step_size * i) for i in range(1, 6) if step_size * i <= max_pct + 1]
        
        # Draw grid lines and labels
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        
        # Horizontal grid (file size percentages) - from 0 to max
        for pct in grid_steps:
            y = chart_bottom - (pct / max_pct) * chart_height
            painter.drawLine(int(chart_left), int(y), int(chart_right), int(y))
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(margin_left - 45, int(y + 5), f"{pct}%")
            painter.setPen(QPen(QColor(70, 70, 70), 1))
        
        # Draw 0% baseline
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(margin_left - 35, int(chart_bottom + 5), "0%")
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        
        # Vertical grid (quality levels)
        for q in [20, 40, 60, 80, 100]:
            x = chart_left + (q / 100) * chart_width
            painter.drawLine(int(x), int(chart_top), int(x), int(chart_bottom))
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(int(x - 12), int(chart_bottom + 18), f"{q}%")
            painter.setPen(QPen(QColor(70, 70, 70), 1))
        
        # Axis labels
        painter.setPen(QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(int(chart_left + chart_width / 2 - 30), int(chart_bottom + 45), "Quality")
        
        # Rotated Y-axis label - more space from edge
        painter.save()
        painter.translate(18, int(chart_top + chart_height / 2 + 50))
        painter.rotate(-90)
        painter.drawText(0, 0, "Size (% of original)")
        painter.restore()
        
        # Draw data lines
        for format_name, data_points in self.results.items():
            if not data_points:
                continue
            
            color = self.FORMAT_COLORS.get(format_name, QColor(128, 128, 128))
            pen = QPen(color, 3)
            painter.setPen(pen)
            
            points = []
            for quality, size in data_points:
                x = chart_left + (quality / 100) * chart_width
                size_pct = (size / self.original_size) * 100
                y = chart_bottom - (size_pct / max_pct) * chart_height
                points.append((int(x), int(y), quality, size, size_pct))
            
            # Draw lines between points
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
            
            # Draw points and store positions for hover
            painter.setBrush(color)
            for x, y, quality, size, size_pct in points:
                # Highlight hovered point
                if self.hovered_point and self.hovered_point[:2] == (format_name, quality):
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawEllipse(x - 6, y - 6, 12, 12)
                    painter.setBrush(color)
                else:
                    painter.drawEllipse(x - 4, y - 4, 8, 8)
                self.data_point_positions.append((x, y, format_name, quality, size, size_pct))
        
        # Draw axes
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(int(chart_left), int(chart_bottom), int(chart_right), int(chart_bottom))
        painter.drawLine(int(chart_left), int(chart_top), int(chart_left), int(chart_bottom))
    
    def on_chart_mouse_move(self, event):
        """Handle mouse move over chart to show tooltips."""
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        
        # Find nearest point within threshold
        threshold = 15
        nearest = None
        min_dist = threshold
        
        for x, y, format_name, quality, size, size_pct in self.data_point_positions:
            dist = ((mouse_x - x) ** 2 + (mouse_y - y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = (format_name, quality, size, size_pct)
        
        if nearest:
            format_name, quality, size, size_pct = nearest
            size_str = SizeEstimator.format_size(size)
            self.tooltip_label.setText(
                f"{format_name} at {quality}% quality: {size_str} ({size_pct:.1f}% of original)"
            )
            if self.hovered_point != (format_name, quality):
                self.hovered_point = (format_name, quality)
                self.chart_widget.update()
        else:
            if self.tooltip_label.text():
                self.tooltip_label.setText("")
            if self.hovered_point:
                self.hovered_point = None
                self.chart_widget.update()


class ComparisonWidget(QWidget):
    """Widget for side-by-side image comparison with draggable divider and zoom."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap: Optional[QPixmap] = None
        self.converted_pixmap: Optional[QPixmap] = None
        self.input_label = "Original"
        self.output_label = "Converted"
        self.left_quality = 0
        self.right_quality = 0
        self.is_rendering = False
        self.divider_position = 0.5  # 0.0 to 1.0
        self.dragging = False
        self.panning = False
        self.last_pan_pos = None
        self.zoom_level = 1.0  # 1.0 = fit to widget
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self._opacity = 1.0
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def setOpacity(self, opacity: float):
        """Set opacity for the widget (used for disabled state)."""
        self._opacity = max(0.0, min(1.0, opacity))
        self.update()
    
    def set_images(self, left: Optional[QPixmap], right: Optional[QPixmap],
                   input_label: str = "Original", output_label: str = "Converted",
                   left_quality: int = 0, right_quality: int = 0):
        """Set the left and right images with labels."""
        self.original_pixmap = left
        self.converted_pixmap = right
        self.input_label = input_label
        self.output_label = output_label
        self.left_quality = left_quality
        self.right_quality = right_quality
        self.is_rendering = False
        self.update()
    
    def set_rendering(self, rendering: bool):
        """Set rendering state to show overlay."""
        self.is_rendering = rendering
        self.update()
    
    def clear(self):
        """Clear the images."""
        self.original_pixmap = None
        self.converted_pixmap = None
        self.input_label = "Original"
        self.output_label = "Converted"
        self.is_rendering = False
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update()
    
    def reset_zoom(self):
        """Reset zoom and pan to default."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update()
    
    def paintEvent(self, event):
        """Paint the comparison view with divider."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        rect = self.rect()
        
        # Rendering overlay
        if self.is_rendering:
            painter.fillRect(rect, QColor(30, 30, 30))
            painter.setPen(QColor(200, 200, 200))
            font = QFont()
            font.setPointSize(17)
            font.setBold(True)
            painter.setFont(font)
            message = getattr(self, 'rendering_message', 'Rendering preview...')
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, message)
            return
        
        if not self.original_pixmap or not self.converted_pixmap:
            # Draw placeholder
            painter.fillRect(rect, QColor(40, 40, 40))
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Select source folder to see preview")
            return
        
        # Calculate base scale to fit widget
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        
        base_scale = min(rect.width() / img_width, rect.height() / img_height)
        scale = base_scale * self.zoom_level
        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)
        
        # Center the image with pan offset
        x_offset = (rect.width() - scaled_width) // 2 + int(self.pan_offset_x)
        y_offset = (rect.height() - scaled_height) // 2 + int(self.pan_offset_y)
        
        # Calculate visible region in source image coordinates
        # This prevents scaling massive images at high zoom
        visible_left = max(0, -x_offset)
        visible_top = max(0, -y_offset)
        visible_right = min(scaled_width, rect.width() - x_offset)
        visible_bottom = min(scaled_height, rect.height() - y_offset)
        
        # Convert to source image coordinates
        src_left = int(visible_left / scale)
        src_top = int(visible_top / scale)
        src_right = min(img_width, int(visible_right / scale) + 1)
        src_bottom = min(img_height, int(visible_bottom / scale) + 1)
        src_width = src_right - src_left
        src_height = src_bottom - src_top
        
        if src_width <= 0 or src_height <= 0:
            return
        
        # Calculate destination size for the visible portion
        dest_width = int(src_width * scale)
        dest_height = int(src_height * scale)
        dest_x = x_offset + int(src_left * scale)
        dest_y = y_offset + int(src_top * scale)
        
        # Extract and scale only the visible portion
        visible_original = self.original_pixmap.copy(src_left, src_top, src_width, src_height)
        visible_converted = self.converted_pixmap.copy(src_left, src_top, src_width, src_height)
        
        # Use fast transformation for large zoom, smooth for small
        transform_mode = (Qt.TransformationMode.FastTransformation 
                         if self.zoom_level > 4.0 
                         else Qt.TransformationMode.SmoothTransformation)
        
        scaled_original = visible_original.scaled(
            dest_width, dest_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            transform_mode
        )
        scaled_converted = visible_converted.scaled(
            dest_width, dest_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            transform_mode
        )
        
        # Calculate divider x position within the image area
        divider_x = x_offset + int(scaled_width * self.divider_position)
        
        # Draw converted on right side first
        painter.drawPixmap(dest_x, dest_y, scaled_converted)
        
        # Draw original on left side - clipped to divider
        left_rect = QRect(dest_x, dest_y, divider_x - dest_x, dest_height)
        painter.setClipRect(left_rect)
        painter.drawPixmap(dest_x, dest_y, scaled_original)
        painter.setClipping(False)
        
        # Draw divider line
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(divider_x, y_offset, divider_x, y_offset + scaled_height)
        
        # Draw divider handle
        handle_size = 30
        handle_y = y_offset + scaled_height // 2 - handle_size // 2
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(divider_x - handle_size // 2, handle_y, handle_size, handle_size)
        
        # Draw arrows on handle
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        arrow_y = handle_y + handle_size // 2
        # Left arrow
        painter.drawLine(divider_x - 8, arrow_y, divider_x - 3, arrow_y - 4)
        painter.drawLine(divider_x - 8, arrow_y, divider_x - 3, arrow_y + 4)
        # Right arrow
        painter.drawLine(divider_x + 8, arrow_y, divider_x + 3, arrow_y - 4)
        painter.drawLine(divider_x + 8, arrow_y, divider_x + 3, arrow_y + 4)
        
        # Draw labels as pills in fixed corners (not affected by zoom/pan)
        # Build label text with quality if not original
        if self.left_quality > 0:
            left_text = f"{self.input_label} ({self.left_quality}%)"
        else:
            left_text = self.input_label
        
        if self.right_quality > 0:
            right_text = f"{self.output_label} ({self.right_quality}%)"
        else:
            right_text = self.output_label
        
        # Calculate right label width dynamically
        font = QFont()
        font.setBold(True)
        font.setPointSize(15)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        right_label_width = metrics.horizontalAdvance(right_text) + 40  # padding
        
        self.draw_pill_label(painter, left_text, 15, 15)
        self.draw_pill_label(painter, right_text, rect.width() - right_label_width, 15)
        
        # Zoom indicator at bottom
        if self.zoom_level != 1.0:
            zoom_text = f"Zoom: {self.zoom_level:.1f}x (scroll=zoom, right-drag=pan, dbl-click=reset)"
            self.draw_pill_label(painter, zoom_text, 15, rect.height() - 45, smaller=True)
            
        # Draw disabled overlay if opacity < 1.0
        if self._opacity < 1.0:
            painter.setBrush(QColor(0, 0, 0, int(255 * (1.0 - self._opacity))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            
            # Draw "Converting..." text if disabled
            if not self.isEnabled():
                painter.setPen(QColor(255, 255, 255))
                font = QFont()
                font.setPointSize(24)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Converting...")
    
    def draw_pill_label(self, painter: QPainter, text: str, x: int, y: int, smaller: bool = False):
        """Draw a label with pill-shaped background."""
        font = QFont()
        font.setBold(True)
        font.setPointSize(12 if smaller else 15)
        painter.setFont(font)
        
        # Measure text
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        
        # Draw pill background
        padding_x = 12 if smaller else 16
        padding_y = 6 if smaller else 10
        pill_rect = QRect(
            x, y,
            text_width + padding_x * 2,
            text_height + padding_y * 2
        )
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 200))  # Semi-transparent black
        painter.drawRoundedRect(pill_rect, 16, 16)
        
        # Draw text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            x + padding_x,
            y + padding_y + metrics.ascent(),
            text
        )
    
    def mousePressEvent(self, event):
        """Handle mouse press for divider dragging or panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.update_divider_position(event.position().x())
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.panning = True
            self.last_pan_pos = event.position()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for divider dragging or panning."""
        if self.panning and self.last_pan_pos:
            delta_x = event.position().x() - self.last_pan_pos.x()
            delta_y = event.position().y() - self.last_pan_pos.y()
            self.pan_offset_x += delta_x
            self.pan_offset_y += delta_y
            self.last_pan_pos = event.position()
            self.update()
        elif self.original_pixmap and self.converted_pixmap:
            # Change cursor when near divider
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
            
            if self.dragging:
                self.update_divider_position(event.position().x())
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.dragging = False
        self.panning = False
        self.last_pan_pos = None
        if self.original_pixmap and self.converted_pixmap:
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
    
    def mouseDoubleClickEvent(self, event):
        """Reset zoom on double-click."""
        self.reset_zoom()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming centered on cursor."""
        if not self.original_pixmap:
            return
        
        # Get mouse position
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        
        # Calculate old zoom parameters
        rect = self.rect()
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        base_scale = min(rect.width() / img_width, rect.height() / img_height)
        old_scale = base_scale * self.zoom_level
        
        # Get zoom delta
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1 / 1.15
        
        # Apply zoom with limits
        old_zoom = self.zoom_level
        new_zoom = self.zoom_level * zoom_factor
        self.zoom_level = max(0.5, min(33.0, new_zoom))
        
        # Calculate new scale
        new_scale = base_scale * self.zoom_level
        
        # Adjust pan offset to keep cursor position fixed
        # Convert mouse position to image-relative coordinates before zoom
        old_center_x = rect.width() / 2 + self.pan_offset_x
        old_center_y = rect.height() / 2 + self.pan_offset_y
        
        # Scale the pan offset
        scale_change = self.zoom_level / old_zoom
        
        # Calculate offset adjustment to zoom toward cursor
        self.pan_offset_x = (self.pan_offset_x - (mouse_x - rect.width() / 2)) * scale_change + (mouse_x - rect.width() / 2)
        self.pan_offset_y = (self.pan_offset_y - (mouse_y - rect.height() / 2)) * scale_change + (mouse_y - rect.height() / 2)
        
        self.update()
    
    def update_divider_position(self, x: float):
        """Update divider position based on mouse x coordinate."""
        if not self.original_pixmap:
            return
        
        # Calculate image bounds with zoom
        rect = self.rect()
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        base_scale = min(rect.width() / img_width, rect.height() / img_height)
        scale = base_scale * self.zoom_level
        scaled_width = int(img_width * scale)
        x_offset = (rect.width() - scaled_width) // 2 + int(self.pan_offset_x)
        
        # Calculate position within image
        relative_x = x - x_offset
        self.divider_position = max(0.0, min(1.0, relative_x / scaled_width))
        self.update()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.conversion_worker: Optional[ConversionWorker] = None
        self.image_files: list[Path] = []
        self.current_original_pixmap: Optional[QPixmap] = None
        self.current_pil_image: Optional[Image.Image] = None
        self.preview_update_timer = QTimer()
        self.preview_update_timer.setSingleShot(True)
        self.preview_update_timer.timeout.connect(self.update_preview_now)
        
        # Analysis cache
        self.cached_analysis_results = None  # {format_name: [(quality, size), ...]}
        self.cached_analysis_params = None  # (file_path, formats, detail_index, original_size)
    
        self.last_update_time = 0
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Mass Photo Converter")
        self.setMinimumSize(1100, 700)
        
        # Scale to 75% of screen size
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        width = int(screen_geometry.width() * 0.75)
        height = int(screen_geometry.height() * 0.75)
        self.resize(width, height)
        
        # Center the window
        x = (screen_geometry.width() - width) // 2
        y = (screen_geometry.height() - height) // 2
        self.move(screen_geometry.x() + x, screen_geometry.y() + y)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout is Horizontal (Left Panel = Controls, Right Panel = Preview)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align panels to top
        
        # ---------------------------------------------------------------------
        # Left Panel (Controls) - 1/3 Width
        # ---------------------------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(20)
        
        # Header row: Title on left, Guide on right
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(25, 0, 0, 0)  # Left padding
        
        # Title (3 lines)
        title_label = QLabel("Mass<br>Photo<br>Converter")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #FFFFFF; line-height: 1.1;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title_label)
        
        # Quick guide (to the right of title)
        guide_text = (
            "1. Select a <b>source folder</b> with images<br>"
            "2. Choose an <b>output folder</b> for converted files<br>"
            "3. Adjust <b>quality</b> and preview results<br>"
            "4. Click <b>Start conversion</b> when ready"
        )
        guide_label = QLabel(guide_text)
        guide_label.setStyleSheet("color: #AAAAAA; font-size: 13px; padding: 10px 20px;")
        guide_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        guide_label.setWordWrap(True)
        header_layout.addWidget(guide_label, 1)
        
        left_layout.addLayout(header_layout)
        
        # 1. Folder Selection
        folder_group = QGroupBox("Folders")
        folder_layout = QVBoxLayout(folder_group)
        
        # Source
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select folder...")
        self.source_edit.textChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.source_edit)
        self.source_btn = QPushButton("Browse...")
        self.source_btn.clicked.connect(self.browse_source)
        source_layout.addWidget(self.source_btn)
        folder_layout.addLayout(source_layout)
        
        # Dest
        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel("Output:"))
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Select output folder...")
        dest_layout.addWidget(self.dest_edit)
        self.dest_btn = QPushButton("Browse...")
        self.dest_btn.clicked.connect(self.browse_dest)
        dest_layout.addWidget(self.dest_btn)
        folder_layout.addLayout(dest_layout)
        
        # Subfolders
        self.subfolder_check = QCheckBox("Include subfolders")
        self.subfolder_check.setChecked(True)
        self.subfolder_check.stateChanged.connect(self.on_source_changed)
        folder_layout.addWidget(self.subfolder_check)

        # Filter by type
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter source:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Supported Images", None)
        self.filter_combo.addItem("JPEG Images (*.jpg, *.jpeg)", {'.jpg', '.jpeg'})
        self.filter_combo.addItem("PNG Images (*.png)", {'.png'})
        self.filter_combo.addItem("TIFF Images (*.tiff, *.tif)", {'.tiff', '.tif'})
        self.filter_combo.addItem("BMP Images (*.bmp)", {'.bmp'})
        self.filter_combo.addItem("WebP Images (*.webp)", {'.webp'})
        self.filter_combo.addItem("AVIF Images (*.avif)", {'.avif'})
        self.filter_combo.addItem("HEIC Images (*.heic)", {'.heic'})
        self.filter_combo.currentIndexChanged.connect(self.on_source_changed)
        filter_layout.addWidget(self.filter_combo, 1)
        folder_layout.addLayout(filter_layout)
        
        left_layout.addWidget(folder_group)
        
        # 2. Preview Settings (Extracted from old Preview Group)
        preview_settings_group = QGroupBox("Preview settings")
        preview_settings_layout = QVBoxLayout(preview_settings_group)
        
        # File & Load
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self.preview_combo = QComboBox()
        self.preview_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        file_row.addWidget(self.preview_combo, 1)
        self.load_preview_btn = QPushButton("Load")
        self.load_preview_btn.clicked.connect(self.on_load_preview_clicked)
        file_row.addWidget(self.load_preview_btn)
        preview_settings_layout.addLayout(file_row)
        
        # Comparison logic
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Left:"))
        self.left_format_combo = QComboBox()
        self.left_format_combo.addItem("Original")
        for name in OUTPUT_FORMATS.keys():
            self.left_format_combo.addItem(name)
        self.left_format_combo.setCurrentText("Original")
        self.left_format_combo.currentTextChanged.connect(self.on_comparison_format_changed)
        self.left_format_combo.currentTextChanged.connect(self.check_analysis_cache)
        compare_row.addWidget(self.left_format_combo)
        
        compare_row.addWidget(QLabel("Right:"))
        self.right_format_combo = QComboBox()
        self.right_format_combo.addItem("Original")
        for name in OUTPUT_FORMATS.keys():
            self.right_format_combo.addItem(name)
        self.right_format_combo.setCurrentText("AVIF")
        self.right_format_combo.currentTextChanged.connect(self.on_comparison_format_changed)
        self.right_format_combo.currentTextChanged.connect(self.check_analysis_cache)
        compare_row.addWidget(self.right_format_combo)
        preview_settings_layout.addLayout(compare_row)
        
        # Quality Slider (Full width in this column)
        quality_layout = QVBoxLayout()
        quality_header = QHBoxLayout()
        quality_header.addWidget(QLabel("Quality setting:"))
        self.quality_label = QLabel("80%")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        quality_header.addWidget(self.quality_label)
        quality_layout.addLayout(quality_header)
        
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setMinimum(5)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setSingleStep(5)
        self.quality_slider.setPageStep(5)
        self.quality_slider.setValue(80)
        self.quality_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 30px;
                background: #303030;
                border-radius: 6px;
                border: 1px solid #505050;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1565C0, stop:1 #2196F3);
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                width: 20px;
                height: 34px;
                margin: -2px 0;
                background: #FFFFFF;
                border-radius: 4px;
            }
            QSlider::handle:horizontal:hover {
                background: #E0E0E0;
            }
        """)
        self.quality_slider.valueChanged.connect(self.on_quality_value_changed)
        self.quality_slider.sliderReleased.connect(self.on_quality_slider_released)
        quality_layout.addWidget(self.quality_slider)
        preview_settings_layout.addLayout(quality_layout)
        
        # Analysis Row
        analysis_row = QHBoxLayout()
        self.analysis_detail_combo = QComboBox()
        self.analysis_detail_combo.addItems(["Low (8)", "Medium (15)", "High (30)"])
        self.analysis_detail_combo.setCurrentIndex(1)
        self.analysis_detail_combo.currentIndexChanged.connect(self.check_analysis_cache)
        analysis_row.addWidget(QLabel("Analysis detail:"))
        analysis_row.addWidget(self.analysis_detail_combo)
        
        self.analyze_btn = QPushButton("📊 Analyze")
        self.analyze_btn.setProperty("class", "primary")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.on_analyze_clicked)
        analysis_row.addWidget(self.analyze_btn)
        preview_settings_layout.addLayout(analysis_row)
        
        left_layout.addWidget(preview_settings_group)
        
        # 3. Output Settings
        output_group = QGroupBox("Target format")
        output_layout = QVBoxLayout(output_group)
        
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Convert to:"))
        self.format_combo = QComboBox()
        for name in OUTPUT_FORMATS.keys():
            self.format_combo.addItem(name)
        self.format_combo.setCurrentText('AVIF')
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        fmt_row.addWidget(self.format_combo)
        output_layout.addLayout(fmt_row)
        
        self.estimate_label = QLabel("Est. Size: --")
        output_layout.addWidget(self.estimate_label)
        
        left_layout.addWidget(output_group)

        # 4. Progress & Action
        action_group = QGroupBox("Execution")
        action_layout = QVBoxLayout(action_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        action_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.addWidget(self.progress_label)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start conversion")
        self.start_btn.setProperty("class", "primary")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.clicked.connect(self.start_conversion)
        btn_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(45)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        action_layout.addLayout(btn_layout)
        
        left_layout.addWidget(action_group)
        
        
        # ---------------------------------------------------------------------
        # Right Panel (Preview) - 2/3 Width
        # ---------------------------------------------------------------------
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #000000; border-radius: 8px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Comparison Widget
        self.comparison_widget = ComparisonWidget()
        right_layout.addWidget(self.comparison_widget, 1)
        
        # Info Label (Overlay or below?) -> Below for now
        self.size_info_label = QLabel("Load a preview to compare results")
        self.size_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_info_label.setStyleSheet("color: #888888; padding: 5px;")
        right_layout.addWidget(self.size_info_label)

        # Add panels to Main Layout
        main_layout.addWidget(left_panel, 1)  # 1/3
        main_layout.addWidget(right_panel, 2) # 2/3

    
    def browse_source(self):
        """Open dialog to select source folder."""
        # Use current value if set, otherwise let Windows use last location
        start_dir = self.source_edit.text() or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", start_dir
        )
        if folder:
            self.source_edit.setText(folder)
    
    def browse_dest(self):
        """Open dialog to select destination folder."""
        # Use current value if set, otherwise let Windows use last location
        start_dir = self.dest_edit.text() or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir
        )
        if folder:
            self.dest_edit.setText(folder)
    
    def on_source_changed(self):
        """Handle source folder change - update file list and preview."""
        source_path = self.source_edit.text()
        
        # Update current allowed extensions for counter
        self.current_allowed_extensions = self.filter_combo.currentData() or INPUT_EXTENSIONS
        
        if not source_path or not Path(source_path).exists():
            self.image_files = []
            self.preview_combo.clear()
            self.comparison_widget.clear()
            self.estimate_label.setText("Select a source folder to see total size estimate")
            self.size_info_label.setText("")
            return
        
        # Find image files
        folder = Path(source_path)
        include_subfolders = self.subfolder_check.isChecked()
        allowed_extensions = self.filter_combo.currentData() or INPUT_EXTENSIONS
        
        self.image_files = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if Path(file).suffix.lower() in allowed_extensions:
                        self.image_files.append(Path(root) / file)
        else:
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in allowed_extensions:
                    self.image_files.append(file)
        
        # Sort by name
        self.image_files.sort(key=lambda p: p.name.lower())
        
        # Update combo box - limit to first 20 for preview
        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        for f in self.image_files[:20]:
            relative = f.relative_to(folder) if include_subfolders else f.name
            self.preview_combo.addItem(str(relative), str(f))  # Store full path as data
        self.preview_combo.blockSignals(False)
        
        # Update estimate
        self.update_estimate()
        
        # Don't auto-load preview - user must click Load Preview button
        # This prevents crashes with large folders
        if self.image_files:
            self.preview_combo.setCurrentIndex(0)
            self.comparison_widget.clear()
            self.size_info_label.setText("Click 'Load Preview' to see comparison")
    
    def on_load_preview_clicked(self):
        """Handle Load Preview button click."""
        index = self.preview_combo.currentIndex()
        if index < 0:
            return
        # Get path from combo data, not from png_files index
        file_path_str = self.preview_combo.currentData()
        if not file_path_str:
            return
        self.load_preview_image(Path(file_path_str))
        self.preview_loaded = True
    
    def on_preview_file_changed(self, index: int):
        """Handle preview file selection change - only load if already loaded once."""
        pass  # Don't auto-load on dropdown change
    
    def load_preview_image(self, file_path: Path):
        """Load an image for preview."""
        try:
            # Store original file info before opening
            self.current_original_size = file_path.stat().st_size
            self.current_input_format = file_path.suffix.upper().lstrip('.')
            if self.current_input_format in ('TIF',):
                self.current_input_format = 'TIFF'
            if self.current_input_format in ('JPG',):
                self.current_input_format = 'JPEG'
            
            with Image.open(file_path) as img:
                # Convert to RGB/RGBA for display (no size limit)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    self.current_pil_image = img.convert('RGBA')
                elif img.mode != 'RGB':
                    self.current_pil_image = img.convert('RGB')
                else:
                    self.current_pil_image = img.copy()
                
                # Convert to pixmap
                self.current_original_pixmap = self.pil_to_pixmap(self.current_pil_image)
                
                # Reset zoom when loading new image
                self.comparison_widget.reset_zoom()
                
                # Enable analyze button
                self.analyze_btn.setEnabled(True)
                self.check_analysis_cache()
                
                # Generate converted preview
                self.update_preview_now()
                
        except Exception as e:
            self.size_info_label.setText(f"Error loading image: {e}")
            self.current_pil_image = None
            self.current_original_pixmap = None
            self.comparison_widget.clear()
    
    def on_quality_value_changed(self, value: int):
        """Handle quality slider value changes - snap to 5% increments."""
        # Snap to nearest 5%
        snapped = max(5, round(value / 5) * 5)
        if snapped != value:
            self.quality_slider.blockSignals(True)
            self.quality_slider.setValue(snapped)
            self.quality_slider.blockSignals(False)
            value = snapped
        
        self.quality_label.setText(f"{value}%")
        self.update_estimate()
        # If not actively dragging, trigger render after a short delay
        # (This handles clicks and keyboard changes)
        if not self.quality_slider.isSliderDown():
            self.preview_update_timer.start(100)
    
    def on_quality_slider_released(self):
        """Handle quality slider release after dragging - snap and trigger render."""
        # Ensure final value is snapped
        value = self.quality_slider.value()
        snapped = max(5, round(value / 5) * 5)
        if snapped != value:
            self.quality_slider.setValue(snapped)
        self.preview_update_timer.start(50)  # Short delay then render
    
    def on_analyze_clicked(self):
        """Handle Analyze button click - start quality analysis or view cached."""
        if not self.current_pil_image:
            return
        
        # Determine which formats to analyze (non-Original selections)
        formats = []
        left = self.left_format_combo.currentText()
        right = self.right_format_combo.currentText()
        
        if left != "Original" and left not in formats:
            formats.append(left)
        if right != "Original" and right not in formats:
            formats.append(right)
        
        if not formats:
            QMessageBox.information(
                self, "No Format Selected",
                "Please select at least one output format (not Original) to analyze."
            )
            return
        
        # Get quality levels from dropdown
        # All levels are denser at higher quality where compression differences matter most
        detail_index = self.analysis_detail_combo.currentIndex()
        if detail_index == 0:  # Low (8)
            quality_levels = [5, 20, 40, 60, 80, 90, 95, 98]
        elif detail_index == 1:  # Medium (15)
            quality_levels = [5, 15, 25, 35, 45, 55, 65, 75, 82, 87, 90, 93, 95, 97, 98]
        else:  # High (30)
            quality_levels = [5, 8, 11, 14, 17, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,
                              60, 64, 68, 72, 76, 80, 84, 87, 90, 92, 94, 95, 96, 97, 98]
        
        # Check cache
        current_params = (
            self.preview_combo.currentData(),  # File path
            tuple(sorted(formats)),            # Formats
            detail_index,                      # Detail level
            self.current_original_size         # Original size (implicit file check)
        )
        
        if (self.cached_analysis_results and 
            self.cached_analysis_params == current_params):
            # Cache hit - show chart immediately
            dialog = QualityChartDialog(self.cached_analysis_results, self.current_original_size, self)
            dialog.exec()
            return

        num_renders = len(formats) * len(quality_levels)
        
        # Confirmation dialog
        result = QMessageBox.question(
            self, "Quality Analysis",
            f"This will render the image {num_renders} times to analyze quality vs size.\n"
            f"Formats: {', '.join(formats)}\n\n"
            "This may take a while. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if result != QMessageBox.StandardButton.Yes:
            return
        
        # Store params for caching on completion
        self._pending_analysis_params = current_params
        
        # Create progress dialog
        self.analysis_progress = QProgressDialog(
            "Analyzing quality levels...", "Cancel", 0, num_renders, self
        )
        self.analysis_progress.setWindowTitle("Analyzing")
        self.analysis_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.analysis_progress.setMinimumDuration(0)
        self.analysis_progress.setValue(0)
        
        # Create and start worker
        self.analysis_worker = AnalysisWorker(
            self.current_pil_image,
            formats,
            self.current_original_size,
            quality_levels
        )
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_progress.canceled.connect(self.on_analysis_cancelled)
        
        self.analysis_worker.start()
    
    def on_analysis_progress(self, current: int, total: int, message: str):
        """Update analysis progress dialog."""
        if hasattr(self, 'analysis_progress') and self.analysis_progress:
            self.analysis_progress.setLabelText(message)
            self.analysis_progress.setValue(current)
    
    def on_analysis_finished(self, results: dict):
        """Handle analysis completion - show chart and cache results."""
        if hasattr(self, 'analysis_progress') and self.analysis_progress:
            self.analysis_progress.close()
        
        # Cache results
        if hasattr(self, '_pending_analysis_params'):
            self.cached_analysis_results = results
            self.cached_analysis_params = self._pending_analysis_params
            del self._pending_analysis_params
            self.check_analysis_cache()  # Update button text
        
        # Show chart dialog
        dialog = QualityChartDialog(results, self.current_original_size, self)
        dialog.exec()
    
    def on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        if hasattr(self, 'analysis_progress') and self.analysis_progress:
            self.analysis_progress.close()
        QMessageBox.critical(self, "Analysis Error", f"Error during analysis: {error_msg}")
    
    def on_analysis_cancelled(self):
        """Handle analysis cancellation."""
        if hasattr(self, 'analysis_worker') and self.analysis_worker:
            self.analysis_worker.cancel()
            
    def check_analysis_cache(self, *args):
        """Check if current analysis settings match cache and update button text."""
        if not self.cached_analysis_results or not self.cached_analysis_params:
            self.analyze_btn.setText("📊 Start analysis")
            return
            
        # Reconstruct current params to check against cache
        formats = []
        left = self.left_format_combo.currentText()
        right = self.right_format_combo.currentText()
        
        if left != "Original" and left not in formats:
            formats.append(left)
        if right != "Original" and right not in formats:
            formats.append(right)
            
        current_params = (
            self.preview_combo.currentData(),  # File path
            tuple(sorted(formats)),            # Formats
            self.analysis_detail_combo.currentIndex(),  # Detail level
            self.current_original_size         # Original size
        )
        
        if self.cached_analysis_params == current_params:
            self.analyze_btn.setText("📊 View analysis")
        else:
            self.analyze_btn.setText("📊 Start analysis")
    
    def on_format_changed(self, format_name: str):
        """Handle output format change."""
        fmt = OUTPUT_FORMATS.get(format_name)
        if fmt:
            self.quality_slider.setValue(fmt.default_quality)
        # Sync right comparison dropdown with main output format
        self.right_format_combo.blockSignals(True)
        self.right_format_combo.setCurrentText(format_name)
        self.right_format_combo.blockSignals(False)
        # Clear actual ratio since format changed
        self.actual_compression_ratio = 0
        self.update_estimate()
        self.preview_update_timer.start(150)
    
    def on_comparison_format_changed(self, _format_name: str):
        """Handle comparison format dropdown changes."""
        # Debounce preview updates
        self.preview_update_timer.start(150)
    
    def _convert_to_format(self, format_name: str, quality: int) -> tuple[QPixmap, int, str]:
        """Convert current image to specified format, return (pixmap, size, label)."""
        if format_name == "Original":
            return self.current_original_pixmap, self.current_original_size, getattr(self, 'current_input_format', 'Original')
        
        fmt = OUTPUT_FORMATS.get(format_name, OUTPUT_FORMATS['AVIF'])
        output_buffer = BytesIO()
        save_params = {
            'format': fmt.pillow_format,
            'quality': quality,
            **fmt.extra_params
        }

        
        # Prepare image (handle alpha etc) - work on a copy to not modify self.current_pil_image
        img_to_save = self.current_pil_image.copy()
        img_to_save = prepare_image_for_save(img_to_save, fmt.name)
        
        img_to_save.save(output_buffer, **save_params)
        converted_size = output_buffer.tell()
        
        output_buffer.seek(0)
        with Image.open(output_buffer) as converted_img:
            mode = 'RGBA' if converted_img.mode == 'RGBA' else 'RGB'
            pixmap = self.pil_to_pixmap(converted_img.convert(mode))
        
        return pixmap, converted_size, format_name
    
    def update_preview_now(self):
        """Actually update the preview (called after debounce timer)."""
        if not self.current_pil_image or not self.current_original_pixmap:
            return
        
        # Guard against double-rendering
        if getattr(self, '_is_rendering', False):
            return
        self._is_rendering = True
        
        # Show rendering overlay
        self.comparison_widget.set_rendering(True)
        QApplication.processEvents()  # Force UI update
        
        try:
            quality = self.quality_slider.value()
            left_format = self.left_format_combo.currentText()
            right_format = self.right_format_combo.currentText()
            
            # Convert left format with progress
            self.comparison_widget.rendering_message = f"Rendering {left_format}..."
            self.comparison_widget.update()
            QApplication.processEvents()
            left_pixmap, left_size, left_label = self._convert_to_format(left_format, quality)
            
            # Convert right format with progress
            self.comparison_widget.rendering_message = f"Rendering {right_format}..."
            self.comparison_widget.update()
            QApplication.processEvents()
            right_pixmap, right_size, right_label = self._convert_to_format(right_format, quality)
            
            # Determine quality values for labels (0 = original, else quality)
            left_quality = 0 if left_format == "Original" else quality
            right_quality = 0 if right_format == "Original" else quality
            
            # Update comparison widget
            self.comparison_widget.set_images(
                left_pixmap, right_pixmap,
                input_label=left_label, output_label=right_label,
                left_quality=left_quality, right_quality=right_quality
            )
            
            # Calculate savings for display
            original_size = self.current_original_size
            
            # Build size info string
            left_str = SizeEstimator.format_size(left_size)
            right_str = SizeEstimator.format_size(right_size)
            
            if left_format == "Original":
                left_info = f"{left_label}: {left_str}"
            else:
                if left_size > original_size:
                    increase_pct = (left_size / original_size - 1) * 100
                    left_info = f"{left_label}: {left_str} ({increase_pct:.0f}% bigger)"
                else:
                    left_reduction = (1 - left_size / original_size) * 100
                    left_info = f"{left_label}: {left_str} ({left_reduction:.0f}% savings)"
            
            if right_format == "Original":
                right_info = f"{right_label}: {right_str}"
            else:
                if right_size > original_size:
                    increase_pct = (right_size / original_size - 1) * 100
                    right_info = f"{right_label}: {right_str} ({increase_pct:.0f}% bigger)"
                else:
                    right_reduction = (1 - right_size / original_size) * 100
                    right_info = f"{right_label}: {right_str} ({right_reduction:.0f}% savings)"
            
            self.size_info_label.setText(f"{left_info}   |   {right_info}")
            
            # Store ratio for the main output format (right side by default)
            if right_format != "Original":
                self.actual_compression_ratio = right_size / original_size
            else:
                self.actual_compression_ratio = 0
            
            self.update_estimate()
            
        except Exception as e:
            self.size_info_label.setText(f"Error generating preview: {e}")
        finally:
            self._is_rendering = False
    
    def pil_to_pixmap(self, pil_image: Image.Image) -> QPixmap:
        """Convert a PIL Image to QPixmap."""
        # Ensure image is in correct mode
        if pil_image.mode == 'RGBA':
            qimage_format = QImage.Format.Format_RGBA8888
            raw_mode = 'RGBA'
        else:
            qimage_format = QImage.Format.Format_RGB888
            raw_mode = 'RGB'
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        
        # Get image data
        width = pil_image.width
        height = pil_image.height
        data = pil_image.tobytes('raw', raw_mode)
        
        # Create QImage with proper stride calculation
        bytes_per_line = len(data) // height
        qimage = QImage(data, width, height, bytes_per_line, qimage_format)
        
        # IMPORTANT: Copy immediately to avoid dangling reference to data buffer
        return QPixmap.fromImage(qimage.copy())
    
    def count_image_files(self, folder: Path, include_subfolders: bool) -> tuple[int, int]:
        """Count image files and total size in folder."""
        count = 0
        total_size = 0
        
        try:
            if include_subfolders:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if Path(file).suffix.lower() in getattr(self, 'current_allowed_extensions', INPUT_EXTENSIONS):
                            file_path = Path(root) / file
                            count += 1
                            total_size += file_path.stat().st_size
            else:
                for file in folder.iterdir():
                    if file.is_file() and file.suffix.lower() in getattr(self, 'current_allowed_extensions', INPUT_EXTENSIONS):
                        count += 1
                        total_size += file.stat().st_size
        except Exception:
            pass
        
        return count, total_size
    
    def update_estimate(self):
        """Update the size estimate display."""
        source_path = self.source_edit.text()
        
        if not source_path or not Path(source_path).exists():
            self.estimate_label.setText("Select a source folder to see total size estimate")
            return
        
        folder = Path(source_path)
        include_subfolders = self.subfolder_check.isChecked()
        quality = self.quality_slider.value()
        format_name = self.format_combo.currentText()
        
        file_count, total_size = self.count_image_files(folder, include_subfolders)
        
        if file_count == 0:
            self.estimate_label.setText("No image files found in selected folder")
            return
        
        if hasattr(self, 'actual_compression_ratio') and self.actual_compression_ratio > 0:
            ratio = self.actual_compression_ratio
            estimated_size = int(total_size * ratio)
            estimate_type = "based on preview"
        else:
            ratio = SizeEstimator.estimate_ratio(quality, format_name)
            estimated_size = int(total_size * ratio)
            estimate_type = "estimated"
        
        original_str = SizeEstimator.format_size(total_size)
        estimated_str = SizeEstimator.format_size(estimated_size)
        
        if estimated_size > total_size:
            increase_pct = (estimated_size / total_size - 1) * 100
            diff_str = f"~{increase_pct:.0f}% bigger"
        else:
            reduction_pct = (1 - estimated_size / total_size) * 100
            diff_str = f"~{reduction_pct:.0f}% savings"

        self.estimate_label.setText(
            f"📁 All {file_count} files ({estimate_type}): {original_str} → ~{estimated_str} ({diff_str})"
        )
    
    def set_ui_enabled(self, enabled: bool):
        """Enable or disable UI elements during conversion."""
        self.source_edit.setEnabled(enabled)
        self.dest_edit.setEnabled(enabled)
        self.source_btn.setEnabled(enabled)
        self.dest_btn.setEnabled(enabled)
        self.subfolder_check.setEnabled(enabled)
        self.filter_combo.setEnabled(enabled)  # Added filter combo
        self.format_combo.setEnabled(enabled)
        self.left_format_combo.setEnabled(enabled)
        self.right_format_combo.setEnabled(enabled)
        self.quality_slider.setEnabled(enabled)
        self.preview_combo.setEnabled(enabled)
        self.load_preview_btn.setEnabled(enabled)  # Added load button
        
        # Analyze button state depends on enabled AND if we have an image
        if enabled:
            self.analyze_btn.setEnabled(self.current_pil_image is not None)
        else:
            self.analyze_btn.setEnabled(False)
            
        self.comparison_widget.setEnabled(enabled) # Disable preview interaction
        self.comparison_widget.setOpacity(1.0 if enabled else 0.5) # Explicitly dim if platform style doesn't
        
        self.start_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
    
    def start_conversion(self):
        """Start the batch conversion process."""
        source = self.source_edit.text()
        dest = self.dest_edit.text()
        
        # Validation
        if not source:
            QMessageBox.warning(self, "Error", "Please select a source folder.")
            return
        if not dest:
            QMessageBox.warning(self, "Error", "Please select an output folder.")
            return
        
        source_path = Path(source)
        dest_path = Path(dest)
        
        if not source_path.exists():
            QMessageBox.warning(self, "Error", "Source folder does not exist.")
            return
        
        # Create destination if it doesn't exist
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot create output folder: {e}")
            return
        
        # Check if destination folder is not empty
        if any(dest_path.iterdir()):
            reply = QMessageBox.question(
                self,
                "Output folder not empty",
                f"The output folder is not empty:\n{dest_path}\n\n"
                "All existing files will be DELETED before conversion.\n\n"
                "Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Delete all contents in the destination folder
            try:
                import shutil
                for item in dest_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to clear output folder: {e}")
                return
        
        # Start conversion
        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting conversion...")
        self.last_update_time = 0
        
        # Get selected output format
        format_name = self.format_combo.currentText()
        output_format = OUTPUT_FORMATS.get(format_name, OUTPUT_FORMATS['AVIF'])
        
        self.conversion_worker = ConversionWorker(
            source_dir=source_path,
            dest_dir=dest_path,
            include_subfolders=self.subfolder_check.isChecked(),
            quality=self.quality_slider.value(),
            output_format=output_format,
            allowed_extensions=self.filter_combo.currentData()
        )
        
        self.conversion_worker.progress_updated.connect(self.on_progress_updated)
        self.conversion_worker.conversion_complete.connect(self.on_conversion_complete)
        self.conversion_worker.error_occurred.connect(self.on_error)
        
        self.conversion_worker.start()
    
    def cancel_conversion(self):
        """Cancel the ongoing conversion."""
        if self.conversion_worker:
            self.conversion_worker.cancel()
            self.progress_label.setText("Cancelling...")
    
    def on_progress_updated(self, current: int, total: int, current_file: str):
        """Handle progress updates from the worker thread."""
        current_time = time.time()
        
        # Throttle updates to ~20fps (50ms) to prevent flickering, 
        # but always allow the final update (current == total)
        if current == total or (current_time - self.last_update_time) >= 0.05:
            percentage = int((current / total) * 100)
            
            # Only update value if changed to avoid unnecessary redraws
            if self.progress_bar.value() != percentage:
                self.progress_bar.setValue(percentage)
            
            # Truncate filename if too long to prevent layout thrashing
            display_name = current_file
            if len(display_name) > 40:
                display_name = "..." + display_name[-37:]
            
            self.progress_label.setText(f"{current} of {total} files - {display_name}")
            self.last_update_time = current_time
    
    def on_conversion_complete(self, results: list[ConversionResult]):
        """Handle completion of the conversion process."""
        self.set_ui_enabled(True)
        
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        total_original = sum(r.original_size for r in results if r.success)
        total_converted = sum(r.converted_size for r in results if r.success)
        
        if total_original > 0:
            if total_converted > total_original:
                increase = (total_converted / total_original - 1) * 100
                diff_str = f"{increase:.1f}% larger"
            else:
                reduction = (1 - total_converted / total_original) * 100
                diff_str = f"{reduction:.1f}% reduction"

            size_info = (
                f"\n\nTotal: {SizeEstimator.format_size(total_original)} → "
                f"{SizeEstimator.format_size(total_converted)} "
                f"({diff_str})"
            )
        else:
            size_info = ""
        
        self.progress_bar.setValue(100)
        self.progress_label.setText("Conversion complete!")
        
        message = f"Converted: {successful} files"
        if failed > 0:
            message += f"\nFailed: {failed} files"
        message += size_info
        
        QMessageBox.information(self, "Conversion Complete", message)
    
    def on_error(self, error_message: str):
        """Handle errors from the worker thread."""
        self.set_ui_enabled(True)
        self.progress_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", error_message)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set application style and increase global font size
    app.setStyle('Fusion')
    app.setStyleSheet(DARK_THEME_STYLESHEET) # Apply dark theme
    font = app.font()
    font.setPointSize(font.pointSize() + 1)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
