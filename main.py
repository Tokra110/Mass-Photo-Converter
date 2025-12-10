"""
PNG to AVIF Batch Converter
A high-performance batch conversion tool for PNG to AVIF format.
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QSlider, QProgressBar,
    QFileDialog, QMessageBox, QFrame, QGroupBox, QComboBox, QSizePolicy
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
    }
    if HEIC_AVAILABLE:
        formats['HEIC'] = OutputFormat('HEIC', '.heic', 'HEIF', default_quality=80)
    if JXL_AVAILABLE:
        formats['JPEG XL'] = OutputFormat('JPEG XL', '.jxl', 'JXL', default_quality=80)
    return formats


OUTPUT_FORMATS = _build_output_formats()
INPUT_EXTENSIONS = {'.png', '.tiff', '.tif', '.jpg', '.jpeg'}


@dataclass
class ConversionResult:
    """Result of a single file conversion."""
    source_path: Path
    dest_path: Path
    original_size: int
    converted_size: int
    success: bool
    error: Optional[str] = None


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
        parent=None
    ):
        super().__init__(parent)
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.include_subfolders = include_subfolders
        self.quality = quality
        self.output_format = output_format
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
                    if Path(file).suffix.lower() in INPUT_EXTENSIONS:
                        image_files.append(Path(root) / file)
        else:
            for file in self.source_dir.iterdir():
                if file.is_file() and file.suffix.lower() in INPUT_EXTENSIONS:
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
                # Convert to RGB/RGBA as needed
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    # Keep alpha channel for transparent images
                    img = img.convert('RGBA')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
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
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
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
            font.setPointSize(16)
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
        font.setPointSize(14)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        right_label_width = metrics.horizontalAdvance(right_text) + 40  # padding
        
        self.draw_pill_label(painter, left_text, 15, 15)
        self.draw_pill_label(painter, right_text, rect.width() - right_label_width, 15)
        
        # Zoom indicator at bottom
        if self.zoom_level != 1.0:
            zoom_text = f"Zoom: {self.zoom_level:.1f}x (scroll=zoom, right-drag=pan, dbl-click=reset)"
            self.draw_pill_label(painter, zoom_text, 15, rect.height() - 45, smaller=True)
    
    def draw_pill_label(self, painter: QPainter, text: str, x: int, y: int, smaller: bool = False):
        """Draw a label with pill-shaped background."""
        font = QFont()
        font.setBold(True)
        font.setPointSize(11 if smaller else 14)
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
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Image Converter")
        self.setMinimumSize(900, 700)
        self.resize(1280, 800)  # Start at HD-ish size
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Image Batch Converter")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Folder selection group
        folder_group = QGroupBox("Folders")
        folder_layout = QVBoxLayout(folder_group)
        
        # Source folder
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select folder containing images (PNG, TIFF, JPEG)...")
        self.source_edit.textChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.source_edit)
        self.source_btn = QPushButton("Browse...")
        self.source_btn.clicked.connect(self.browse_source)
        source_layout.addWidget(self.source_btn)
        folder_layout.addLayout(source_layout)
        
        # Destination folder
        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel("Output:"))
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Select output folder for converted files...")
        dest_layout.addWidget(self.dest_edit)
        self.dest_btn = QPushButton("Browse...")
        self.dest_btn.clicked.connect(self.browse_dest)
        dest_layout.addWidget(self.dest_btn)
        folder_layout.addLayout(dest_layout)
        
        # Include subfolders checkbox
        self.subfolder_check = QCheckBox("Include subfolders (recreate folder structure)")
        self.subfolder_check.setChecked(True)
        self.subfolder_check.stateChanged.connect(self.on_source_changed)
        folder_layout.addWidget(self.subfolder_check)
        
        main_layout.addWidget(folder_group)
        
        # Preview group
        preview_group = QGroupBox("Preview comparison")
        preview_layout = QVBoxLayout(preview_group)
        
        # Controls row: Left side (file + compare), Right side (quality slider)
        controls_layout = QHBoxLayout()
        
        # Left column: File selector and Compare dropdowns
        left_controls = QVBoxLayout()
        
        # File selector row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self.preview_combo = QComboBox()
        self.preview_combo.setMaximumWidth(180)
        file_row.addWidget(self.preview_combo)
        self.load_preview_btn = QPushButton("Load")
        self.load_preview_btn.setMaximumWidth(50)
        self.load_preview_btn.clicked.connect(self.on_load_preview_clicked)
        file_row.addWidget(self.load_preview_btn)
        file_row.addStretch()
        left_controls.addLayout(file_row)
        
        # Compare selectors row
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Compare:"))
        self.left_format_combo = QComboBox()
        self.left_format_combo.addItem("Original")
        for name in OUTPUT_FORMATS.keys():
            self.left_format_combo.addItem(name)
        self.left_format_combo.setCurrentText("Original")
        self.left_format_combo.currentTextChanged.connect(self.on_comparison_format_changed)
        compare_row.addWidget(self.left_format_combo)
        compare_row.addWidget(QLabel("vs"))
        self.right_format_combo = QComboBox()
        self.right_format_combo.addItem("Original")
        for name in OUTPUT_FORMATS.keys():
            self.right_format_combo.addItem(name)
        self.right_format_combo.setCurrentText("AVIF")
        self.right_format_combo.currentTextChanged.connect(self.on_comparison_format_changed)
        compare_row.addWidget(self.right_format_combo)
        compare_row.addStretch()
        left_controls.addLayout(compare_row)
        
        controls_layout.addLayout(left_controls)
        controls_layout.addSpacing(30)
        
        # Right column: Thick fill-bar quality slider
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setMinimum(5)  # Min 5%
        self.quality_slider.setMaximum(100)
        self.quality_slider.setSingleStep(5)  # 5% increments
        self.quality_slider.setPageStep(5)
        self.quality_slider.setValue(80)
        self.quality_slider.setMinimumWidth(180)
        self.quality_slider.setMinimumHeight(40)  # Thick slider
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
        # Only update label while dragging, render on release
        self.quality_slider.valueChanged.connect(self.on_quality_value_changed)
        self.quality_slider.sliderReleased.connect(self.on_quality_slider_released)
        quality_layout.addWidget(self.quality_slider)
        self.quality_label = QLabel("80%")
        self.quality_label.setMinimumWidth(40)
        quality_layout.addWidget(self.quality_label)
        
        controls_layout.addLayout(quality_layout)
        preview_layout.addLayout(controls_layout)
        
        # Comparison widget
        self.comparison_widget = ComparisonWidget()
        preview_layout.addWidget(self.comparison_widget, 1)
        
        # Size info label (will show both formats)
        self.size_info_label = QLabel("")
        self.size_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.size_info_label)
        
        main_layout.addWidget(preview_group, 1)
        
        # Output settings group (for actual conversion)
        output_group = QGroupBox("Conversion output")
        output_layout = QHBoxLayout(output_group)
        
        # Format selector
        output_layout.addWidget(QLabel("Convert to:"))
        self.format_combo = QComboBox()
        for name in OUTPUT_FORMATS.keys():
            self.format_combo.addItem(name)
        self.format_combo.setCurrentText('AVIF')  # Default
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        output_layout.addWidget(self.format_combo)
        
        output_layout.addSpacing(20)
        
        # Estimate label
        self.estimate_label = QLabel("Select a source folder to see estimate")
        output_layout.addWidget(self.estimate_label)
        
        output_layout.addStretch()
        
        main_layout.addWidget(output_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        main_layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.setMinimumWidth(150)
        self.start_btn.setMinimumHeight(35)
        self.start_btn.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumWidth(100)
        self.cancel_btn.setMinimumHeight(35)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
    
    def browse_source(self):
        """Open dialog to select source folder."""
        # Start from current value, or last used folder, or home
        start_dir = self.source_edit.text() or self.dest_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", start_dir
        )
        if folder:
            self.source_edit.setText(folder)
    
    def browse_dest(self):
        """Open dialog to select destination folder."""
        # Start from current value, or source folder, or home
        start_dir = self.dest_edit.text() or self.source_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir
        )
        if folder:
            self.dest_edit.setText(folder)
    
    def on_source_changed(self):
        """Handle source folder change - update file list and preview."""
        source_path = self.source_edit.text()
        
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
        
        self.image_files = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if Path(file).suffix.lower() in INPUT_EXTENSIONS:
                        self.image_files.append(Path(root) / file)
        else:
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in INPUT_EXTENSIONS:
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
        self.current_pil_image.save(output_buffer, **save_params)
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
                left_reduction = (1 - left_size / original_size) * 100
                left_info = f"{left_label}: {left_str} ({left_reduction:.0f}% savings)"
            
            if right_format == "Original":
                right_info = f"{right_label}: {right_str}"
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
                        if Path(file).suffix.lower() in INPUT_EXTENSIONS:
                            file_path = Path(root) / file
                            count += 1
                            total_size += file_path.stat().st_size
            else:
                for file in folder.iterdir():
                    if file.is_file() and file.suffix.lower() in INPUT_EXTENSIONS:
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
        
        # Use actual ratio from preview if available, otherwise use estimate
        if hasattr(self, 'actual_compression_ratio') and self.actual_compression_ratio > 0:
            ratio = self.actual_compression_ratio
            estimated_size = int(total_size * ratio)
            reduction_pct = (1 - ratio) * 100
            estimate_type = "based on preview"
        else:
            ratio = SizeEstimator.estimate_ratio(quality, format_name)
            estimated_size = int(total_size * ratio)
            reduction_pct = (1 - ratio) * 100
            estimate_type = "estimated"
        
        original_str = SizeEstimator.format_size(total_size)
        estimated_str = SizeEstimator.format_size(estimated_size)
        
        self.estimate_label.setText(
            f"📁 All {file_count} files ({estimate_type}): {original_str} → ~{estimated_str} (~{reduction_pct:.0f}% savings)"
        )
    
    def set_ui_enabled(self, enabled: bool):
        """Enable or disable UI elements during conversion."""
        self.source_edit.setEnabled(enabled)
        self.dest_edit.setEnabled(enabled)
        self.source_btn.setEnabled(enabled)
        self.dest_btn.setEnabled(enabled)
        self.subfolder_check.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.left_format_combo.setEnabled(enabled)
        self.right_format_combo.setEnabled(enabled)
        self.quality_slider.setEnabled(enabled)
        self.preview_combo.setEnabled(enabled)
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
        
        # Get selected output format
        format_name = self.format_combo.currentText()
        output_format = OUTPUT_FORMATS.get(format_name, OUTPUT_FORMATS['AVIF'])
        
        self.conversion_worker = ConversionWorker(
            source_dir=source_path,
            dest_dir=dest_path,
            include_subfolders=self.subfolder_check.isChecked(),
            quality=self.quality_slider.value(),
            output_format=output_format
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
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"{current} of {total} files - {current_file}")
    
    def on_conversion_complete(self, results: list[ConversionResult]):
        """Handle completion of the conversion process."""
        self.set_ui_enabled(True)
        
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        total_original = sum(r.original_size for r in results if r.success)
        total_converted = sum(r.converted_size for r in results if r.success)
        
        if total_original > 0:
            reduction = (1 - total_converted / total_original) * 100
            size_info = (
                f"\n\nTotal: {SizeEstimator.format_size(total_original)} → "
                f"{SizeEstimator.format_size(total_converted)} "
                f"({reduction:.1f}% reduction)"
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
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
