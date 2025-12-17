
import sys
import os
import base64
import typing
import shutil
from pathlib import Path
from io import BytesIO
from PIL import Image

# GUI Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QTextEdit, QComboBox, QCheckBox,
    QSizePolicy, QFrame, QStyle
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QPen, QFont, QCursor
from typing import Optional

# AI Imports
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Configure Logging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkerThread(QThread):
    finished = pyqtSignal(object) # Returns the image data or error message
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # (stage 0-100, status message)

    def __init__(self, api_key: str, image_path: str, prompt: str, resolution: str):
        super().__init__()
        self.api_key = api_key
        self.image_path = image_path
        self.prompt = prompt
        self.resolution = resolution

    def run(self):
        if not HAS_GENAI:
            self.error.emit("google-genai library is not installed.\nPlease run: pip install google-genai (or ensure google-generativeai >= 0.8.0)")
            return

        try:
            # Stage 1: Preparing
            logger.info("Starting worker thread. Preparing image...")
            self.progress.emit(10, "Preparing image...")
            
            # Initialize Client
            client = genai.Client(api_key=self.api_key)
            
            # Load the image using PIL
            image = Image.open(self.image_path)
            logger.debug(f"Image loaded: {self.image_path} size={image.size} mode={image.mode}")
            
            # Construct prompt for the editing task
            prompt_text = self.prompt
            
            # Stage 2: Uploading/Sending
            logger.info(f"Sending request to Google AI (Model: gemini-3-pro-image-preview, Res: {self.resolution})...")
            self.progress.emit(30, "Sending request to Google AI...")
            
            # Switch to indeterminate mode while waiting
            self.progress.emit(-1, "Waiting for API response... (Processing remotely)")
            
            # Using model: gemini-3-pro-image-preview
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[prompt_text, image],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        image_size=self.resolution
                    )
                )
            )
            
            # Stage 3: Response received
            logger.info("Response received from API.")
            self.progress.emit(80, "Response received, processing...")
            
            # Parse the response to find the image
            found_image = False
            
            if response.parts:
                logger.debug(f"Response contains {len(response.parts)} parts.")
                for i, part in enumerate(response.parts):
                    if part.text:
                        logger.debug(f"Part {i}: Found text content: {part.text[:100]}...")
                    
                    if part.inline_data:
                        logger.debug(f"Part {i}: Found INLINE DATA. MimeType: {part.inline_data.mime_type}")
                        try:
                            # Get raw image bytes directly from inline_data
                            raw_bytes = part.inline_data.data
                            logger.info(f"Part {i}: Got raw image bytes, length={len(raw_bytes)}")
                            
                            # The data might be base64 encoded or raw bytes
                            # Try to decode if it's base64
                            if isinstance(raw_bytes, str):
                                import base64
                                raw_bytes = base64.b64decode(raw_bytes)
                                logger.debug("Decoded base64 string to bytes")
                            
                            # Verify it's valid image data by loading with PIL
                            from PIL import Image as PILImage
                            test_img = PILImage.open(BytesIO(raw_bytes))
                            logger.info(f"Valid image: {test_img.size} {test_img.mode}")
                            
                            # Stage 4: Complete
                            logger.info("Image bytes verified. Emitting finish signal.")
                            self.progress.emit(100, "Complete!")
                            self.finished.emit(raw_bytes)
                            found_image = True
                            break
                        except Exception as e:
                            logger.error(f"Failed to process inline image data: {e}", exc_info=True)
            else:
                logger.warning("Response contained NO parts.")
            
            if not found_image:
                logger.warning("No image found in response parts. Falling back to text.")
                self.progress.emit(100, "Complete (text response)")
                try:
                    text_content = response.text if response.text else "No content returned."
                except Exception as e:
                    text_content = f"Could not extract text: {e}"
                
                logger.info(f"Returning text content: {text_content}")
                self.finished.emit(f"Model returned text instead of image:\n{text_content}")

        except Exception as e:
            logger.error(f"WorkerThread Error: {e}", exc_info=True)
            self.progress.emit(0, "Error")
            self.error.emit(str(e))

class BeforeAfterWidget(QWidget):
    """Widget for side-by-side image comparison with draggable divider and zoom."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap: Optional[QPixmap] = None
        self.converted_pixmap: Optional[QPixmap] = None
        self.input_label = "Original"
        self.output_label = "Colorized"
        self.left_quality = 0
        self.right_quality = 0
        self.is_rendering = False
        self.divider_position = 0.5  # 0.0 to 1.0 (relative to image width)
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
    
    def set_original(self, pixmap):
        self.original_pixmap = pixmap
        self.update()

    def set_colorized(self, pixmap):
        self.converted_pixmap = pixmap
        self.update()

    def reset_view(self):
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
        
        # Rendering overlay (kept for compatibility with main.py style)
        if self.is_rendering:
            painter.fillRect(rect, QColor(30, 30, 30))
            painter.setPen(QColor(200, 200, 200))
            font = QFont()
            font.setPointSize(17)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Processing...")
            return
        
        if not self.original_pixmap:
            # Draw placeholder
            painter.fillRect(rect, QColor(40, 40, 40))
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No Image Selected")
            return
            
        # Draw single image if no result yet
        if not self.converted_pixmap:
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
            
            dest_x = x_offset
            dest_y = y_offset
            
            # Draw full original
            painter.drawPixmap(QRect(dest_x, dest_y, scaled_width, scaled_height), self.original_pixmap)
            
            # Zoom indicator
            if self.zoom_level != 1.0:
                zoom_text = f"Zoom: {self.zoom_level:.1f}x"
                self.draw_pill_label(painter, zoom_text, 15, rect.height() - 45, smaller=True)
            return
        
        # --- Dual View Logic ---
        
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
        
        # --- Handle Different Resolutions/Ratios ---
        # Calculate ratios to map coordinates from Original -> Converted
        conv_w = self.converted_pixmap.width()
        conv_h = self.converted_pixmap.height()
        orig_w = self.original_pixmap.width()
        orig_h = self.original_pixmap.height()
        
        ratio_x = conv_w / orig_w
        ratio_y = conv_h / orig_h
        
        # Map source coordinates to Converted image space
        conv_src_left = int(src_left * ratio_x)
        conv_src_top = int(src_top * ratio_y)
        conv_src_width = int(src_width * ratio_x)
        conv_src_height = int(src_height * ratio_y)
        
        # Safety clamp to valid bounds
        conv_src_width = min(conv_src_width, conv_w - conv_src_left)
        conv_src_height = min(conv_src_height, conv_h - conv_src_top)

        if conv_src_width <= 0 or conv_src_height <= 0:
             visible_converted = QPixmap() # Empty
        else:
             visible_converted = self.converted_pixmap.copy(
                 conv_src_left, conv_src_top, conv_src_width, conv_src_height
             )
        
        # Use fast transformation for large zoom, smooth for small
        transform_mode = (Qt.TransformationMode.FastTransformation 
                         if self.zoom_level > 4.0 
                         else Qt.TransformationMode.SmoothTransformation)
        
        scaled_original = visible_original.scaled(
            dest_width, dest_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            transform_mode
        )
        
        # Force the converted chunk to conform to the exact same screen destination rect.
        # This implicitly handles any slight aspect ratio drift by stretching it to match the Original.
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
        
        # Draw labels pill style
        self.draw_pill_label(painter, self.input_label, 15, 15)
        
        font = QFont()
        font.setBold(True)
        font.setPointSize(15)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        right_label_width = metrics.horizontalAdvance(self.output_label) + 40
        
        self.draw_pill_label(painter, self.output_label, rect.width() - right_label_width, 15)
        
        # Zoom indicator at bottom
        if self.zoom_level != 1.0:
            zoom_text = f"Zoom: {self.zoom_level:.1f}x (scroll=zoom, right-drag=pan, dbl-click=reset)"
            self.draw_pill_label(painter, zoom_text, 15, rect.height() - 45, smaller=True)
    
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
            # Change cursor when near divider - making this wider for better UX
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
            
            if self.dragging:
                self.update_divider_position(event.position().x())
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.dragging = False
        self.panning = False
        self.last_pan_pos = None
        if self.original_pixmap and self.converted_pixmap:
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
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
        if scaled_width > 0:
            self.divider_position = max(0.0, min(1.0, relative_x / scaled_width))
        self.update()


class ColorizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Colorizer")
        self.resize(1200, 850)
        
        # Data
        self.current_image_path = None
        self.api_key = "" # We do not save or persist this.
        self.result_image_bytes = None # Stores bytes of the colorized result
        
        # Batch Data
        self.image_files = []      # List of Path objects
        self.current_index = -1
        self.output_dir = None     # Path object or None
        
        # Preset prompts with descriptions
        self.prompt_presets = {
            "Documentary Realism": {
                "description": "Historical accuracy, museum-quality restoration. Keeps the 'feel' of the original era.",
                "prompt": "A highly realistic colorization and restoration based strictly on the provided black and white photograph taken in Hungary between 1930 and 1980. CRITICAL PRIORITY: The faces, expressions, and unique facial features of all individuals must remain exactly as they are in the original input; do not alter, smooth, or reinterpret the subjects' identities. HOWEVER, DO NOT OVER-SHARPEN THE FACES. The resolution, grain, and sharpness of the facial features must blend perfectly with the background and clothing. Do not create a high-definition 'cutout' effect for the face; it must look like it belongs in the same optical system as the rest of the image. INTEGRITY CHECK: Do not duplicate, triplicate, or hallucinate new details. The output must structurally match the input exactly. If a shirt has 5 buttons, the output must have 5 buttons. If there are 3 chickens, there must be exactly 3 chickens. Do not add random patterns, flora, or fauna that are not present in the source image. If the individual is in a countryside setting then, it usually means absolutely NO makeup. The rest of the image needs extensive cleanup: remove all dust, scratches, water damage, sepia toning, and excessive grain noise. Apply a naturalistic, historically accurate color palette appropriate for Central Europe in the mid-20th century. Clothing should have authentic fabric textures and muted, realistic dyes (wool, cotton patterns). Buildings and background elements should reflect the actual colors of Budapest or the Hungarian countryside during that era. The lighting should be natural daylight, enhancing realistic skin tones without over-saturating."
            },
            "Kodachrome Vibrant": {
                "description": "Rich, warm vintage film look. Great for faded family snapshots and outdoor scenes.",
                "prompt": "A vibrant restoration and colorization of the input black and white image, processed to look like authentic vintage color film stock (similar to Kodachrome used in the 1950s-1970s). MANDATORY CONSTRAINT: The subjects' faces are to be preserved with perfect fidelity to the original photo. Ensure eyes, nose shapes, and mouth expressions are locked and not generated anew. VITAL: The face must NOT appear shaper or higher resolution than the rest of the image. Apply the vintage film grain evenly across the face and background to ensure they are cohesive. The face should not stand out as 'AI generated' or 'super-resolved'. CONTENT FIDELITY: Do not hallucinate or multiply objects. Count integrity is essential: five buttons must remain five buttons; three animals must remain three animals. Do not add imaginary details or textures that do not exist in the source. The image needs to be flawlessly cleaned of all aging artifacts, tears, and discoloration. Bring rich, deep color to the scene. Hungarian streetscapes or landscapes should have warm greens, rich terracotta roofs, and period-appropriate vehicle colors. Clothing should appear fresh and colorful but not neon. The final result should look like a pristine, well-preserved slide from a Hungarian family archive."
            },
            "Focus-Pull Portrait": {
                "description": "Perfect for portraits. Cleans up the subject while gently softening distracting backgrounds.",
                "prompt": "A high-fidelity color portrait restoration based on the provided B&W image. THE MAIN FOCUS is the preservation of the facial features and identity of the subject(s). Do not modify their likeness. Apply realistic, healthy skin tones, accurate hair color based on gray values, and detailed texture to their clothing. IMPORTANT: While the subject is in focus, do not artificially over-sharpen the skin or eyes. The facial texture must remain natural and photographic, blending seamlessly with the lighting of the scene. STRICT ANTI-HALLUCINATION: Do not add random details. If the clothing has a specific number of buttons or patterns, they must be preserved exactly. Do not multiply objects or add elements that are not there. If there are 3 chickens, keep strictly to 3. The background should be cleaned of all damage and colorized with historically appropriate hues for mid-century Hungary, but apply a slight depth-of-field effect, making the background elements slightly softer to de-emphasize cluttered environments, ensuring the sharpest focus is entirely on the restored, colorized people. The lighting should look natural and flattering, as if taken by a professional photographer in the 1940s or 1960s."
            },
            "Custom": {
                "description": "Write your own prompt for full control over the colorization.",
                "prompt": "Colorize this black and white photo naturally."
            }
        }
        
        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 1. Top Bar: API Key
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("Google API Key:"))
        self.api_input = QLineEdit(self.api_key)
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_input.setPlaceholderText("Enter your Gemini/PaLM API Key")
        api_layout.addWidget(self.api_input)
        layout.addLayout(api_layout)
        
        # 1.5 Folder & Navigation Controls
        folder_group = QGroupBox("Batch Processing")
        folder_layout = QVBoxLayout()
        
        # Row 1: Input Folder
        row_input = QHBoxLayout()
        btn_input = QPushButton("Select Input Folder")
        btn_input.clicked.connect(self.select_input_folder)
        self.lbl_input_folder = QLabel("No folder selected")
        self.lbl_input_folder.setStyleSheet("color: #888;")
        row_input.addWidget(btn_input)
        row_input.addWidget(self.lbl_input_folder, stretch=1)
        folder_layout.addLayout(row_input)
        
        # Row 2: Output Folder
        row_output = QHBoxLayout()
        btn_output = QPushButton("Select Output Folder")
        btn_output.clicked.connect(self.select_output_folder)
        self.lbl_output_folder = QLabel("Output: (Same as Input)")
        self.lbl_output_folder.setStyleSheet("color: #888;")
        row_output.addWidget(btn_output)
        row_output.addWidget(self.lbl_output_folder, stretch=1)
        folder_layout.addLayout(row_output)
        
        # Row 3: Navigation
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setFixedWidth(80)
        self.btn_prev.clicked.connect(self.on_prev)
        
        self.combo_files = QComboBox()
        self.combo_files.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_files.currentIndexChanged.connect(self.on_file_selected)
        
        self.btn_next = QPushButton("Next >")
        self.btn_next.setFixedWidth(80)
        self.btn_next.clicked.connect(self.on_next)
        
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.combo_files)
        nav_row.addWidget(self.btn_next)
        folder_layout.addLayout(nav_row)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # 2. Main Display Area (Single Widget)
        # Using our custom BeforeAfterWidget
        self.viewer = BeforeAfterWidget()
        layout.addWidget(self.viewer, stretch=1)
        
        # 3. Prompt Section
        prompt_group = QGroupBox("Prompt settings")
        prompt_layout = QVBoxLayout()
        
        # Preset selector row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(self.prompt_presets.keys()))
        self.combo_preset.currentTextChanged.connect(self.on_preset_changed)
        self.combo_preset.setMinimumWidth(180)
        preset_row.addWidget(self.combo_preset)
        
        self.lbl_preset_desc = QLabel(self.prompt_presets["Documentary Realism"]["description"])
        self.lbl_preset_desc.setStyleSheet("color: #aaa; font-style: italic;")
        preset_row.addWidget(self.lbl_preset_desc, stretch=1)
        preset_row.addStretch()
        prompt_layout.addLayout(preset_row)
        
        # Prompt text area
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter prompt for colorization...")
        self.prompt_input.setText(self.prompt_presets["Documentary Realism"]["prompt"])
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setReadOnly(True)  # Read-only for presets
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # 4. Controls Row
        # 4. Controls Row
        controls_layout = QHBoxLayout()
        
        # --- Group 1: Options (Left) ---
        options_layout = QVBoxLayout()
        self.chk_tech_details = QCheckBox("Add technical details to filename")
        self.chk_tech_details.setToolTip("Appends resolution (e.g. '4K') and preset name to the output filename.")
        self.chk_tech_details.setChecked(True)
        
        self.chk_save_original = QCheckBox("Save copy of original image")
        self.chk_save_original.setChecked(True)
        
        options_layout.addWidget(self.chk_tech_details)
        options_layout.addWidget(self.chk_save_original)
        controls_layout.addLayout(options_layout)
        
        # --- Group 2: Progress & Status (Middle - Filling Blank Space) ---
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #ccc;")
        # Progress Bar definition
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: black;
                background-color: #ddd;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        controls_layout.addLayout(status_layout, stretch=1)

        # --- Group 3: Resolution & Info (Right-Middle) ---
        res_layout = QVBoxLayout()
        
        # Resolution Selector
        self.combo_res = QComboBox()
        self.combo_res.addItems(["1K", "2K", "4K"])
        self.combo_res.setMinimumWidth(80)
        
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        res_row.addWidget(self.combo_res)
        res_layout.addLayout(res_row)
        
        # Info Bubble
        self.frm_res_info = QFrame()
        self.frm_res_info.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 6px;
            }
            QLabel {
                color: #0d47a1;
                font-size: 12px;
                border: none;
                background: transparent;
            }
        """)
        self.frm_res_info.setVisible(True)
        
        info_layout = QHBoxLayout(self.frm_res_info)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(10)
        
        lbl_icon = QLabel("ℹ")
        lbl_icon.setStyleSheet("font-weight: bold; font-size: 16px;")
        info_layout.addWidget(lbl_icon)
        
        self.lbl_res_text = QLabel("Load an image to see recommended resolution.")
        self.lbl_res_text.setWordWrap(True)
        info_layout.addWidget(self.lbl_res_text, stretch=1)
        
        res_layout.addWidget(self.frm_res_info)
        controls_layout.addLayout(res_layout)

        # --- Group 4: Buttons (Right) ---
        # Run Button
        self.btn_run = QPushButton("Colorize (Gemini API)")
        self.btn_run.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
        self.btn_run.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.btn_run.setToolTip(
            "<b>⚠️ Paid Service Warning</b><br>"
            "This action sends the image to Google Gemini API.<br>"
            "Each request may incur usage costs or quota limits."
        )
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_run.clicked.connect(self.run_colorization)
        self.btn_run.setEnabled(False)
        
        # Save Button
        self.btn_save = QPushButton("Save Result")
        self.btn_save.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.btn_save.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        
        controls_layout.addWidget(self.btn_run)
        controls_layout.addWidget(self.btn_save)
        
        layout.addLayout(controls_layout)
        
        # Removed old status layout
        
        # Apply style similar to main.py if possible, but keep it simple
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QLabel { color: #ffffff; }
            QGroupBox { color: #ffffff; font-weight: bold; border: 1px solid #555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLineEdit { padding: 5px; border-radius: 3px; border: 1px solid #555; background-color: #333; color: white; }
            QTextEdit { padding: 5px; border-radius: 3px; border: 1px solid #555; background-color: #333; color: white; }
            QPushButton { padding: 5px 15px; border-radius: 3px; background-color: #444; color: white; border: 1px solid #555; }
            QPushButton:hover { background-color: #555; }
            QComboBox { padding: 5px; border-radius: 3px; border: 1px solid #555; background-color: #333; color: white; }
        """)

    def on_preset_changed(self, preset_name: str):
        """Handle preset selection change."""
        preset = self.prompt_presets.get(preset_name, {})
        self.lbl_preset_desc.setText(preset.get("description", ""))
        self.prompt_input.setText(preset.get("prompt", ""))
        
        # Allow editing only for Custom preset
        if preset_name == "Custom":
            self.prompt_input.setReadOnly(False)
            self.prompt_input.setStyleSheet("background-color: #3a3a3a;")
        else:
            self.prompt_input.setReadOnly(True)
            self.prompt_input.setStyleSheet("")

    # --- Batch / Folder Logic ---

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.load_folder_images(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = Path(folder)
            self.lbl_output_folder.setText(f"Output: {self.output_dir.name}")
            self.lbl_output_folder.setStyleSheet("color: black; font-weight: bold;")
        else:
            self.output_dir = None
            self.lbl_output_folder.setText("Output: (Same as Input)")
            self.lbl_output_folder.setStyleSheet("color: #888;")

    def load_folder_images(self, folder_path):
        p = Path(folder_path)
        # Scan for images
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = sorted([
            f for f in p.iterdir() 
            if f.is_file() and f.suffix.lower() in exts
        ])
        
        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No compatible images found in this folder.")
            return

        self.lbl_input_folder.setText(f"{p.name} ({len(self.image_files)} images)")
        self.lbl_input_folder.setStyleSheet("color: black; font-weight: bold;")
        
        # Populate combo
        self.combo_files.blockSignals(True)
        self.combo_files.clear()
        self.combo_files.addItems([f.name for f in self.image_files])
        self.combo_files.blockSignals(False)
        
        # Load first image
        self.load_image_at_index(0)

    def load_image_at_index(self, index):
        if not self.image_files or index < 0 or index >= len(self.image_files):
            return
        
        self.current_index = index
        # Sync combo without triggering signal loop
        self.combo_files.blockSignals(True)
        self.combo_files.setCurrentIndex(index)
        self.combo_files.blockSignals(False)
        
        # Button states
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < len(self.image_files) - 1)
        
        # Load Image logic (similar to old select_image)
        file_path = str(self.image_files[index])
        self.current_image_path = file_path
        
        # Load into viewer
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.viewer.set_original(pixmap)
            self.viewer.set_colorized(None) # Reset result
            self.viewer.reset_view()
            
            # Smart Resolution Recommendation
            w, h = pixmap.width(), pixmap.height()
            rec_res = self.recommend_resolution(w, h)
            self.combo_res.setCurrentText(rec_res)
            
            self.lbl_res_text.setText(f"Input image is {w}x{h}px.\nRecommended Output: {rec_res}")
            self.frm_res_info.setVisible(True)
            
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)} ({index + 1}/{len(self.image_files)})")
            self.btn_run.setEnabled(True)
            self.btn_save.setEnabled(False) # Reset save button until run
        else:
             QMessageBox.critical(self, "Error", "Failed to load image file.")

    def on_prev(self):
        if self.current_index > 0:
            self.load_image_at_index(self.current_index - 1)

    def on_next(self):
        if self.current_index < len(self.image_files) - 1:
            self.load_image_at_index(self.current_index + 1)

    def on_file_selected(self, index):
        if index >= 0:
            self.load_image_at_index(index)

    def auto_save_result(self):
        """Automatically save the result if output dir is set or just generally."""
        if not self.result_image_bytes or not self.current_image_path:
            return

        # Determine Output Directory
        # If user selected output dir, use it. Else use input dir.
        target_dir = self.output_dir if self.output_dir else Path(self.current_image_path).parent
        
        # Filename Logic (Reused from save_result logic roughly)
        original_stem = Path(self.current_image_path).stem
        
        suffix = ""
        if self.chk_tech_details.isChecked():
            resolution = self.combo_res.currentText()
            preset_name = self.combo_preset.currentText()
            clean_preset = "".join(x for x in preset_name if x.isalnum())
            suffix = f"_{resolution}_{clean_preset}"
            
        filename = f"{original_stem}{suffix}_colorized.png"
        save_path = target_dir / filename
        
        try:
            # Ensure dir exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(self.result_image_bytes)
            
            msg = f"Auto-saved: {filename}"
            self.status_label.setText(msg)
            
            # Optional: If output dir is DIFFERENT, we might also want to save the original copy there?
            # The User said: "image should be auto saved to that output output directory"
            # It's good practice to respect the "Save copy of original" checkbox too.
            if self.chk_save_original.isChecked() and self.current_image_path:
                 orig_copy_name = f"{original_stem}_original{Path(self.current_image_path).suffix}"
                 orig_copy_path = target_dir / orig_copy_name
                 # Only copy if paths differ
                 if Path(self.current_image_path).resolve() != orig_copy_path.resolve():
                     import shutil
                     shutil.copy2(self.current_image_path, orig_copy_path)
                     msg += " (+Original)"
                     self.status_label.setText(msg)

        except Exception as e:
            self.status_label.setText(f"Auto-save failed: {e}")
            print(f"Auto-save error: {e}")

    # --- End Batch Logic ---

    # Keeping select_image just in case someone wants it or for compatibility, 
    # but strictly it's replaced by load_folder_images logic in the UI.
    def select_image(self):
        # Start in last directory or home
        start_dir = ""
        if self.current_image_path:
            start_dir = os.path.dirname(self.current_image_path)
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", start_dir, "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.current_image_path = file_path
            
            # Load into viewer
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.viewer.set_original(pixmap)
                self.viewer.set_colorized(None) # Reset result
                self.viewer.reset_view()
                
                # Smart Resolution Recommendation
                w, h = pixmap.width(), pixmap.height()
                rec_res = self.recommend_resolution(w, h)
                self.combo_res.setCurrentText(rec_res)
                
                self.lbl_res_text.setText(f"Input image is {w}x{h}px.\nRecommended Output: <b>{rec_res}</b>")
                self.frm_res_info.setVisible(True)
                
                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.btn_run.setEnabled(True)
            else:
                 QMessageBox.critical(self, "Error", "Failed to load image file.")

    def recommend_resolution(self, width, height):
        """Recommend output resolution based on input size."""
        max_dim = max(width, height)
        if max_dim <= 1280:
            return "1K"
        elif max_dim <= 2560:
            return "2K"
        else:
            return "4K"

    def run_colorization(self):
        api_key = self.api_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Please enter your Google Gen AI API Key.")
            return

        prompt = self.prompt_input.toPlainText()
        resolution = self.combo_res.currentText()
        
        self.btn_run.setEnabled(False)
        self.status_label.setText("Starting...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        # self.lbl_result.setText("Processing...") # Removed as labeling handled by widget
        
        # Start Thread
        self.worker = WorkerThread(api_key, self.current_image_path, prompt, resolution)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value, message):
        if value < 0:
            self.progress_bar.setRange(0, 0) # Indeterminate (pulsing)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def handle_result(self, result):
        self.btn_run.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Processing complete.")
        
        # In a real scenario, 'result' would be bytes we can load into QPixmap
        # For this POC, it's likely text or a placeholder message 
        # unless handling complex image response parsing.
        
        # In a real scenario, 'result' would be bytes we can load into QPixmap
        # For this POC, it's likely text or a placeholder message 
        # unless handling complex image response parsing.
        
        if isinstance(result, bytes):
            # Store the bytes for saving
            self.result_image_bytes = result
            self.btn_save.setEnabled(True)
            
            # If we successfully got bytes
            qimg = QImage.fromData(result)
            pixmap = QPixmap.fromImage(qimg)
            
            # Update viewer
            self.viewer.set_colorized(pixmap)
            
            # TRIGGGER AUTO-SAVE
            self.auto_save_result()
            
        else:
            # Text response - no image to save
            self.result_image_bytes = None
            self.btn_save.setEnabled(False)
            QMessageBox.warning(self, "Response", str(result))

    def handle_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.progress_bar.hide()
        self.result_image_bytes = None
        self.status_label.setText("Error occurred.")
        QMessageBox.critical(self, "Error", error_msg)
        # self.lbl_result.setText("Error occurred during processing.") # Removed

    def save_result(self):
        if not self.result_image_bytes:
            QMessageBox.warning(self, "No Image", "No colorized image to save.")
            return
        
        # 1. Determine start directory (Same as input)
        start_dir = ""
        original_stem = "colorized_image"
        if self.current_image_path:
            p = Path(self.current_image_path)
            start_dir = str(p.parent)
            original_stem = p.stem
            
        # 2. Determine default filename
        # Check explicit option to append technical details
        suffix = ""
        if self.chk_tech_details.isChecked():
            # Get Resolution
            resolution = self.combo_res.currentText()
            
            # Get Preset Name
            preset_name = self.combo_preset.currentText()
            # Clean text for filename
            clean_preset = "".join(x for x in preset_name if x.isalnum())
            
            suffix = f"_{resolution}_{clean_preset}"
            
        default_name = f"{original_stem}{suffix}_colorized.png"
        
        # 3. Open Dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Colorized Image", os.path.join(start_dir, default_name), "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )
        
        if file_path:
            saved_paths = []
            try:
                # Save Colorized
                with open(file_path, 'wb') as f:
                    f.write(self.result_image_bytes)
                saved_paths.append(os.path.basename(file_path))
                
                # Check option: Save Original Copy
                if self.chk_save_original.isChecked() and self.current_image_path:
                    # Save alongside the new file
                    target_dir = os.path.dirname(file_path)
                    
                    # Construct original copy name
                    orig_copy_name = f"{original_stem}_original{os.path.splitext(self.current_image_path)[1]}"
                    orig_copy_path = os.path.join(target_dir, orig_copy_name)
                    
                    # Copy file logic
                    import shutil
                    shutil.copy2(self.current_image_path, orig_copy_path)
                    saved_paths.append(orig_copy_name)
                
                self.status_label.setText(f"Saved: {', '.join(saved_paths)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")

def main():
    app = QApplication(sys.argv)
    window = ColorizerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
