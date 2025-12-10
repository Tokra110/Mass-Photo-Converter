# Mass Photo Converter

**Mass Photo Converter** is a high-performance, multi-threaded batch image conversion tool designed to optimize images for modern web standards. It specializes in converting high-quality PNG, TIFF, and JPEG images into next-generation formats like AVIF and WebP, offering significant size reductions with minimal visual loss.

## Key Features

- **🚀 High-Performance Batch Processing**: Utilizes all available CPU cores to convert thousands of images in parallel, ensuring maximum efficiency.
- **👀 Smart Interactive Preview**: Compare "Original" vs "Converted" images side-by-side in real-time.
  - **Split-View**: Draggable divider to inspect changes pixel-by-pixel.
  - **Zoom & Pan**: Deep zoom support for checking fine details.
- **📊 Quality vs. Size Analysis**: A dedicated analysis tool that renders your image at multiple quality levels (Low/Medium/High details) to plot a curve of File Size vs. Quality. This helps you find the perfect "sweet spot" for compression without guessing.
- **📂 Deep Folder Scanning**: Recursively scans source directories and replicates the entire folder structure in the output destination.
- **🎨 Modern Output Formats**:
  - **AVIF**: Excellent compression, ideal for web use.
  - **WebP**: Widely supported, effective transparency.
  - **HEIC**: High effective compression (requires platform support).
  - **JPEG XL**: Next-gen JPEG update (requires plugin).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MassPhotoConverter.git
   cd MassPhotoConverter
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `PyQt6` for the GUI and `Pillow` along with necessary plugins for AVIF/HEIC support.*

## Usage

1. **Launch the Application**:
   ```bash
   python main.py
   ```

2. **Select Folders**:
   - Click **Browse** next to "Source" to select the folder containing your original images.
   - Click **Browse** next to "Output" to select where converted files should be saved.

3. **Preview & Configure**:
   - The dropdown list will populate with images found in your source folder.
   - Click **Load** to open an image in the preview window.
   - Adjust the **Quality Slider** (5% - 100%) to see real-time changes in the "Converted" pane.
   - Use the **Compare** dropdowns to check different formats against each other (e.g., AVIF vs WebP).

4. **Analyze (Optional)**:
   - Click **📊 Start Analysis** to generate a chart showing how different quality settings affect file size for your specific image. This is great for making data-driven decisions on compression settings.

5. **Convert**:
   - Select your target format (e.g., AVIF) from the "Conversion output" section.
   - Click **Start Conversion**. A progress bar will track the batch process.

## Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **Libraries**:
  - `PyQt6`
  - `Pillow`
  - `pillow-heif`
  - `pillow-jxl-plugin`

## License

[MIT License](LICENSE)
