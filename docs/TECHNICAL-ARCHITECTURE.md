# Technical Architecture: Manga Translator Plugin

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           GIMP Environment                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                GIMP Integration Layer                        │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │    │
│  │  │   Python-Fu     │  │   Script-Fu     │  │  libgimp    │  │    │
│  │  │   (GIMP 2.10)   │  │   (Fallback)    │  │ (GIMP 3.0+) │  │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                   │                                  │
│  ┌─────────────────────────────────▼─────────────────────────────┐    │
│  │                    Plugin Core Engine                        │    │
│  │                                                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │    │
│  │  │   Bubble    │  │     OCR     │  │     Translator      │   │    │
│  │  │  Detector   │  │   Engine    │  │      Engine         │   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │    │
│  │         │                 │                    │             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │    │
│  │  │ Inpainter   │  │ Typesetter  │  │   Config Manager    │   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐        ┌────────▼────────┐        ┌───────▼────────┐
│ ML Models      │        │ Translation APIs │        │ Font Resources │
│ • manga-ocr    │        │ • OpenAI GPT     │        │ • Comic Sans   │
│ • PaddleOCR    │        │ • DeepL API      │        │ • CC Wild Words│
│ • LaMa         │        │ • Google Trans   │        │ • Manga Fonts  │
│ • Tesseract    │        │ • Argos Offline  │        │ • Custom Fonts │
└────────────────┘        └─────────────────┘        └────────────────┘
```

## 🧩 Component Breakdown

### 1. Bubble Detector Component

**Purpose**: Automatically identify speech bubbles, thought bubbles, and text regions in manga panels.

**Architecture**:
```
Input Image → Preprocessing → Contour Detection → ML Classification → Bubble Boundaries
     │              │               │                │                    │
     │              │               │                │                    └─→ Output
     │              │               │                └─→ Confidence Score
     │              │               └─→ Shape Analysis
     │              └─→ Edge Enhancement
     └─→ GIMP Layer Data
```

**Technologies**:
- **OpenCV**: Primary computer vision library
- **scikit-image**: Image processing utilities
- **PyTorch/TensorFlow**: Neural network inference (optional ML models)
- **Custom algorithms**: Geometric shape analysis

**Models**:
- **Primary**: Custom CNN trained on manga bubble datasets
- **Fallback**: Traditional computer vision (contour detection + geometric analysis)
- **Training Data**: 50,000+ annotated manga panels from public domain sources

### 2. OCR Engine Component

**Purpose**: Extract text content from detected regions with high accuracy for manga-specific fonts.

**Architecture**:
```
Text Region → Language Detection → Model Selection → Text Extraction → Post-Processing
     │              │                     │               │               │
     │              │                     │               │               └─→ Confidence
     │              │                     │               └─→ Raw Text
     │              │                     └─→ Optimal OCR Model
     │              └─→ Script Analysis (CJK/Latin)
     └─→ Cropped Image Patch
```

**Model Hierarchy**:
1. **manga-ocr** (Primary for Japanese)
   - Specialized for manga fonts and layouts
   - Handles mixed hiragana/katakana/kanji
   - Trained on manga-specific datasets
   - URL: https://github.com/kha-white/manga-ocr

2. **PaddleOCR** (CJK languages)
   - Chinese (Simplified/Traditional)
   - Korean (Hangul)
   - Japanese (backup)
   - URL: https://github.com/PaddlePaddle/PaddleOCR

3. **Tesseract** (Fallback)
   - Latin scripts and other languages
   - Good for English text in translated manga
   - URL: https://github.com/tesseract-ocr/tesseract

### 3. Translator Engine Component

**Purpose**: Translate extracted text with context awareness and manga-specific terminology.

**Architecture**:
```
Source Text → Context Analysis → Engine Selection → Translation Request → Post-Processing
     │              │                    │                   │               │
     │              │                    │                   │               └─→ Final Translation
     │              │                    │                   └─→ Raw Translation
     │              │                    └─→ Best Engine for Context
     │              └─→ Text Type (Dialogue/Narration/SFX)
     └─→ Language Detection
```

**Supported Engines**:

1. **OpenAI GPT-4/GPT-3.5-turbo**
   - Best for context-aware translation
   - Handles character speech patterns
   - Custom prompts for manga terminology
   - API: https://platform.openai.com/docs/api-reference

2. **DeepL API**
   - High quality neural translation
   - Good for formal dialogue
   - Fast response times
   - API: https://www.deepl.com/docs-api

3. **Google Translate API**
   - Wide language support
   - Cost-effective for bulk processing
   - Good baseline quality
   - API: https://cloud.google.com/translate/docs

4. **Argos Translate** (Offline)
   - Privacy-focused local translation
   - No internet required
   - Lower quality but fully offline
   - GitHub: https://github.com/argosopentech/argos-translate

### 4. Inpainter Component

**Purpose**: Remove original text from speech bubbles while preserving background artwork.

**Architecture**:
```
Original Image + Text Mask → Inpainting Algorithm → Quality Check → Final Image
        │            │              │                   │            │
        │            │              │                   │            └─→ Clean Background
        │            │              │                   └─→ Artifact Detection
        │            │              └─→ Background Reconstruction
        │            └─→ Text Region Mask
        └─→ Input Layer
```

**Inpainting Methods**:

1. **LaMa (Large Mask Inpainting)** - Primary
   - State-of-the-art neural inpainting
   - Excellent for complex backgrounds
   - Handles large text regions
   - GitHub: https://github.com/saic-mdal/lama

2. **MAT (Mask-Aware Transformer)** - Alternative
   - Transformer-based architecture
   - Good for artistic backgrounds
   - GitHub: https://github.com/fenglinglwb/MAT

3. **GIMP Built-in Tools** - Fallback
   - Resynthesizer plugin
   - Heal Selection tool
   - Clone tool automation
   - Always available, no external dependencies

### 5. Typesetter Component

**Purpose**: Render English translation text into speech bubbles with appropriate fonts and layout.

**Architecture**:
```
Translation + Bubble Shape → Font Selection → Text Layout → Rendering → Layer Creation
      │            │              │             │           │            │
      │            │              │             │           │            └─→ GIMP Text Layer
      │            │              │             │           └─→ Styled Text
      │            │              │             └─→ Multi-line Layout
      │            │              └─→ Optimal Font & Size
      │            └─→ Available Space
      └─→ Translated Text
```

**Font Management**:
- **Comic Sans MS**: Standard comic font
- **CC Wild Words**: Open source comic lettering
- **Manga Temple**: Specialized manga font
- **Custom Font Detection**: Load user's installed fonts
- **Font Fallback**: Automatic fallback chain for missing fonts

**Layout Algorithms**:
- **Auto-sizing**: Fit text within bubble boundaries
- **Multi-line wrapping**: Intelligent line breaks
- **Alignment**: Center, left, or custom alignment
- **Style preservation**: Bold, italic, outlined text effects

### 6. GIMP Integration Layer

**Purpose**: Bridge between plugin core and GIMP's API systems.

**API Compatibility**:

#### GIMP 2.10 (Python-Fu)
```python
# Registration
from gimpfu import *

def manga_translate(image, drawable, mode="auto"):
    # Plugin implementation
    pass

register(
    "python-fu-manga-translate",
    "Translate manga page",
    "Automatically detect and translate manga text",
    "Author Name",
    "GPL v3",
    "2024",
    "<Image>/Filters/Manga/Translate Page",
    "RGB*, GRAY*",
    [
        (PF_OPTION, "mode", "Translation mode", 0, 
         ["Auto", "Semi-Auto", "Manual"])
    ],
    [],
    manga_translate)

main()
```

#### GIMP 3.0+ (libgimp Python)
```python
# New libgimp API (simplified example)
import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp

class MangaTranslatePlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return ['manga-translate']
    
    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN,
            self.run_translate, None)
        return procedure
```

## 🔄 Data Flow

### Complete Translation Pipeline

```
[User Clicks "Translate Page"]
            │
            ▼
┌─────────────────────────┐
│   1. Input Validation   │ ← Validate GIMP image layer
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  2. Bubble Detection    │ ← OpenCV contour analysis
│     • Preprocessing     │   + optional ML model
│     • Contour finding   │
│     • Shape filtering   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   3. Text Extraction    │ ← manga-ocr / PaddleOCR
│     • Language detect   │   / Tesseract
│     • OCR processing    │
│     • Confidence check  │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│    4. Translation       │ ← OpenAI / DeepL / Google
│     • Context analysis  │   / Argos Translate
│     • Engine selection  │
│     • Translate text    │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│    5. Text Removal      │ ← LaMa neural inpainting
│     • Mask generation   │   / GIMP healing tools
│     • Inpainting        │
│     • Quality check     │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│    6. Typesetting       │ ← Font selection + layout
│     • Font selection    │   algorithms
│     • Text layout       │
│     • Style application │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   7. Layer Creation     │ ← GIMP layer management
│     • Create new layer  │
│     • Apply text        │
│     • Organize layers   │
└─────────────────────────┘
```

### Error Handling Flow

```
Processing Step → Error Detection → Error Classification → Recovery Strategy
       │               │                   │                    │
       │               │                   │                    └─→ Continue/Retry/Abort
       │               │                   └─→ User/System/Data Error
       │               └─→ Exception Catching + Logging
       └─→ Any Component

Error Types:
• OCR Confidence < Threshold → Manual review mode
• Translation API Timeout → Fallback to different service  
• Inpainting Artifacts → Fall back to GIMP tools
• Font Loading Failed → Use system fallback fonts
• GPU Memory Error → Switch to CPU processing
```

## 🔌 GIMP Plugin API Details

### Python-Fu Integration (GIMP 2.10)

**Plugin Structure**:
```
manga-translator/
├── __init__.py          # Plugin entry point
├── manga_translator.py  # Main plugin logic
├── components/          # Core components
│   ├── bubble_detector.py
│   ├── ocr_engine.py
│   ├── translator.py
│   ├── inpainter.py
│   └── typesetter.py
├── models/             # ML model files
├── fonts/              # Font resources
├── ui/                 # User interface
│   ├── dialogs.py
│   └── progress.py
└── config/             # Configuration
    └── settings.py
```

**Registration Process**:
1. Plugin discovered in GIMP's plugin directories
2. Python-Fu loads and executes registration
3. Menu entries created in GIMP interface
4. Plugin ready for user activation

### Libgimp Integration (GIMP 3.0+)

**Key Differences**:
- Object-oriented plugin architecture
- Improved Python bindings
- Better integration with GIMP's core
- Enhanced security model

**Migration Strategy**:
- Maintain Python-Fu version for GIMP 2.10 compatibility
- Develop libgimp version for GIMP 3.0+
- Shared core logic between versions
- Runtime detection of GIMP version

## 📦 Dependency Management

### Python Package Dependencies

**Core Requirements**:
```
# Computer Vision
opencv-python>=4.8.0
scikit-image>=0.19.0
pillow>=9.0.0

# Machine Learning
torch>=1.13.0
torchvision>=0.14.0
transformers>=4.20.0

# OCR Libraries
manga-ocr>=0.1.6
paddleocr>=2.6.0
pytesseract>=0.3.10

# Translation APIs
openai>=0.28.0
deepl>=1.12.0
googletrans>=4.0.0
argostranslate>=1.8.0

# GIMP Integration
numpy>=1.21.0
```

**Bundling Strategy**:
1. **Local Installation**: pip install in user's Python environment
2. **Portable Bundle**: Package dependencies with plugin
3. **Conda Environment**: Pre-configured environment for easy setup
4. **Docker Container**: Complete isolated environment

### Model File Management

**Storage Locations**:
- **User Directory**: `~/.gimp-X.Y/manga-translator/models/`
- **System Directory**: `/usr/share/gimp/X.Y/manga-translator/models/`
- **Plugin Directory**: Relative to plugin installation

**Download Strategy**:
- Models downloaded on first use
- Progress indication during download
- Fallback to smaller models if disk space limited
- Manual model management interface

## ⚡ Performance Considerations

### GPU Acceleration

**CUDA Support**:
- Automatic GPU detection and utilization
- Fallback to CPU if GPU unavailable
- Memory management for large images
- Batch processing optimization

**Model Optimization**:
- TensorRT optimization for NVIDIA GPUs
- ONNX Runtime for cross-platform acceleration
- Model quantization for memory efficiency
- Dynamic batching for multiple bubbles

### Memory Management

**Large Image Handling**:
```
Image Size Strategy:
• <10MB: Process entire image at once
• 10-50MB: Tile-based processing with overlap
• >50MB: Reduce resolution for detection, full-res for final output
```

**Memory Profiling**:
- Monitor memory usage during processing
- Release model memory between operations
- Implement memory pressure callbacks
- Warn users of memory limitations

### Batch Processing

**Multi-Page Optimization**:
- Reuse loaded models across pages
- Parallel processing of independent steps
- Progress tracking across batch
- Error isolation (one failed page doesn't stop batch)

**Threading Strategy**:
- UI thread for GIMP integration
- Worker threads for CPU-intensive tasks
- GPU queue management for model inference
- Inter-thread communication for progress updates

## 🔒 Security Considerations

### API Key Management

**Storage Security**:
- Encrypted storage of API keys
- OS keychain integration where available
- No plaintext keys in configuration files
- Secure transmission to translation services

### Privacy Protection

**Data Handling**:
- Optional offline-only mode (no cloud APIs)
- Temporary file cleanup
- No image data sent to services unless explicitly allowed
- User consent for data transmission

### Input Validation

**Image Security**:
- Validate image formats and sizes
- Protect against malformed image exploits
- Sanitize file paths and names
- Limit resource consumption

## 🧪 Testing Strategy

### Automated Testing

**Unit Tests**:
- Component isolation testing
- Mock external services (OCR, translation APIs)
- Edge case coverage (empty bubbles, overlapping text)
- Performance regression testing

**Integration Tests**:
- End-to-end translation pipeline
- GIMP API compatibility testing
- Cross-platform compatibility
- Different image formats and sizes

### Manual Testing

**User Acceptance Testing**:
- Representative manga pages from different genres
- Various image qualities and resolutions
- Different user workflows and preferences
- Accessibility testing

**Performance Testing**:
- Load testing with large batches
- Memory leak detection
- GPU/CPU performance profiling
- Network connectivity edge cases