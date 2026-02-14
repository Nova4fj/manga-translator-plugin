# Manga Translator Plugin for GIMP

A comprehensive GIMP plugin for automated translation of manga and comic panels from any language to English. This plugin combines computer vision, OCR, machine translation, and intelligent typesetting to provide a seamless workflow for manga translation within GIMP.

## 🎯 What It Does

The Manga Translator Plugin automates the traditionally manual and time-consuming process of manga translation by:

- **Automatically detecting** speech bubbles and text regions in manga panels
- **Extracting text** using specialized OCR engines optimized for manga/comics
- **Translating** the extracted text using multiple translation backends
- **Removing original text** through intelligent inpainting
- **Typesetting** translated English text back into the bubbles with appropriate fonts and styling

## 👥 Target Users

### Primary Users
- **Scanlation Groups** - Fan translation teams who need to process large volumes of manga chapters efficiently
- **Professional Manga Translators** - Commercial translators working on licensed manga and webtoons
- **Manga Enthusiasts** - Individual fans who want to translate favorite series not available in their language

### Secondary Users
- **Webcomic Creators** - Artists who want to localize their work into multiple languages
- **Digital Archivists** - Preservationists working with historical comic materials
- **Language Learners** - Students using manga as learning material with side-by-side translations

## ✨ Key Features

### 🔍 Intelligent Text Detection
- Advanced speech bubble boundary detection using contour analysis and ML models
- Handles various bubble shapes: traditional ovals, irregular shapes, thought bubbles, rectangular panels
- Detects text both inside and outside bubbles (SFX, narration boxes, signs)

### 📖 Multi-Language OCR
- **Japanese**: Specialized manga-ocr model for mixed hiragana/katakana/kanji text
- **Chinese**: Traditional and simplified character recognition
- **Korean**: Hangul text extraction
- **Universal**: PaddleOCR and Tesseract fallback for any language
- **Vertical text support** for Asian languages

### 🌐 Flexible Translation
- **Cloud APIs**: OpenAI GPT, DeepL, Google Translate
- **Offline Options**: Argos Translate for privacy-sensitive work
- **Context-aware**: Maintains character consistency and manga-specific terminology
- **Customizable dictionaries** for character names and series-specific terms

### 🎨 Advanced Inpainting
- **Neural inpainting**: LaMa model for seamless text removal
- **Traditional methods**: GIMP's built-in healing and clone tools as fallback
- **Smart background reconstruction** that preserves artistic style

### ✍️ Intelligent Typesetting
- **Font matching**: Automatically selects appropriate English fonts for different text types
- **Auto-sizing**: Fits text within bubble boundaries while maintaining readability
- **Style preservation**: Matches original text styling (bold, italic, outlined text)
- **Multi-line text wrapping** with proper alignment

### 🔧 GIMP Integration
- **Native plugin**: Seamlessly integrated into GIMP's menu system
- **Layer management**: Organized output with separate layers for easy editing
- **Undo support**: Full integration with GIMP's undo system
- **Batch processing**: Handle multiple pages or entire chapters

## 🎮 Workflow Modes

### 🤖 Auto Mode
One-click translation for fast processing of simple pages
- Automatic bubble detection → OCR → translation → typesetting
- Perfect for clean, standard manga layouts

### 🎛️ Semi-Auto Mode  
Balanced approach with user review and control
- Plugin detects bubbles → user reviews/adjusts → automated processing
- Ideal for complex layouts or quality-critical work

### ✋ Manual Mode
Full user control for challenging pages
- User manually selects regions → OCR → translation → manual text placement
- Best for artistic panels, unusual layouts, or special effects text

## 🏆 Success Metrics

- **Accuracy**: >95% bubble detection rate on standard manga layouts
- **Speed**: <30 seconds per page for auto mode processing
- **Quality**: Translation quality comparable to manual scanlation workflows
- **Usability**: Reduce translation time by 70-80% compared to manual methods
- **Compatibility**: Support for all major manga formats and GIMP versions

## 📋 Requirements

### GIMP Compatibility
- **GIMP 2.10**: Python-Fu plugin system
- **GIMP 3.0**: libgimp Python bindings (future support)

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **RAM**: 8GB minimum, 16GB recommended for large images
- **GPU**: NVIDIA GPU recommended for neural model acceleration
- **Storage**: 2GB for models and dependencies

### Dependencies
- Python 3.8+ with pip
- PyTorch (for neural models)
- OpenCV (for computer vision)
- Specialized OCR libraries (manga-ocr, PaddleOCR)
- Translation API keys (optional, for cloud services)

## 🚀 Quick Start

1. **Install the plugin** via GIMP's plugin manager or manual installation
2. **Configure translation settings** (API keys, preferred languages)
3. **Open a manga page** in GIMP
4. **Run the plugin** from `Filters → Manga → Translate Page`
5. **Review and adjust** the results as needed
6. **Export** your translated manga

## 📚 Documentation

- [Product Specification](docs/PRODUCT-SPEC.md) - Detailed feature requirements
- [Technical Architecture](docs/TECHNICAL-ARCHITECTURE.md) - System design and components
- [Component Specifications](docs/COMPONENT-SPECS.md) - Detailed component APIs
- [UI Design](docs/UI-DESIGN.md) - Interface mockups and user flows
- [Implementation Plan](docs/IMPLEMENTATION-PLAN.md) - Development roadmap
- [Research & Analysis](docs/RESEARCH.md) - Technology comparisons and decisions

## 🤝 Contributing

This project aims to democratize manga translation while respecting copyright and supporting official releases. We encourage contributions that improve translation quality, expand language support, and enhance the user experience.

## 📄 License

[License details to be determined - likely GPL to match GIMP's licensing]

---

**Note**: This plugin is designed to assist translators and should be used in compliance with copyright laws and manga publisher policies. Always respect official translations and support manga creators.