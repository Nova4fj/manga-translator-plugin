# Manga Translator Plugin

Automated manga/comic translation pipeline: bubble detection, OCR, translation, inpainting, and typesetting. Works as a **GIMP plugin** or **standalone CLI**.

## Features

- **Bubble detection** — Automatic speech bubble finding via OpenCV contour analysis
- **OCR** — manga-ocr (Japanese-specialized), PaddleOCR, or Tesseract
- **Translation** — DeepL, OpenAI GPT, or Argos Translate (offline)
- **Inpainting** — Text removal with OpenCV or LaMa neural inpainting
- **Typesetting** — Smart text fitting with font category support (dialogue, narration, SFX)
- **Batch processing** — Multi-threaded parallel page translation
- **Translation memory** — SQLite-backed exact + fuzzy matching for consistency
- **Quality control** — Automated checks for untranslated text, length, terminology
- **Performance monitoring** — Per-stage timing and profiling
- **Export** — PNG, JPEG, PDF, CBZ output formats
- **Project management** — Track multi-page translation projects

## Installation

```bash
# Core only (detection + inpainting)
pip install -e .

# With OCR + translation engines
pip install -e ".[all]"

# Development
pip install -e ".[all,dev]"
```

## Quick Start

### CLI

```bash
# Translate a single page
manga-translator page.png -s ja -t en

# Batch translate
manga-translator batch pages/*.png -o output/

# Use a specific engine
manga-translator page.png --engine argos

# Quality preset
manga-translator page.png --preset quality

# With quality control + performance report
manga-translator page.png --qc --perf

# Dry run (show what would happen)
manga-translator page.png --dry-run

# Check dependencies
manga-translator check
```

### Python API

```python
from manga_translator.manga_translator import MangaTranslationPipeline, translate_file
from manga_translator.config.settings import SettingsManager

# Simple file translation
result = translate_file("page.png", source_lang="ja", target_lang="en")
print(f"{len(result.bubbles)} bubbles, {result.success_rate:.0%} success")

# With pipeline customization
from manga_translator.perf_monitor import PerfMonitor
from manga_translator.quality_control import QualityChecker
from manga_translator.translation_memory import TranslationMemory

settings = SettingsManager().get_settings()
pipeline = MangaTranslationPipeline(
    settings,
    perf_monitor=PerfMonitor(),
    quality_checker=QualityChecker(),
    translation_memory=TranslationMemory(),
)

import cv2
image = cv2.imread("page.png")
result = pipeline.translate_page(image, source_lang="ja", target_lang="en")
```

### GIMP Plugin

1. Copy the plugin to GIMP's plugin directory
2. Restart GIMP
3. Go to **Filters > Manga > Translate Page**

## Configuration

Settings are stored at `~/.config/manga-translator/settings.json`.

```bash
# Quality presets: fast, balanced, quality
manga-translator page.png --preset fast

# Environment variables for API keys
export DEEPL_API_KEY=your-key
export OPENAI_API_KEY=your-key
```

### Unified Config Precedence

1. CLI flags (highest priority)
2. Project-specific overrides
3. Config file (`~/.config/manga-translator/settings.json`)
4. Defaults

### Translation Memory

```python
from manga_translator.translation_memory import TranslationMemory

tm = TranslationMemory()  # ~/.manga-translator/translation_memory.db
tm.add_entry("テスト", "Test", source_lang="ja", target_lang="en")
tm.add_term("ナルト", "Naruto", category="character_name", series="Naruto")

# Export/import
tm.export_json("memory.json")
tm.import_json("memory.json")
```

## CLI Reference

```
manga-translator [OPTIONS] INPUT
manga-translator batch [OPTIONS] FILES...
manga-translator check

Options:
  -o, --output PATH        Output path
  -s, --source-lang LANG   Source language (default: ja)
  -t, --target-lang LANG   Target language (default: en)
  --engine ENGINE          Translation engine: deepl, openai, argos
  --preset PRESET          Quality preset: fast, balanced, quality
  --qc                     Enable quality control checks
  --perf                   Show performance report
  --tm-db PATH             Translation memory database path
  --export-format FMT      Output format: png, jpg, pdf, cbz
  --dry-run                Show plan without executing
  -v, --verbose            Verbose logging
  -q, --quiet              Suppress progress output
  --version                Show version
```

## Architecture

```
manga_translator/
├── __init__.py              # Package init + GIMP auto-registration
├── __main__.py              # CLI entry point
├── manga_translator.py      # Main pipeline orchestrator
├── batch_processor.py       # Multi-threaded batch processing
├── translation_memory.py    # SQLite TM with fuzzy matching
├── quality_control.py       # Automated QC checks
├── perf_monitor.py          # Performance timing
├── export_manager.py        # Multi-format export (PNG/JPEG/PDF/CBZ)
├── project_manager.py       # Project tracking
├── workflow.py              # Semi-auto workflow controller
├── error_recovery.py        # Error handling + retry logic
├── components/
│   ├── bubble_detector.py   # Speech bubble detection
│   ├── ocr_engine.py        # OCR (manga-ocr, PaddleOCR, Tesseract)
│   ├── translator.py        # Translation engines
│   ├── inpainter.py         # Text removal (OpenCV, LaMa)
│   └── typesetter.py        # Text rendering
├── config/
│   ├── settings.py          # Settings management + presets
│   └── unified_config.py    # Multi-source config merging
├── core/
│   ├── image_processor.py   # Image I/O and transforms
│   ├── layer_manager.py     # Layer stack management
│   └── plugin_manager.py    # GIMP plugin registration
└── tests/                   # 380+ tests
```

## Supported Languages

**Source:** Japanese (ja), Chinese (zh), Korean (ko)
**Target:** English (en) and others via translation engine

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=manga_translator

# Check dependencies
manga-translator check
```

## License

GPL-3.0-or-later
