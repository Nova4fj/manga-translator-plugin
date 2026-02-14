# Changelog

All notable changes to Manga Translator Plugin.

## [Unreleased]

### Added
- GitHub Actions CI workflow with test matrix (Python 3.10, 3.11, 3.12), lint, and coverage jobs
- Makefile with `test`, `lint`, `coverage`, `ci`, and `clean` targets for local development
- Coverage threshold enforcement at 80% via `pyproject.toml`

### Fixed
- Resolved 94 ruff lint warnings (unused imports, ambiguous variable names, unused assignments)

## [0.5.0] - 2026-02-14

### Added
- Cross-page translation context: dialogue history, character names, and glossary terms carry across pages for consistent multi-page translations
- `CrossPageContext` accumulator with configurable page window and dialogue line limits
- Heuristic character name detection from translated text (Title Case patterns)
- Name consistency checking across pages
- `--cross-page-context` CLI flag for batch translation (forces sequential mode)
- 18 new tests (553 total)

## [0.4.0] - 2026-02-14

### Added
- Bubble shape classifier: classifies bubbles as speech, thought, shout, narration, or caption using contour analysis
- SFX/onomatopoeia detector: finds sound effects drawn outside speech bubbles
- Font matcher: intelligent font selection based on bubble type (speech→comic, shout→gothic, etc.)
- Panel-aware reading order optimizer: clusters bubbles into panels, sorts in RTL (manga) or LTR (manhwa) order
- 44 new tests (521 total)

## [0.3.0] - 2026-02-14

### Added
- Public API exports from package root (`from manga_translator import translate_file`)
- Input validation at all entry points (CLI, API, GIMP plugin)
- Text region filter: pre-OCR screening skips non-text bubble regions
- Translation context: surrounding dialogue enriches translation prompts
- GIMP plugin quality presets (Fast / Balanced / Quality)
- GIMP plugin error recovery and progress labels
- CHANGELOG.md

### Changed
- Modernized build backend to `setuptools.build_meta`
- Added project URLs to pyproject.toml

## [0.2.0] - 2026-02-14

### Added
- Pipeline integration: PerfMonitor, QualityChecker, TranslationMemory wired into main pipeline
- Unified config system merging CLI, config file, and project overrides
- CLI flags: `--qc`, `--perf`, `--tm-db`, `--dry-run`, `--quiet`, `--preset`, `--export-format`
- End-to-end integration tests (16 tests with synthetic images)
- Packaging: entry points, dependency groups, classifiers

### Changed
- README rewritten with full CLI reference and Python API docs

## [0.1.0] - 2026-02-14

### Added
- Bubble detection with contour analysis and ellipse fitting
- OCR: manga-ocr, PaddleOCR, Tesseract with engine fallback
- Translation: OpenAI, DeepL, Argos with automatic failover
- Inpainting: OpenCV (Telea/NS) + LaMa neural inpainting
- Typesetting: auto font sizing, word wrap, outline, vertical text
- Batch processing with parallel workers
- Translation memory (SQLite-backed, exact + fuzzy matching)
- Export manager (PNG, JPG, PDF, CBZ)
- Quality control with length ratio and untranslated checks
- Performance monitoring with per-stage timing
- Semi-auto workflow for manual review
- Settings presets (fast / balanced / quality)
- Error recovery with retry and fallback strategies
- GIMP Python-Fu plugin integration
- CLI mode: `python -m manga_translator <image>`
- 441 tests
