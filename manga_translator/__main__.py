"""CLI entry point for standalone manga translation (outside GIMP)."""

import argparse
import logging
import logging.handlers
import sys


def setup_logging(verbose, log_file=None):
    """Configure logging with optional file output.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
        log_file: Optional path to a log file. Uses RotatingFileHandler
                  with max 5MB size and 3 backup files.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt)

    if log_file:
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)


def check_dependencies(args=None):
    """Check and report status of all optional dependencies.

    Args:
        args: Parsed argparse namespace. Supports ``--validate-keys`` flag
              to test whether API keys are configured.
    """
    import os
    import shutil

    deps = {
        "opencv-python": ("cv2", "Core image processing"),
        "numpy": ("numpy", "Core array operations"),
        "Pillow": ("PIL", "Font rendering and typesetting"),
        "pytesseract": ("pytesseract", "OCR (Tesseract engine)"),
        "manga-ocr": ("manga_ocr", "OCR (manga-specialized, Japanese)"),
        "openai": ("openai", "Translation (OpenAI GPT)"),
        "deepl": ("deepl", "Translation (DeepL)"),
        "argostranslate": ("argostranslate", "Translation (offline, Argos)"),
    }

    print("Manga Translator — Dependency Check")
    print("=" * 50)

    available = 0
    missing = 0
    for name, (module, desc) in deps.items():
        try:
            __import__(module)
            print(f"  [OK]     {name:<20s} {desc}")
            available += 1
        except ImportError:
            print(f"  [MISSING] {name:<20s} {desc}")
            missing += 1

    # Check tesseract binary
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print(f"  [OK]     {'tesseract-ocr':<20s} Tesseract binary")
    except Exception:
        print(f"  [MISSING] {'tesseract-ocr':<20s} Tesseract binary (apt install tesseract-ocr)")

    print(f"\n{available} available, {missing} missing")
    if missing > 0:
        print("\nInstall missing packages with:")
        print("  pip install <package-name>")

    # GPU Status
    print("\nGPU Status")
    print("-" * 50)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK]     CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  [INFO]   CUDA: Not available (CPU mode)")
    except ImportError:
        print("  [INFO]   PyTorch not installed (CPU mode)")

    # API Key Validation (only with --validate-keys)
    if getattr(args, 'validate_keys', False):
        print("\nAPI Key Validation")
        print("-" * 50)
        # Check OPENAI_API_KEY
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            print(f"  [OK]     OPENAI_API_KEY: Set ({len(key)} chars)")
        else:
            print("  [MISSING] OPENAI_API_KEY: Not set")
        # Check DEEPL_AUTH_KEY
        key = os.environ.get("DEEPL_AUTH_KEY", "")
        if key:
            print(f"  [OK]     DEEPL_AUTH_KEY: Set ({len(key)} chars)")
        else:
            print("  [MISSING] DEEPL_AUTH_KEY: Not set")

    # Cache Directory Check
    print("\nCache Directory")
    print("-" * 50)
    cache_dir = os.path.expanduser("~/.manga-translator/models")
    if os.path.isdir(cache_dir):
        size = shutil.disk_usage(cache_dir).used
        print(f"  [OK]     {cache_dir} ({size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  [INFO]   {cache_dir} (not created yet)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Manga Translator — Translate manga pages from the command line"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default translate command (positional arg)
    parser.add_argument("input", nargs="?", help="Input manga image path")
    parser.add_argument("-o", "--output", help="Output path (default: <input>_translated.<ext>)")
    parser.add_argument("-s", "--source-lang", default="ja", help="Source language (default: ja)")
    parser.add_argument("-t", "--target-lang", default="en", help="Target language (default: en)")
    parser.add_argument(
        "--engine",
        choices=["deepl", "openai", "argos"],
        default=None,
        help="Translation engine override",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--log-file", default=None, help="Write logs to file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--qc", action="store_true", help="Enable quality control checks")
    parser.add_argument("--perf", action="store_true", help="Show performance report")
    parser.add_argument("--tm-db", default=None, help="Translation memory database path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument(
        "--exclude-region",
        default=None,
        help="Exclude regions from bubble detection (x,y,w,h;x,y,w,h)",
    )
    parser.add_argument(
        "--export-format",
        choices=["png", "jpg", "pdf", "cbz"],
        default=None,
        help="Output export format",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default=None,
        help="Translation quality preset",
    )
    parser.add_argument(
        "--reading-direction",
        choices=["rtl", "ltr"],
        default="rtl",
        help="Reading order: rtl (manga) or ltr (manhwa) (default: rtl)",
    )
    parser.add_argument("--detect-sfx", action="store_true", help="Enable SFX/onomatopoeia detection")
    parser.add_argument("--fonts-dir", default=None, help="Custom fonts directory for typesetting")
    parser.add_argument("--version", action="version", version="manga-translator 0.5.0")

    # Subcommands
    check_parser = subparsers.add_parser("check", help="Check dependency status")
    check_parser.add_argument("--validate-keys", action="store_true", help="Test API key validity")

    batch_parser = subparsers.add_parser("batch", help="Batch translate multiple files")
    batch_parser.add_argument("files", nargs="+", help="Input image files")
    batch_parser.add_argument("-o", "--output-dir", help="Output directory (default: same as input)")
    batch_parser.add_argument("-s", "--source-lang", default="ja")
    batch_parser.add_argument("-t", "--target-lang", default="en")
    batch_parser.add_argument("--engine", choices=["deepl", "openai", "argos"], default=None)
    batch_parser.add_argument("-v", "--verbose", action="store_true")
    batch_parser.add_argument("--log-file", default=None, help="Write logs to file")
    batch_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    batch_parser.add_argument("--qc", action="store_true", help="Enable quality control checks")
    batch_parser.add_argument("--perf", action="store_true", help="Show performance report")
    batch_parser.add_argument("--tm-db", default=None, help="Translation memory database path")
    batch_parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    batch_parser.add_argument(
        "--export-format",
        choices=["png", "jpg", "pdf", "cbz"],
        default=None,
        help="Output export format",
    )
    batch_parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default=None,
        help="Translation quality preset",
    )
    batch_parser.add_argument(
        "--assemble",
        default=None,
        choices=["cbz", "pdf"],
        help="Assemble all translated pages into a single CBZ or PDF",
    )
    batch_parser.add_argument(
        "--cross-page-context",
        action="store_true",
        help="Enable cross-page context for consistent character names and terminology (sequential mode)",
    )

    args = parser.parse_args()

    if args.command == "check":
        return check_dependencies(args)

    if args.command == "batch":
        return batch_translate(args)

    if not args.input:
        parser.print_help()
        return 1

    from manga_translator.input_validator import InputValidator, ValidationError
    try:
        InputValidator.validate_image_path(args.input)
        InputValidator.validate_language(args.source_lang, "source_lang")
        InputValidator.validate_language(args.target_lang, "target_lang")
        if args.output:
            InputValidator.validate_output_path(args.output)
    except ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    setup_logging(args.verbose, args.log_file)

    from manga_translator.config.settings import SettingsManager
    from manga_translator.manga_translator import translate_file

    sm = SettingsManager()
    if args.preset:
        sm.apply_preset(args.preset)
    settings = sm.get_settings()
    if args.engine:
        settings.translation.primary_engine = args.engine

    output = args.output
    if args.export_format:
        import os
        if output:
            base, _ = os.path.splitext(output)
            output = f"{base}.{args.export_format}"
        else:
            base, _ = os.path.splitext(args.input)
            output = f"{base}_translated.{args.export_format}"

    if not args.quiet:
        print(f"Translating: {args.input}")
        print(f"  {args.source_lang} → {args.target_lang}")
        if args.preset:
            print(f"  Preset: {args.preset}")

    if args.dry_run:
        print(f"[dry-run] Would translate {args.input}")
        print(f"  Output: {output or '<auto>'}")
        print(f"  Engine: {settings.translation.primary_engine}")
        if args.tm_db:
            print(f"  TM database: {args.tm_db}")
        if args.qc:
            print("  Quality control: enabled")
        return 0

    result = translate_file(
        input_path=args.input,
        output_path=output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        settings=settings,
        tm_db_path=args.tm_db,
        enable_qc=args.qc,
        enable_perf=args.perf,
        exclude_regions=args.exclude_region,
        reading_direction=args.reading_direction,
        detect_sfx=args.detect_sfx,
        fonts_dir=args.fonts_dir,
    )

    if not args.quiet:
        print(f"\nResults: {len(result.bubbles)} bubbles translated")
        print(f"Success rate: {result.success_rate:.0%}")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for err in result.errors:
                print(f"  - {err}")

    if args.qc and result.qc_report:
        print(f"\n{result.qc_report.summary()}")

    if args.perf and result.perf_summary:
        print(f"\n{result.perf_summary}")

    return 0 if result.success_rate > 0 else 1


def batch_translate(args):
    """Batch translate multiple manga pages."""
    import os

    setup_logging(args.verbose, args.log_file)

    from manga_translator.config.settings import SettingsManager
    from manga_translator.manga_translator import translate_file

    sm = SettingsManager()
    if args.preset:
        sm.apply_preset(args.preset)
    settings = sm.get_settings()
    if args.engine:
        settings.translation.primary_engine = args.engine

    from manga_translator.input_validator import InputValidator, ValidationError
    try:
        InputValidator.validate_language(args.source_lang, "source_lang")
        InputValidator.validate_language(args.target_lang, "target_lang")
    except ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    valid_files, validation_errors = InputValidator.validate_batch_paths(args.files)
    for err in validation_errors:
        print(f"Warning: {err}", file=sys.stderr)

    if not valid_files:
        print("Error: No valid input files", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    total = len(valid_files)
    successes = 0
    failures = 0
    output_paths = []

    if args.dry_run:
        print(f"[dry-run] Would batch translate {total} files")
        for f in valid_files:
            print(f"  {f}")
        print(f"  Engine: {settings.translation.primary_engine}")
        return 0

    # Cross-page context for consistent multi-page translations
    cross_page_ctx = None
    if getattr(args, 'cross_page_context', False):
        from manga_translator.cross_page_context import CrossPageContext
        cross_page_ctx = CrossPageContext()

    if not args.quiet:
        print(f"Batch translating {total} files ({args.source_lang} → {args.target_lang})")
        if cross_page_ctx:
            print("  Cross-page context: enabled (sequential mode)")
        print("=" * 50)

    for i, input_path in enumerate(valid_files, 1):
        if not os.path.exists(input_path):
            print(f"  [{i}/{total}] SKIP {input_path} (not found)")
            failures += 1
            continue

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.basename(input_path)
            name, ext = os.path.splitext(base)
            out_ext = f".{args.export_format}" if args.export_format else ext
            output_path = os.path.join(output_dir, f"{name}_translated{out_ext}")
        else:
            output_path = None

        try:
            if not args.quiet:
                print(f"  [{i}/{total}] {input_path}...", end=" ", flush=True)
            result = translate_file(
                input_path=input_path,
                output_path=output_path,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                settings=settings,
                tm_db_path=args.tm_db,
                enable_qc=args.qc,
                enable_perf=args.perf,
                cross_page_context=cross_page_ctx,
            )
            status = f"{len(result.bubbles)} bubbles, {result.success_rate:.0%} success"
            if not args.quiet:
                print(status)
                if args.qc and result.qc_report:
                    print(f"    {result.qc_report.summary()}")
                if args.perf and result.perf_summary:
                    print(f"    {result.perf_summary}")
            if result.success_rate > 0:
                successes += 1
                if output_path:
                    output_paths.append(output_path)
            else:
                failures += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failures += 1

    if args.assemble and successes > 0 and output_paths:
        import cv2
        from manga_translator.export_manager import ExportManager
        em = ExportManager()
        images = []
        for p in output_paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
        if images:
            assemble_path = os.path.join(output_dir or ".", f"translated.{args.assemble}")
            em.export(images, assemble_path, format=args.assemble)
            if not args.quiet:
                print(f"Assembled {len(images)} pages into {assemble_path}")

    if not args.quiet:
        print("=" * 50)
        print(f"Done: {successes} succeeded, {failures} failed out of {total}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
