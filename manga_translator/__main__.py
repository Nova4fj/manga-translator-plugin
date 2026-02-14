"""CLI entry point for standalone manga translation (outside GIMP)."""

import argparse
import logging
import sys


def check_dependencies():
    """Check and report status of all optional dependencies."""
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
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--qc", action="store_true", help="Enable quality control checks")
    parser.add_argument("--perf", action="store_true", help="Show performance report")
    parser.add_argument("--tm-db", default=None, help="Translation memory database path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
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
    parser.add_argument("--version", action="version", version="manga-translator 0.2.0")

    # Subcommands
    subparsers.add_parser("check", help="Check dependency status")

    batch_parser = subparsers.add_parser("batch", help="Batch translate multiple files")
    batch_parser.add_argument("files", nargs="+", help="Input image files")
    batch_parser.add_argument("-o", "--output-dir", help="Output directory (default: same as input)")
    batch_parser.add_argument("-s", "--source-lang", default="ja")
    batch_parser.add_argument("-t", "--target-lang", default="en")
    batch_parser.add_argument("--engine", choices=["deepl", "openai", "argos"], default=None)
    batch_parser.add_argument("-v", "--verbose", action="store_true")
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

    args = parser.parse_args()

    if args.command == "check":
        return check_dependencies()

    if args.command == "batch":
        return batch_translate(args)

    if not args.input:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from manga_translator.config.settings import SettingsManager
    from manga_translator.manga_translator import translate_file

    sm = SettingsManager()
    if args.preset:
        sm.apply_preset(args.preset)
    settings = sm.get_settings()
    if args.engine:
        settings.translation.primary_engine = args.engine

    output = args.output
    if args.export_format and output:
        import os
        base, _ = os.path.splitext(output)
        output = f"{base}.{args.export_format}"

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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from manga_translator.config.settings import SettingsManager
    from manga_translator.manga_translator import translate_file

    sm = SettingsManager()
    if args.preset:
        sm.apply_preset(args.preset)
    settings = sm.get_settings()
    if args.engine:
        settings.translation.primary_engine = args.engine

    output_dir = args.output_dir
    total = len(args.files)
    successes = 0
    failures = 0

    if args.dry_run:
        print(f"[dry-run] Would batch translate {total} files")
        for f in args.files:
            print(f"  {f}")
        print(f"  Engine: {settings.translation.primary_engine}")
        return 0

    if not args.quiet:
        print(f"Batch translating {total} files ({args.source_lang} → {args.target_lang})")
        print("=" * 50)

    for i, input_path in enumerate(args.files, 1):
        if not os.path.exists(input_path):
            print(f"  [{i}/{total}] SKIP {input_path} (not found)")
            failures += 1
            continue

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.basename(input_path)
            name, ext = os.path.splitext(base)
            output_path = os.path.join(output_dir, f"{name}_translated{ext}")
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
            else:
                failures += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failures += 1

    if not args.quiet:
        print("=" * 50)
        print(f"Done: {successes} succeeded, {failures} failed out of {total}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
