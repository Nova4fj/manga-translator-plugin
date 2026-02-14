"""CLI entry point for standalone manga translation (outside GIMP)."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Manga Translator — Translate manga pages from the command line"
    )
    parser.add_argument("input", help="Input manga image path")
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
    parser.add_argument("--version", action="version", version="manga-translator 0.1.0")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from manga_translator.config.settings import SettingsManager
    from manga_translator.manga_translator import translate_file

    settings = SettingsManager().get_settings()
    if args.engine:
        settings.translation.primary_engine = args.engine

    print(f"Translating: {args.input}")
    print(f"  {args.source_lang} → {args.target_lang}")

    result = translate_file(
        input_path=args.input,
        output_path=args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        settings=settings,
    )

    print(f"\nResults: {len(result.bubbles)} bubbles translated")
    print(f"Success rate: {result.success_rate:.0%}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")

    return 0 if result.success_rate > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
