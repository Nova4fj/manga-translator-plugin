"""Tests for CLI logging setup (setup_logging helper)."""

import logging
import logging.handlers

from manga_translator.__main__ import setup_logging


def _clear_root_handlers():
    """Remove all handlers from the root logger to isolate tests."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()


def test_setup_logging_without_log_file():
    """setup_logging configures root logger without adding a file handler."""
    _clear_root_handlers()
    try:
        setup_logging(verbose=False, log_file=None)

        root = logging.getLogger()
        assert root.level == logging.INFO
        # No RotatingFileHandler should be present
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 0
    finally:
        _clear_root_handlers()


def test_setup_logging_with_log_file(tmp_path):
    """setup_logging adds a RotatingFileHandler when log_file is provided."""
    _clear_root_handlers()
    log_file = tmp_path / "test.log"
    try:
        setup_logging(verbose=True, log_file=str(log_file))

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

        handler = file_handlers[0]
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 3

        # Verify that logging actually writes to the file
        test_logger = logging.getLogger("test_cli_logging")
        test_logger.info("hello from test")
        handler.flush()

        content = log_file.read_text()
        assert "hello from test" in content
    finally:
        _clear_root_handlers()
