"""Tests for the enhanced check_dependencies command."""

import argparse
from unittest import mock

from manga_translator.__main__ import check_dependencies


class TestCheckDependenciesBasic:
    """Tests for check_dependencies without --validate-keys."""

    def test_returns_zero(self):
        """check_dependencies should return 0 (success) regardless of which deps are installed."""
        args = argparse.Namespace(validate_keys=False)
        result = check_dependencies(args)
        assert result == 0

    def test_runs_without_args(self):
        """check_dependencies should work when called with args=None (backward compat)."""
        result = check_dependencies()
        assert result == 0

    def test_gpu_section_printed(self, capsys):
        """Output should contain the GPU Status section."""
        check_dependencies()
        captured = capsys.readouterr()
        assert "GPU Status" in captured.out

    def test_cache_section_printed(self, capsys):
        """Output should contain the Cache Directory section."""
        check_dependencies()
        captured = capsys.readouterr()
        assert "Cache Directory" in captured.out

    def test_no_api_section_without_flag(self, capsys):
        """API Key Validation section should NOT appear without --validate-keys."""
        args = argparse.Namespace(validate_keys=False)
        check_dependencies(args)
        captured = capsys.readouterr()
        assert "API Key Validation" not in captured.out


class TestCheckDependenciesValidateKeys:
    """Tests for check_dependencies with --validate-keys."""

    def test_api_section_printed(self, capsys):
        """API Key Validation section should appear with --validate-keys."""
        args = argparse.Namespace(validate_keys=True)
        check_dependencies(args)
        captured = capsys.readouterr()
        assert "API Key Validation" in captured.out

    def test_keys_missing_when_unset(self, capsys):
        """Both keys should show MISSING when env vars are not set."""
        args = argparse.Namespace(validate_keys=True)
        with mock.patch.dict("os.environ", {}, clear=True):
            check_dependencies(args)
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY: Not set" in captured.out
        assert "DEEPL_AUTH_KEY: Not set" in captured.out

    def test_keys_ok_when_set(self, capsys):
        """Both keys should show OK when env vars are set."""
        args = argparse.Namespace(validate_keys=True)
        openai_key = "sk-test12345"
        deepl_key = "dl-key-abc"
        env = {"OPENAI_API_KEY": openai_key, "DEEPL_AUTH_KEY": deepl_key}
        with mock.patch.dict("os.environ", env, clear=True):
            check_dependencies(args)
        captured = capsys.readouterr()
        assert f"OPENAI_API_KEY: Set ({len(openai_key)} chars)" in captured.out
        assert f"DEEPL_AUTH_KEY: Set ({len(deepl_key)} chars)" in captured.out
