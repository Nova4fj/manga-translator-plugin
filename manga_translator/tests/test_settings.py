"""Tests for settings management."""

import json
import tempfile
import pytest

from manga_translator.config.settings import (
    PluginSettings,
    SettingsManager,
    TranslationSettings,
    OCRSettings,
)


class TestPluginSettings:
    def test_default_settings(self):
        settings = PluginSettings()
        assert settings.translation.primary_engine == "deepl"
        assert settings.ocr.primary_engine == "manga-ocr"
        assert settings.workflow_mode == "auto"

    def test_nested_dataclass(self):
        settings = PluginSettings()
        assert isinstance(settings.translation, TranslationSettings)
        assert isinstance(settings.ocr, OCRSettings)


class TestSettingsManager:
    def test_init(self):
        manager = SettingsManager()
        assert manager is not None

    def test_get_settings(self):
        manager = SettingsManager()
        settings = manager.get_settings()
        assert isinstance(settings, PluginSettings)

    def test_save_and_load(self, tmp_path):
        manager = SettingsManager(config_dir=str(tmp_path))
        settings = manager.get_settings()
        settings.translation.primary_engine = "openai"
        manager.save()

        # Load fresh
        manager2 = SettingsManager(config_dir=str(tmp_path))
        settings2 = manager2.get_settings()
        assert settings2.translation.primary_engine == "openai"

    def test_reset_to_defaults(self, tmp_path):
        manager = SettingsManager(config_dir=str(tmp_path))
        settings = manager.get_settings()
        settings.translation.primary_engine = "openai"
        manager.save()

        manager.reset_to_defaults()
        settings = manager.get_settings()
        assert settings.translation.primary_engine == "deepl"

    def test_validate(self):
        manager = SettingsManager()
        issues = manager.validate()
        assert isinstance(issues, list)
