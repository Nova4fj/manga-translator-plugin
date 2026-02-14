"""Tests for settings management."""

import json
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

    def test_validate_lama_method(self):
        """lama and auto should be valid inpainting methods."""
        manager = SettingsManager()
        settings = manager.get_settings()
        settings.inpainting.method = "lama"
        issues = manager.validate()
        inpaint_errors = [e for e in issues if "inpainting method" in e.lower()]
        assert len(inpaint_errors) == 0

    def test_validate_auto_method(self):
        manager = SettingsManager()
        settings = manager.get_settings()
        settings.inpainting.method = "auto"
        issues = manager.validate()
        inpaint_errors = [e for e in issues if "inpainting method" in e.lower()]
        assert len(inpaint_errors) == 0


class TestPresets:
    def test_apply_fast_preset(self):
        manager = SettingsManager()
        manager.apply_preset("fast")
        settings = manager.get_settings()
        assert settings.inpainting.method == "blur"
        assert settings.inpainting.use_neural is False

    def test_apply_balanced_preset(self):
        manager = SettingsManager()
        manager.apply_preset("balanced")
        settings = manager.get_settings()
        assert settings.inpainting.method == "auto"
        assert settings.inpainting.use_neural is True

    def test_apply_quality_preset(self):
        manager = SettingsManager()
        manager.apply_preset("quality")
        settings = manager.get_settings()
        assert settings.inpainting.method == "lama"
        assert settings.ocr.primary_engine == "manga-ocr"

    def test_unknown_preset_raises(self):
        manager = SettingsManager()
        with pytest.raises(ValueError, match="Unknown preset"):
            manager.apply_preset("nonexistent")

    def test_preset_preserves_other_settings(self):
        manager = SettingsManager()
        settings = manager.get_settings()
        settings.translation.source_language = "zh"
        manager.apply_preset("fast")
        # Unaffected settings should be preserved
        assert manager.get_settings().translation.source_language == "zh"


class TestImportExport:
    def test_export_profile(self, tmp_path):
        manager = SettingsManager()
        settings = manager.get_settings()
        settings.translation.primary_engine = "openai"

        path = tmp_path / "profile.json"
        result = manager.export_profile(str(path))
        assert path.exists()
        assert result == path

        with open(path) as f:
            data = json.load(f)
        assert data["translation"]["primary_engine"] == "openai"

    def test_import_profile(self, tmp_path):
        # Create a profile file
        profile = {
            "translation": {"primary_engine": "argos"},
            "inpainting": {"method": "lama"},
        }
        path = tmp_path / "profile.json"
        with open(path, "w") as f:
            json.dump(profile, f)

        manager = SettingsManager()
        manager.import_profile(str(path))
        settings = manager.get_settings()
        assert settings.translation.primary_engine == "argos"
        assert settings.inpainting.method == "lama"

    def test_import_partial_preserves_defaults(self, tmp_path):
        profile = {"translation": {"source_language": "ko"}}
        path = tmp_path / "partial.json"
        with open(path, "w") as f:
            json.dump(profile, f)

        manager = SettingsManager()
        manager.import_profile(str(path))
        settings = manager.get_settings()
        assert settings.translation.source_language == "ko"
        # Other settings unchanged
        assert settings.translation.primary_engine == "deepl"

    def test_import_nonexistent_raises(self):
        manager = SettingsManager()
        with pytest.raises(FileNotFoundError):
            manager.import_profile("/nonexistent/path.json")

    def test_export_import_roundtrip(self, tmp_path):
        manager = SettingsManager()
        settings = manager.get_settings()
        settings.typesetting.alignment = "left"
        settings.typesetting.font_category = "sfx"
        settings.inpainting.quality_threshold = 0.8

        path = tmp_path / "roundtrip.json"
        manager.export_profile(str(path))

        manager2 = SettingsManager()
        manager2.import_profile(str(path))
        s2 = manager2.get_settings()
        assert s2.typesetting.alignment == "left"
        assert s2.typesetting.font_category == "sfx"
        assert s2.inpainting.quality_threshold == 0.8
