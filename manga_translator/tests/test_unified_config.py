"""Tests for the UnifiedConfig module."""

import json
import pytest

from manga_translator.config.unified_config import UnifiedConfig
from manga_translator.config.settings import PluginSettings
from manga_translator.project_manager import ProjectManager


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Provide an empty temporary config directory."""
    return tmp_path / "config"


@pytest.fixture
def tmp_projects_dir(tmp_path):
    """Provide an empty temporary projects directory."""
    return tmp_path / "projects"


@pytest.fixture
def config_file(tmp_config_dir):
    """Write a settings.json with some non-default values."""
    tmp_config_dir.mkdir(parents=True)
    data = {
        "workflow_mode": "manual",
        "translation": {"source_language": "zh"},
        "ocr": {"confidence_threshold": 0.9},
    }
    path = tmp_config_dir / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def project_with_overrides(tmp_projects_dir):
    """Create a project that has settings_overrides."""
    pm = ProjectManager(projects_dir=str(tmp_projects_dir))
    project = pm.create_project("test-proj", name="Test Project")
    project.settings_overrides = {
        "translation": {"source_language": "ko", "primary_engine": "openai"},
        "log_level": "DEBUG",
    }
    pm.save_project(project)
    return project


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDefaultSettings:
    """Verify UnifiedConfig with no overrides yields defaults."""

    def test_defaults_match_plugin_settings(self, tmp_config_dir):
        uc = UnifiedConfig(config_dir=tmp_config_dir)
        settings = uc.get_settings()
        defaults = PluginSettings()
        assert settings.workflow_mode == defaults.workflow_mode
        assert settings.translation.source_language == defaults.translation.source_language
        assert settings.ocr.confidence_threshold == defaults.ocr.confidence_threshold

    def test_no_active_project_by_default(self, tmp_config_dir):
        uc = UnifiedConfig(config_dir=tmp_config_dir)
        assert uc.active_project is None


class TestConfigFileSettings:
    """Verify settings loaded from the config file."""

    def test_config_file_values_applied(self, tmp_config_dir, config_file):
        uc = UnifiedConfig(config_dir=tmp_config_dir)
        settings = uc.get_settings()
        assert settings.workflow_mode == "manual"
        assert settings.translation.source_language == "zh"
        assert settings.ocr.confidence_threshold == 0.9

    def test_config_file_preserves_other_defaults(self, tmp_config_dir, config_file):
        uc = UnifiedConfig(config_dir=tmp_config_dir)
        settings = uc.get_settings()
        # These were not in the config file, so they should remain default.
        assert settings.translation.primary_engine == "deepl"
        assert settings.log_level == "INFO"


class TestCLIOverrides:
    """Verify CLI overrides take highest precedence."""

    def test_cli_overrides_flat_key(self, tmp_config_dir, config_file):
        uc = UnifiedConfig(
            cli_overrides={"workflow_mode": "auto"},
            config_dir=tmp_config_dir,
        )
        # Config file says "manual", CLI says "auto" -> CLI wins.
        assert uc.get_settings().workflow_mode == "auto"

    def test_cli_overrides_nested_key(self, tmp_config_dir, config_file):
        uc = UnifiedConfig(
            cli_overrides={"translation": {"source_language": "ja"}},
            config_dir=tmp_config_dir,
        )
        # Config file says "zh", CLI says "ja" -> CLI wins.
        assert uc.get_settings().translation.source_language == "ja"

    def test_cli_overrides_without_config_file(self, tmp_config_dir):
        uc = UnifiedConfig(
            cli_overrides={"log_level": "DEBUG"},
            config_dir=tmp_config_dir,
        )
        assert uc.get_settings().log_level == "DEBUG"


class TestProjectOverrides:
    """Verify project-specific overrides apply correctly."""

    def test_project_overrides_applied(
        self, tmp_config_dir, tmp_projects_dir, project_with_overrides
    ):
        uc = UnifiedConfig(
            project_id="test-proj",
            projects_dir=str(tmp_projects_dir),
            config_dir=tmp_config_dir,
        )
        settings = uc.get_settings()
        assert settings.translation.source_language == "ko"
        assert settings.translation.primary_engine == "openai"
        assert settings.log_level == "DEBUG"

    def test_active_project_is_set(
        self, tmp_config_dir, tmp_projects_dir, project_with_overrides
    ):
        uc = UnifiedConfig(
            project_id="test-proj",
            projects_dir=str(tmp_projects_dir),
            config_dir=tmp_config_dir,
        )
        assert uc.active_project is not None
        assert uc.active_project.id == "test-proj"

    def test_missing_project_is_graceful(self, tmp_config_dir, tmp_projects_dir):
        uc = UnifiedConfig(
            project_id="nonexistent",
            projects_dir=str(tmp_projects_dir),
            config_dir=tmp_config_dir,
        )
        assert uc.active_project is None
        # Settings should still be valid defaults.
        assert uc.get_settings().workflow_mode == "auto"


class TestPrecedence:
    """Verify the full three-layer precedence chain."""

    def test_cli_beats_project_beats_config(
        self, tmp_config_dir, tmp_projects_dir, config_file, project_with_overrides
    ):
        """All three layers set translation.source_language differently:
        config=zh, project=ko, CLI=ja.  CLI must win."""
        uc = UnifiedConfig(
            cli_overrides={"translation": {"source_language": "ja"}},
            project_id="test-proj",
            projects_dir=str(tmp_projects_dir),
            config_dir=tmp_config_dir,
        )
        settings = uc.get_settings()
        # CLI override wins.
        assert settings.translation.source_language == "ja"
        # Project override still applies where CLI doesn't touch it.
        assert settings.translation.primary_engine == "openai"
        # Config file value survives where neither project nor CLI overrides it.
        assert settings.ocr.confidence_threshold == 0.9

    def test_project_beats_config(
        self, tmp_config_dir, tmp_projects_dir, config_file, project_with_overrides
    ):
        """Without CLI overrides, project should beat config."""
        uc = UnifiedConfig(
            project_id="test-proj",
            projects_dir=str(tmp_projects_dir),
            config_dir=tmp_config_dir,
        )
        settings = uc.get_settings()
        # Project says "ko", config says "zh" -> project wins.
        assert settings.translation.source_language == "ko"
        # Config file value still present for keys project doesn't override.
        assert settings.workflow_mode == "manual"


class TestAPIKeyDelegation:
    """Verify get_api_key delegates to SettingsManager."""

    def test_api_key_from_settings(self, tmp_config_dir):
        config_dir = tmp_config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        data = {"translation": {"deepl_api_key": "test-key-123"}}
        (config_dir / "settings.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        uc = UnifiedConfig(config_dir=config_dir)
        assert uc.get_api_key("deepl") == "test-key-123"

    def test_api_key_env_var_takes_precedence(self, tmp_config_dir, monkeypatch):
        config_dir = tmp_config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        data = {"translation": {"deepl_api_key": "file-key"}}
        (config_dir / "settings.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        monkeypatch.setenv("DEEPL_API_KEY", "env-key")
        uc = UnifiedConfig(config_dir=config_dir)
        assert uc.get_api_key("deepl") == "env-key"

    def test_unknown_service_returns_empty(self, tmp_config_dir):
        uc = UnifiedConfig(config_dir=tmp_config_dir)
        assert uc.get_api_key("unknown_service") == ""


class TestDescribeSources:
    """Verify the describe_sources helper."""

    def test_describe_with_all_layers(
        self, tmp_config_dir, tmp_projects_dir, project_with_overrides
    ):
        uc = UnifiedConfig(
            cli_overrides={"log_level": "WARNING"},
            project_id="test-proj",
            projects_dir=str(tmp_projects_dir),
            config_dir=tmp_config_dir,
        )
        desc = uc.describe_sources()
        assert "Config file:" in desc
        assert "Project: test-proj" in desc
        assert "CLI overrides:" in desc
