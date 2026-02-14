"""Unified configuration that merges settings from multiple sources.

Provides a single entry point for obtaining fully resolved plugin settings
by combining defaults, config-file values, project-specific overrides, and
CLI overrides with a clear precedence order.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from manga_translator.config.settings import PluginSettings, SettingsManager
from manga_translator.project_manager import Project, ProjectManager

logger = logging.getLogger(__name__)


class UnifiedConfig:
    """Merges settings from multiple sources with clear precedence.

    Precedence (highest to lowest):
        1. CLI overrides (passed as dict)
        2. Project-specific overrides (from ``project.settings_overrides``)
        3. Config file settings (``~/.config/manga-translator/settings.json``)
        4. Defaults (``PluginSettings()``)

    The class owns a :class:`SettingsManager` internally and optionally loads
    a project via :class:`ProjectManager` to layer in project-level overrides.

    Args:
        cli_overrides: Arbitrary dict of settings that take highest priority.
        project_id: If given, loads this project and applies its
            ``settings_overrides`` on top of the config-file settings.
        projects_dir: Root directory for projects (forwarded to
            ``ProjectManager``).  Defaults to ``~/.manga-translator/projects``.
        config_dir: Config directory for ``SettingsManager``.  Defaults to
            ``~/.config/manga-translator``.
    """

    def __init__(
        self,
        cli_overrides: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        projects_dir: Optional[str] = None,
        config_dir: Optional[Path] = None,
    ) -> None:
        self._cli_overrides = cli_overrides or {}
        self._project_id = project_id
        self._project: Optional[Project] = None

        # Layer 3+4: SettingsManager loads defaults then config file.
        self._settings_manager = SettingsManager(config_dir=config_dir)

        # Layer 2: Project-specific overrides.
        if project_id is not None:
            try:
                pm = ProjectManager(projects_dir=projects_dir)
                self._project = pm.load_project(project_id)
                if self._project.settings_overrides:
                    logger.debug(
                        "Applying project overrides from '%s': %s",
                        project_id,
                        self._project.settings_overrides,
                    )
                    self._settings_manager.update_settings(
                        self._project.settings_overrides
                    )
            except FileNotFoundError:
                logger.warning(
                    "Project '%s' not found; skipping project overrides.",
                    project_id,
                )

        # Layer 1: CLI overrides (highest priority).
        if self._cli_overrides:
            logger.debug("Applying CLI overrides: %s", self._cli_overrides)
            self._settings_manager.update_settings(self._cli_overrides)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> PluginSettings:
        """Return the fully merged :class:`PluginSettings`."""
        return self._settings_manager.get_settings()

    def get_api_key(self, service: str) -> str:
        """Return the API key for *service*, delegating to SettingsManager.

        Environment variables are checked first (handled internally by
        :meth:`SettingsManager.get_api_key`).
        """
        return self._settings_manager.get_api_key(service)

    @property
    def active_project(self) -> Optional[Project]:
        """Return the loaded project, or ``None`` if no project was loaded."""
        return self._project

    @property
    def settings_manager(self) -> SettingsManager:
        """Expose the underlying :class:`SettingsManager` for advanced use."""
        return self._settings_manager

    def validate(self):
        """Validate the merged settings. Returns a list of error strings."""
        return self._settings_manager.validate()

    def describe_sources(self) -> str:
        """Return a human-readable summary of which config layers are active."""
        parts = ["Config file: " + str(self._settings_manager._config_path)]
        if self._project is not None:
            parts.append(f"Project: {self._project.id}")
            if self._project.settings_overrides:
                keys = ", ".join(sorted(self._project.settings_overrides.keys()))
                parts.append(f"  overrides: {keys}")
        if self._cli_overrides:
            keys = ", ".join(sorted(self._cli_overrides.keys()))
            parts.append(f"CLI overrides: {keys}")
        return "\n".join(parts)
