"""Project management for multi-chapter manga translation.

Organises pages into projects with persistent state, tracks translation
progress per page, and manages project-level settings and metadata.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

_PROJECTS_DIR = Path.home() / ".manga-translator" / "projects"
_PROJECT_FILE = "project.json"


@dataclass
class PageInfo:
    """Tracking info for a single page within a project."""
    filename: str
    status: str = "pending"  # pending, in_progress, translated, reviewed, exported
    source_path: str = ""
    output_path: str = ""
    bubble_count: int = 0
    translated_count: int = 0
    last_modified: float = 0.0
    notes: str = ""

    @property
    def progress(self) -> float:
        if self.bubble_count == 0:
            return 1.0 if self.status in ("translated", "reviewed", "exported") else 0.0
        return self.translated_count / self.bubble_count


@dataclass
class ProjectMetadata:
    """Project-level metadata."""
    name: str = ""
    series: str = ""
    chapter: str = ""
    source_lang: str = "ja"
    target_lang: str = "en"
    created_at: float = 0.0
    updated_at: float = 0.0
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Project:
    """A manga translation project."""
    id: str = ""
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    pages: List[PageInfo] = field(default_factory=list)
    settings_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def progress(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.progress for p in self.pages) / len(self.pages)

    @property
    def completed_count(self) -> int:
        return sum(1 for p in self.pages if p.status in ("translated", "reviewed", "exported"))

    def summary(self) -> str:
        name = self.metadata.name or self.id
        return (
            f"Project: {name} | "
            f"{self.completed_count}/{self.page_count} pages | "
            f"{self.progress:.0%} complete"
        )


class ProjectManager:
    """Manages manga translation projects on disk.

    Each project is stored as a directory under the projects root,
    containing a ``project.json`` file with metadata and page tracking.

    Args:
        projects_dir: Root directory for all projects. Defaults to
            ``~/.manga-translator/projects``.
    """

    def __init__(self, projects_dir: Optional[str] = None):
        self._root = Path(projects_dir) if projects_dir else _PROJECTS_DIR
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_project(
        self,
        project_id: str,
        name: str = "",
        series: str = "",
        chapter: str = "",
        source_lang: str = "ja",
        target_lang: str = "en",
    ) -> Project:
        """Create a new project directory and metadata file."""
        project_dir = self._root / project_id
        if project_dir.exists():
            raise ValueError(f"Project '{project_id}' already exists")

        project_dir.mkdir(parents=True)
        now = time.time()

        project = Project(
            id=project_id,
            metadata=ProjectMetadata(
                name=name or project_id,
                series=series,
                chapter=chapter,
                source_lang=source_lang,
                target_lang=target_lang,
                created_at=now,
                updated_at=now,
            ),
        )
        self._save(project)
        logger.info("Created project: %s", project_id)
        return project

    def load_project(self, project_id: str) -> Project:
        """Load a project from disk."""
        path = self._root / project_id / _PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._from_dict(data)

    def save_project(self, project: Project) -> None:
        """Persist project state to disk."""
        project.metadata.updated_at = time.time()
        self._save(project)

    def delete_project(self, project_id: str) -> None:
        """Delete a project directory."""
        project_dir = self._root / project_id
        if not project_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found")
        shutil.rmtree(project_dir)
        logger.info("Deleted project: %s", project_id)

    def list_projects(self) -> List[Project]:
        """List all projects."""
        projects = []
        for entry in sorted(self._root.iterdir()):
            project_file = entry / _PROJECT_FILE
            if entry.is_dir() and project_file.exists():
                try:
                    projects.append(self.load_project(entry.name))
                except Exception as e:
                    logger.warning("Could not load project %s: %s", entry.name, e)
        return projects

    # ------------------------------------------------------------------
    # Page management
    # ------------------------------------------------------------------

    def add_pages(self, project_id: str, image_paths: List[str]) -> List[PageInfo]:
        """Add pages to a project from image file paths."""
        project = self.load_project(project_id)
        existing = {p.filename for p in project.pages}
        added = []

        for path in image_paths:
            filename = os.path.basename(path)
            if filename in existing:
                logger.debug("Skipping duplicate: %s", filename)
                continue

            page = PageInfo(
                filename=filename,
                source_path=str(Path(path).resolve()),
                last_modified=time.time(),
            )
            project.pages.append(page)
            existing.add(filename)
            added.append(page)

        # Sort pages by filename for consistent ordering
        project.pages.sort(key=lambda p: p.filename)
        self.save_project(project)
        return added

    def update_page(
        self,
        project_id: str,
        filename: str,
        status: Optional[str] = None,
        output_path: Optional[str] = None,
        bubble_count: Optional[int] = None,
        translated_count: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> PageInfo:
        """Update a page's tracking info."""
        project = self.load_project(project_id)
        page = self._find_page(project, filename)

        if status is not None:
            page.status = status
        if output_path is not None:
            page.output_path = output_path
        if bubble_count is not None:
            page.bubble_count = bubble_count
        if translated_count is not None:
            page.translated_count = translated_count
        if notes is not None:
            page.notes = notes

        page.last_modified = time.time()
        self.save_project(project)
        return page

    def remove_page(self, project_id: str, filename: str) -> None:
        """Remove a page from the project."""
        project = self.load_project(project_id)
        project.pages = [p for p in project.pages if p.filename != filename]
        self.save_project(project)

    def get_page(self, project_id: str, filename: str) -> PageInfo:
        """Get a single page's info."""
        project = self.load_project(project_id)
        return self._find_page(project, filename)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _save(self, project: Project) -> None:
        project_dir = self._root / project.id
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / _PROJECT_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(project), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _to_dict(project: Project) -> dict:
        return {
            "id": project.id,
            "metadata": asdict(project.metadata),
            "pages": [asdict(p) for p in project.pages],
            "settings_overrides": project.settings_overrides,
        }

    @staticmethod
    def _from_dict(data: dict) -> Project:
        meta = data.get("metadata", {})
        return Project(
            id=data["id"],
            metadata=ProjectMetadata(**meta),
            pages=[PageInfo(**p) for p in data.get("pages", [])],
            settings_overrides=data.get("settings_overrides", {}),
        )

    @staticmethod
    def _find_page(project: Project, filename: str) -> PageInfo:
        for page in project.pages:
            if page.filename == filename:
                return page
        raise KeyError(f"Page '{filename}' not found in project '{project.id}'")
