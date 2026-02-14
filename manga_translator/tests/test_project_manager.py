"""Tests for project management system."""

import json
import os
import pytest

from manga_translator.project_manager import (
    PageInfo,
    ProjectMetadata,
    Project,
    ProjectManager,
)


@pytest.fixture
def pm(tmp_path):
    """Create a ProjectManager with a temp directory."""
    return ProjectManager(projects_dir=str(tmp_path))


@pytest.fixture
def project(pm):
    """Create a sample project."""
    return pm.create_project("test-project", name="Test", series="Naruto", chapter="1")


class TestPageInfo:
    def test_defaults(self):
        p = PageInfo(filename="page01.png")
        assert p.status == "pending"
        assert p.progress == 0.0

    def test_progress(self):
        p = PageInfo(filename="p.png", bubble_count=10, translated_count=5)
        assert p.progress == 0.5

    def test_progress_zero_bubbles_pending(self):
        p = PageInfo(filename="p.png", bubble_count=0, status="pending")
        assert p.progress == 0.0

    def test_progress_zero_bubbles_translated(self):
        p = PageInfo(filename="p.png", bubble_count=0, status="translated")
        assert p.progress == 1.0


class TestProject:
    def test_empty_project(self):
        p = Project(id="test")
        assert p.page_count == 0
        assert p.progress == 0.0
        assert p.completed_count == 0

    def test_progress(self):
        p = Project(
            id="test",
            pages=[
                PageInfo(filename="a.png", bubble_count=10, translated_count=10, status="translated"),
                PageInfo(filename="b.png", bubble_count=10, translated_count=0, status="pending"),
            ],
        )
        assert p.page_count == 2
        assert p.completed_count == 1
        assert p.progress == 0.5

    def test_summary(self):
        p = Project(
            id="test",
            metadata=ProjectMetadata(name="My Manga"),
            pages=[
                PageInfo(filename="a.png", status="translated"),
            ],
        )
        s = p.summary()
        assert "My Manga" in s
        assert "1/1" in s


class TestCreateAndLoad:
    def test_create(self, pm):
        p = pm.create_project("proj1", name="P1", source_lang="ko")
        assert p.id == "proj1"
        assert p.metadata.name == "P1"
        assert p.metadata.source_lang == "ko"
        assert p.metadata.created_at > 0

    def test_create_duplicate(self, pm):
        pm.create_project("proj1")
        with pytest.raises(ValueError, match="already exists"):
            pm.create_project("proj1")

    def test_load(self, pm, project):
        loaded = pm.load_project("test-project")
        assert loaded.id == "test-project"
        assert loaded.metadata.series == "Naruto"

    def test_load_missing(self, pm):
        with pytest.raises(FileNotFoundError):
            pm.load_project("nonexistent")

    def test_roundtrip(self, pm):
        p = pm.create_project("rt", name="Roundtrip", chapter="5")
        loaded = pm.load_project("rt")
        assert loaded.metadata.name == "Roundtrip"
        assert loaded.metadata.chapter == "5"


class TestDeleteAndList:
    def test_delete(self, pm, project):
        pm.delete_project("test-project")
        with pytest.raises(FileNotFoundError):
            pm.load_project("test-project")

    def test_delete_missing(self, pm):
        with pytest.raises(FileNotFoundError):
            pm.delete_project("nonexistent")

    def test_list_empty(self, pm):
        assert pm.list_projects() == []

    def test_list_projects(self, pm):
        pm.create_project("a")
        pm.create_project("b")
        projects = pm.list_projects()
        assert len(projects) == 2
        ids = {p.id for p in projects}
        assert ids == {"a", "b"}


class TestPageManagement:
    def test_add_pages(self, pm, project, tmp_path):
        paths = [str(tmp_path / f"page_{i}.png") for i in range(3)]
        for p in paths:
            open(p, "w").close()

        added = pm.add_pages("test-project", paths)
        assert len(added) == 3

        loaded = pm.load_project("test-project")
        assert loaded.page_count == 3

    def test_add_duplicate_skipped(self, pm, project, tmp_path):
        path = str(tmp_path / "page.png")
        open(path, "w").close()

        pm.add_pages("test-project", [path])
        added = pm.add_pages("test-project", [path])
        assert len(added) == 0

        loaded = pm.load_project("test-project")
        assert loaded.page_count == 1

    def test_update_page(self, pm, project, tmp_path):
        path = str(tmp_path / "page.png")
        open(path, "w").close()
        pm.add_pages("test-project", [path])

        updated = pm.update_page(
            "test-project", "page.png",
            status="translated", bubble_count=5, translated_count=5,
        )
        assert updated.status == "translated"
        assert updated.bubble_count == 5

    def test_update_page_not_found(self, pm, project):
        with pytest.raises(KeyError):
            pm.update_page("test-project", "missing.png", status="done")

    def test_remove_page(self, pm, project, tmp_path):
        path = str(tmp_path / "page.png")
        open(path, "w").close()
        pm.add_pages("test-project", [path])

        pm.remove_page("test-project", "page.png")
        loaded = pm.load_project("test-project")
        assert loaded.page_count == 0

    def test_get_page(self, pm, project, tmp_path):
        path = str(tmp_path / "page.png")
        open(path, "w").close()
        pm.add_pages("test-project", [path])

        page = pm.get_page("test-project", "page.png")
        assert page.filename == "page.png"

    def test_pages_sorted(self, pm, project, tmp_path):
        for name in ["c.png", "a.png", "b.png"]:
            open(str(tmp_path / name), "w").close()
        pm.add_pages("test-project", [str(tmp_path / n) for n in ["c.png", "a.png", "b.png"]])

        loaded = pm.load_project("test-project")
        filenames = [p.filename for p in loaded.pages]
        assert filenames == ["a.png", "b.png", "c.png"]


class TestSettingsOverrides:
    def test_overrides_persist(self, pm):
        p = pm.create_project("ov")
        p.settings_overrides = {"inpainting": {"method": "lama"}}
        pm.save_project(p)

        loaded = pm.load_project("ov")
        assert loaded.settings_overrides["inpainting"]["method"] == "lama"
