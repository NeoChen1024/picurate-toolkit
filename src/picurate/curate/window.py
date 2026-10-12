"""The curation window: one project, a tab per kind of decision about its images: Duplicates and Culling."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from ..dedupe.keeping import Policy
from ..dedupe.review import ReviewDB
from ..project.project import REVIEW, Project
from .culling import CullingTab
from .duplicates import DuplicatesTab


class CurateWindow(QMainWindow):
    """Holds the project's review lock while open, so a dedupe run cannot rewrite the groups being decided."""

    def __init__(self, project_dir, *, policies: dict[str, Policy] | None = None, prefetch: int = 3):
        super().__init__()
        self.project = Project(project_dir)
        try:
            self.project.lock("review")
        except ValueError:
            self.project.close()
            raise
        self.setWindowTitle(f"Curate — {self.project.directory.name}")
        self.resize(1500, 950)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.review: ReviewDB | None = None
        self.duplicates: DuplicatesTab | None = None
        path = self.project.directory / REVIEW
        if path.exists():
            self.review = ReviewDB(str(path))
            self.duplicates = DuplicatesTab(self.project, self.review, policies=policies, prefetch=prefetch)
            self.tabs.addTab(self.duplicates, "Duplicates")
        else:
            label = QLabel(f"No duplicate groups yet: run  picurate-dedupe dedupe {self.project.directory}")
            label.setMargin(24)
            self.tabs.addTab(label, "Duplicates")
        self.culling = CullingTab(self.project)
        self.tabs.addTab(self.culling, "Culling")
        self.tabs.currentChanged.connect(lambda _: self.tabs.currentWidget() is self.culling and self.culling.activate())

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.duplicates is not None:
            self.duplicates.shutdown()
        self.culling.shutdown()
        if self.review is not None:
            self.review.close()
        self.project.close()
        super().closeEvent(event)
