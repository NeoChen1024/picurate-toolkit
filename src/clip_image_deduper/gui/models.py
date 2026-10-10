"""List models and delegates for the review window."""

from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QStyledItemDelegate

from ..review import GroupRow

STATUS_ROLE = Qt.ItemDataRole.UserRole
GROUP_ROLE = Qt.ItemDataRole.UserRole + 1

# letter, label, colour: the editor's U/A/S badges, with the review statuses
STATUS_BADGES = {
    "pending": ("P", "Pending", "#ef8888"),
    "decided": ("D", "Decided", "#78d899"),
    "skipped": ("S", "Skipped", "#7ed6df"),
    "applied": ("A", "Applied", "#9a9a9a"),
    "stale": ("!", "Stale", "#d8b878"),
}


class BadgeDelegate(QStyledItemDelegate):
    """Draws a coloured status letter in the top-right corner of each row."""

    def paint(self, painter: QPainter, option, index) -> None:
        super().paint(painter, option, index)
        letter, _, colour = STATUS_BADGES.get(index.data(STATUS_ROLE), STATUS_BADGES["pending"])
        painter.save()
        painter.setClipRect(option.rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        size = max(18, painter.fontMetrics().height() + 2)
        badge = QRectF(option.rect.right() - size - 4, option.rect.top() + (option.rect.height() - size) / 2, size, size)
        painter.setPen(QPen(QColor("#253038"), 1))
        painter.setBrush(QColor(colour))
        painter.drawRoundedRect(badge, 4, 4)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, letter)
        painter.restore()


class GroupListModel(QAbstractListModel):
    """Rows of :class:`GroupRow`, displayed as ``<n> images · <min distance>``."""

    def __init__(self, groups: list[GroupRow] | None = None):
        super().__init__()
        self.groups: list[GroupRow] = groups or []

    def set_groups(self, groups: list[GroupRow]) -> None:
        self.beginResetModel()
        self.groups = list(groups)
        self.endResetModel()

    def update_group(self, group: GroupRow) -> int:
        """Replace the row with the same id (status or note changed). Returns its row, or -1."""
        for row, g in enumerate(self.groups):
            if g.id == group.id:
                self.groups[row] = group
                self.dataChanged.emit(self.index(row), self.index(row))
                return row
        return -1

    def row_of(self, group_id: int) -> int:
        for row, g in enumerate(self.groups):
            if g.id == group_id:
                return row
        return -1

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API  # type: ignore[override]
        return 0 if parent.isValid() else len(self.groups)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid() or index.row() >= len(self.groups):
            return None
        g = self.groups[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            mark = ("✕ " if g.discard else "∅ " if g.kept == 0 else "") + ("≈ " if g.pixel == "same" else "")
            return f"{mark}{g.n} images · {g.min_distance:.2f}" + (f" – {g.max_distance:.2f}" if g.max_distance - g.min_distance > 0.005 else "")
        if role == STATUS_ROLE:
            return g.status
        if role == GROUP_ROLE:
            return g
        if role == Qt.ItemDataRole.ToolTipRole:
            label = STATUS_BADGES.get(g.status, STATUS_BADGES["pending"])[1]
            pixel = {"same": "\nPixels: every pair differs by compression only", "different": "\nPixels: some pair differs by more than compression"}.get(g.pixel or "", "")
            if g.discard:
                keeps = "\n✕ Discarded: Apply moves every member"
            elif g.kept == 0:
                keeps = "\n∅ Keeps nothing: Apply skips this group until a member is kept or the group is discarded (R)"
            else:
                keeps = ""
            return f"Group {g.id}: {g.n} images, distances {g.min_distance:.3f} to {g.max_distance:.3f} — {label}{pixel}{keeps}" + (f"\n{g.note}" if g.note else "")
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(180, 26)
        return None
