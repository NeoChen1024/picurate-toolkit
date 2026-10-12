"""The thumbnail list with review badges, and the image canvas for crops and fill rectangles."""

from __future__ import annotations

from PIL import Image, ImageOps
from PySide6.QtCore import QAbstractListModel, QModelIndex, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QStyledItemDelegate

from ..common.qt import Job, qimage

REVIEW_ROLE = Qt.ItemDataRole.UserRole
REVIEW_BADGES = {
    "unreviewed": ("U", "Unreviewed", "#ef8888"),
    "approved": ("A", "Approved", "#78d899"),
    "skip": ("S", "Skipped", "#7ed6df"),
}


class ThumbnailDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        letter, _, color = REVIEW_BADGES.get(index.data(REVIEW_ROLE), REVIEW_BADGES["unreviewed"])
        painter.save()
        painter.setClipRect(option.rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        size = max(24, painter.fontMetrics().height() + 6)
        badge = QRectF(option.rect.right() - size - 5, option.rect.top() + 5, size, size)
        painter.setPen(QPen(QColor("#253038"), 1))
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(badge, 4, 4)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, letter)
        painter.restore()


def thumbnail(path, size: tuple[int, int]):
    with Image.open(path) as im:
        im.draft("RGB", size)  # JPEG: decode at a reduced scale
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail(size)
    return qimage(im)


class Thumbnails(QAbstractListModel):
    def __init__(self, paths, pool):
        super().__init__()
        self.paths, self.pool = paths, pool
        self.cache, self.pending = {}, set()
        self.status = {}
        self.issues = {}
        self.excluded = {}  # row -> why

    def rowCount(self, parent=QModelIndex()):
        return len(self.paths)

    def data(self, index, role):
        row = index.row()
        if role == Qt.ItemDataRole.DisplayRole:
            return ("✕ " if row in self.excluded else "") + self.paths[row].name + (" [!]" if row in self.issues else "")
        if role == REVIEW_ROLE:
            return self.status.get(row, "unreviewed")
        if role in (Qt.ItemDataRole.ToolTipRole, Qt.ItemDataRole.AccessibleTextRole):
            status = self.status.get(row, "unreviewed")
            label = REVIEW_BADGES.get(status, REVIEW_BADGES["unreviewed"])[1]
            return (
                f"{self.paths[row].name} — {label}"
                + (f"\n{self.excluded[row]}" if row in self.excluded else "")
                + (f"\n{self.issues[row]}" if row in self.issues else "")
            )
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(160, 165)
        if role == Qt.ItemDataRole.DecorationRole:
            if row in self.cache:
                return self.cache[row]
            if row not in self.pending:
                self.pending.add(row)
                path = self.paths[row]

                def read(signals):
                    return row, thumbnail(path, (140, 130))

                self.pool.start(Job(read).connect(self.ready, lambda error, r=row: self.failed(r)))
        return None

    def failed(self, row):
        self.pending.discard(row)
        self.cache[row] = QPixmap()
        self.issues[row] = "Source image missing or unreadable"
        self.dataChanged.emit(self.index(row), self.index(row))

    def ready(self, result):
        row, im = result
        self.pending.discard(row)
        if len(self.cache) >= 300:
            self.cache.pop(next(iter(self.cache)))
        self.cache[row] = QPixmap.fromImage(im)
        self.dataChanged.emit(self.index(row), self.index(row))


class Canvas(QGraphicsView):
    rectangle = Signal(object)
    navigate_requested = Signal(int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#252525"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pix = self.scene().addPixmap(QPixmap())
        self.mode = "crop"
        self.crop = None
        self.fills = []
        self.selected_fill = -1
        self.overlays = []
        self.start = None
        self.drag = None
        self.image_size = QSize()
        self.ratio = 0
        self.space = False
        self.wheel_remainder = 0

    def set_image(self, im, fit=False):
        self.pix.setPixmap(QPixmap.fromImage(qimage(im)))
        self.image_size = QSize(im.width, im.height)
        self.scene().setSceneRect(0, 0, im.width, im.height)
        if fit:
            self.fit()
        # Overlays depend on both the current pixel dimensions and view scale.
        self.draw()

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def draw(self):
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays = []
        if self.mode == "preview" or self.pix.pixmap().isNull():
            return
        rectangles = [
            (fill["rect"], "#ff66cc" if i == self.selected_fill else "#ffcc80", i == self.selected_fill)
            for i, fill in enumerate(self.fills)
        ]
        if self.crop:
            rectangles.append((self.crop, "#80c8ff", False))
        for rect, color, selected in rectangles:
            pen = QPen(QColor(color), 3 if selected else 2)
            pen.setCosmetic(True)
            item = self.scene().addRect(QRectF(*rect), pen)
            item.setZValue(2 if selected else 0)
            self.overlays.append(item)
        if self.crop:
            x, y, w, h = self.crop
            width, height = self.image_size.width(), self.image_size.height()
            for box in ((0, 0, width, y), (0, y + h, width, height - y - h), (0, y, x, h), (x + w, y, width - x - w, h)):
                shade = self.scene().addRect(QRectF(*box), QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0, 100)))
                shade.setZValue(-0.5)
                self.overlays.append(shade)
            self.pix.setZValue(-1)
            pen = QPen(QColor("#80c8ff"), 2)
            pen.setCosmetic(True)
            size = 7 / max(self.transform().m11(), 0.001)
            for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
                self.overlays.append(self.scene().addRect(QRectF(px - size / 2, py - size / 2, size, size), pen))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.wheel_remainder = 0
            self.scale(1.2 ** (delta / 120), 1.2 ** (delta / 120))
            self.draw()
        else:
            # Accumulate high-resolution wheel events into ordinary wheel notches.
            if delta * self.wheel_remainder < 0:
                self.wheel_remainder = 0
            self.wheel_remainder += delta
            steps = int(self.wheel_remainder / 120)
            self.wheel_remainder -= steps * 120
            if steps:
                self.navigate_requested.emit(-steps)
        event.accept()

    def point(self, event):
        point = self.mapToScene(event.position().toPoint())
        return QPointF(max(0, min(self.image_size.width(), point.x())), max(0, min(self.image_size.height(), point.y())))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or (self.space and event.button() == Qt.MouseButton.LeftButton):
            self.pan = event.position().toPoint()
            return
        if event.button() != Qt.MouseButton.LeftButton or self.mode == "preview" or self.pix.pixmap().isNull():
            super().mousePressEvent(event)
            return
        self.start = self.point(event)
        self.drag = "new"
        if self.mode == "crop" and self.crop:
            x, y, w, h = self.crop
            self.original = list(self.crop)
            tolerance = 12 / max(self.transform().m11(), 0.001)
            for px, py, ox, oy in ((x, y, x + w, y + h), (x + w, y, x, y + h), (x, y + h, x + w, y), (x + w, y + h, x, y)):
                if abs(self.start.x() - px) < tolerance and abs(self.start.y() - py) < tolerance:
                    self.start = QPointF(ox, oy)
                    self.drag = "resize"
                    break
            else:
                if QRectF(*self.crop).contains(self.start):
                    self.drag = "move"

    def mouseMoveEvent(self, event):
        if getattr(self, "pan", None) is not None:
            delta = event.position().toPoint() - self.pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan = event.position().toPoint()
            return
        if self.start is None:
            super().mouseMoveEvent(event)
            return
        p = self.point(event)
        if self.drag == "move":
            x, y, w, h = self.original
            rect = [
                round(max(0, min(self.image_size.width() - w, x + p.x() - self.start.x()))),
                round(max(0, min(self.image_size.height() - h, y + p.y() - self.start.y()))),
                w,
                h,
            ]
        else:
            box = QRectF(self.start, p).normalized()
            x, y, w, h = round(box.x()), round(box.y()), round(box.width()), round(box.height())
            if self.mode == "crop" and self.ratio:
                w = min(w, round(h * self.ratio))
                h = round(w / self.ratio)
            rect = [x, y, min(w, self.image_size.width() - x), min(h, self.image_size.height() - y)]
        self.draft = rect
        if self.mode == "crop":
            self.crop = rect
        else:
            self.fills = self.fills[: getattr(self, "base_count", len(self.fills))]
            self.base_count = len(self.fills)
            self.fills.append({"rect": rect})
        self.draw()

    def mouseReleaseEvent(self, event):
        if getattr(self, "pan", None) is not None:
            self.pan = None
            return
        if self.start is not None:
            if hasattr(self, "draft") and min(self.draft[2:]) > 0:
                self.rectangle.emit(self.draft)
            self.start = None
            if hasattr(self, "draft"):
                del self.draft
            if hasattr(self, "base_count"):
                del self.base_count
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = True
        elif (
            self.mode == "crop" and self.crop and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
        ):
            x, y, w, h = self.crop
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            x += step * ((event.key() == Qt.Key.Key_Right) - (event.key() == Qt.Key.Key_Left))
            y += step * ((event.key() == Qt.Key.Key_Down) - (event.key() == Qt.Key.Key_Up))
            self.rectangle.emit([max(0, min(x, self.image_size.width() - w)), max(0, min(y, self.image_size.height() - h)), w, h])
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = False
        super().keyReleaseEvent(event)
