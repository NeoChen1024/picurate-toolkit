"""The comparison canvas: two image views sharing one zoom/pan, with flip and difference modes."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QHBoxLayout, QWidget

MODES = ("side", "flip", "diff")


class ImageView(QGraphicsView):
    """One pixmap in a scene; zoom/pan requests are forwarded to the canvas so both views move together."""

    wheel = Signal(object)  # the QWheelEvent
    space_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#252525"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(self.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pix = self.scene().addPixmap(QPixmap())
        self.image_size = (0, 0)
        self.auto_fit = False

    def set_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self.pix.setPixmap(QPixmap())
            self.image_size = (0, 0)
            self.scene().setSceneRect(QRectF())
            return
        self.pix.setPixmap(QPixmap.fromImage(image))
        self.image_size = (image.width(), image.height())
        self.scene().setSceneRect(0, 0, image.width(), image.height())
        if self.auto_fit:
            self.fit()

    def fit(self) -> None:
        """Scale the image to the viewport and keep doing so whenever the viewport changes size.

        The canvas asks for a fit while the layout is still stale (a view just shown, the splitter just moved,
        the window not yet maximised), so the one-shot ``fitInView`` was often computed against the wrong size.
        Fitting again from ``resizeEvent`` means the last fit always sees the final viewport. The size used is the
        one without scrollbars, so a fit after 100% does not shrink to make room for bars that then disappear.
        """
        self.auto_fit = True
        iw, ih = self.image_size
        size = self.maximumViewportSize()
        vw, vh = size.width() - 2, size.height() - 2  # the margin Qt's fitInView uses, against rounding into scrollbars
        if not iw or not ih or vw <= 0 or vh <= 0:
            return
        scale = min(vw / iw, vh / ih)
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(self.sceneRect().center())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.wheel.emit(event)
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_changed.emit(True)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_changed.emit(False)
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Re-dispatch as a left press so the hand drag starts.
            fake = type(event)(event.type(), event.position(), event.globalPosition(), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, event.modifiers())
            super().mousePressEvent(fake)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            fake = type(event)(event.type(), event.position(), event.globalPosition(), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, event.modifiers())
            super().mouseReleaseEvent(fake)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        super().mouseReleaseEvent(event)


class CompareCanvas(QWidget):
    """Two linked :class:`ImageView`s.

    ``side``: A left, B right. ``flip``: one view showing A, or B while ``flipped`` (Space) is held. ``diff``: one
    view showing the amplified absolute difference over A at half brightness, or on black while ``flipped`` is held
    (both supplied by the window, computed off-thread).
    """

    navigate = Signal(int)  # +1 / -1 from the wheel
    mode_changed = Signal(str)
    zoom_changed = Signal(str)  # "fit", "actual" or "free"

    def __init__(self) -> None:
        super().__init__()
        self.left = ImageView()
        self.right = ImageView()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.left)
        layout.addWidget(self.right)
        self.mode = "side"
        self.flipped = False
        self.a: QImage | None = None
        self.b: QImage | None = None
        self.diff: QImage | None = None
        self.diff_plain: QImage | None = None
        self._auto_fit = True
        self.zoom_state = "fit"
        self._syncing = False
        for view in (self.left, self.right):
            view.wheel.connect(self._wheel)
            view.space_changed.connect(self._space)
        for a, b in ((self.left, self.right), (self.right, self.left)):
            a.horizontalScrollBar().valueChanged.connect(lambda v, src=a, other=b: self._sync_scroll(src, other, "h", v))
            a.verticalScrollBar().valueChanged.connect(lambda v, src=a, other=b: self._sync_scroll(src, other, "v", v))

    # -- content ---------------------------------------------------------------------------------------------------

    def set_images(self, a: QImage | None, b: QImage | None, *, fit: bool = True) -> None:
        self.a, self.b, self.diff, self.diff_plain = a, b, None, None
        self._show()
        if fit:
            self.fit()

    def set_diff(self, diff: QImage | None, plain: QImage | None = None) -> None:
        self.diff, self.diff_plain = diff, plain
        if self.mode == "diff":
            self._show_keeping_view()

    def _single_image(self) -> QImage | None:
        if self.mode == "flip":
            return self.b if self.flipped else self.a
        return (self.diff_plain if self.flipped else self.diff) if self.mode == "diff" else None

    def _show_keeping_view(self) -> None:
        """Swap the single view's image without losing zoom or scroll position.

        In fit mode the swap fits the new image itself, and the saved transform must not be put back: in diff mode
        it would be the one from before the diff was computed, i.e. the previous group's scale."""
        transform = self.left.transform()
        h, v = self.left.horizontalScrollBar().value(), self.left.verticalScrollBar().value()
        self.left.set_image(self._single_image())
        if self.left.auto_fit:
            return
        self.left.setTransform(transform)
        self.left.horizontalScrollBar().setValue(h)
        self.left.verticalScrollBar().setValue(v)

    def clear(self) -> None:
        self.set_images(None, None)

    def _show(self) -> None:
        if self.mode == "side":
            self.right.show()
            self.left.set_image(self.a)
            self.right.set_image(self.b)
        else:
            self.right.hide()
            self.left.set_image(self._single_image())

    # -- modes -----------------------------------------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        if mode == self.mode:
            return
        transform = self.left.transform()
        self.mode = mode
        self._show()
        self.left.setTransform(transform)
        self.right.setTransform(transform)
        if self._auto_fit:
            self.fit()
        self.mode_changed.emit(mode)

    def cycle_mode(self) -> str:
        self.set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])
        return self.mode

    def set_flipped(self, flipped: bool) -> None:
        if flipped == self.flipped:
            return
        self.flipped = flipped
        if self.mode in ("flip", "diff"):
            self._show_keeping_view()

    # -- zoom / pan ------------------------------------------------------------------------------------------------

    def fit(self) -> None:
        """Fit each visible view to its own image. The two pictures may have very different resolutions, so they
        get different scales; zooming keeps multiplying both, and scrolling is linked proportionally. Each view
        keeps refitting itself as its viewport changes until the zoom is changed by hand."""
        self._set_zoom_state("fit")
        for view in (self.left, self.right):
            view.fit()

    def _set_zoom_state(self, state: str) -> None:
        self._auto_fit = state == "fit"
        if state != self.zoom_state:
            self.zoom_state = state
            self.zoom_changed.emit(state)

    def actual_size(self) -> None:
        self._set_zoom_state("actual")
        for view in (self.left, self.right):
            view.auto_fit = False
            view.resetTransform()

    def zoom(self, factor: float) -> None:
        self._set_zoom_state("free")
        for view in (self.left, self.right):
            view.auto_fit = False
            view.scale(factor, factor)

    def scale_factor(self) -> float:
        return self.left.transform().m11()

    def _wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if not delta:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.2 ** (delta / 120))
        else:
            self.navigate.emit(-1 if delta > 0 else 1)

    def _space(self, down: bool) -> None:
        mode = QGraphicsView.DragMode.ScrollHandDrag if down else QGraphicsView.DragMode.NoDrag
        for view in (self.left, self.right):
            view.setDragMode(mode)

    def _sync_scroll(self, source: ImageView, other: ImageView, axis: str, value: int) -> None:
        """Keep the two views at the same relative position: scales differ, so map by fraction of the range."""
        if self._syncing or self.mode != "side":
            return
        src = source.horizontalScrollBar() if axis == "h" else source.verticalScrollBar()
        dst = other.horizontalScrollBar() if axis == "h" else other.verticalScrollBar()
        span = src.maximum() - src.minimum()
        if span <= 0 or dst.maximum() - dst.minimum() <= 0:
            return
        fraction = (value - src.minimum()) / span
        self._syncing = True
        try:
            dst.setValue(round(dst.minimum() + fraction * (dst.maximum() - dst.minimum())))
        finally:
            self._syncing = False
