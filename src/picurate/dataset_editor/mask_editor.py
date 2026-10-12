"""ROI mask editing and explicit LaMa preview/accept dialog."""

from __future__ import annotations

import copy
import math

from PIL import Image, ImageDraw, ImageFilter
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..common.qt import Job, qimage
from .repair import MODEL_DEFAULTS


class MaskView(QGraphicsView):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setScene(QGraphicsScene(self))
        self.item = self.scene().addPixmap(QPixmap())
        self.last = None
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.decorations = []

    def position(self, event):
        p = self.mapToScene(event.position().toPoint())
        return (max(0, min(self.owner.image.width - 1, round(p.x()))), max(0, min(self.owner.image.height - 1, round(p.y()))))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        p = self.position(event)
        tool = self.owner.tool.currentText()
        if tool in ("Positive point", "Negative point"):
            self.owner.points.append(p)
            self.owner.labels.append(1 if tool == "Positive point" else 0)
            self.owner.invalidate()
            return
        self.last = p
        if tool == "Box prompt":
            self.box_start = p
        else:
            self.owner.history.append((self.owner.additions.copy(), self.owner.erasures.copy()))
            self.stroke(p, p)

    def stroke(self, a, b):
        erase = self.owner.tool.currentText() == "Eraser"
        width = self.owner.brush.value()
        for image, value in ((self.owner.additions, 0 if erase else 255), (self.owner.erasures, 255 if erase else 0)):
            draw = ImageDraw.Draw(image)
            draw.line([a, b], fill=value, width=width)
            r = width / 2
            for x, y in (a, b):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=value)
        self.owner.invalidate()

    def mouseMoveEvent(self, event):
        if self.last is None:
            return
        p = self.position(event)
        if self.owner.tool.currentText() == "Box prompt":
            a = self.box_start
            self.owner.box = [min(a[0], p[0]), min(a[1], p[1]), max(a[0], p[0]), max(a[1], p[1])]
            self.owner.invalidate()
        else:
            self.stroke(self.last, p)
        self.last = p

    def mouseReleaseEvent(self, event):
        self.last = None

    def wheelEvent(self, event):
        self.scale(1.2 ** (event.angleDelta().y() / 120), 1.2 ** (event.angleDelta().y() / 120))
        event.accept()


class MaskEditor(QDialog):
    def __init__(self, image, rect, settings, engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LaMa — Mask Editor")
        self.resize(1050, 700)
        self.full_image = image.copy()
        self.rect = list(rect)
        x, y, w, h = rect
        self.image = image.crop((x, y, x + w, y + h))
        self.settings = copy.deepcopy({**MODEL_DEFAULTS, **settings})
        self.engine = engine
        self.logits = None
        self.additions = Image.new("L", self.image.size)
        self.erasures = Image.new("L", self.image.size)
        self.history = []
        self.points = []
        self.labels = []
        self.box = None
        self.result = None
        self.mask_metadata = {"backend": "Manual"}
        self.lama_metadata = {}
        self.busy = False
        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        outer.addLayout(body, 1)
        self.view = MaskView(self)
        body.addWidget(self.view, 1)
        controls = QVBoxLayout()
        body.addLayout(controls)
        form = QFormLayout()
        controls.addLayout(form)
        self.backend = QComboBox()
        self.backend.addItems(["Hi-SAM", "SAM-TS", "SAM 2.1", "Manual"])
        self.backend.setCurrentText(self.settings["mask_backend"])
        form.addRow("Mask backend", self.backend)
        self.tool = QComboBox()
        self.tool.addItems(["Brush", "Eraser", "Box prompt", "Positive point", "Negative point"])
        form.addRow("Tool", self.tool)
        self.brush = QSpinBox()
        self.brush.setRange(1, 200)
        self.brush.setValue(12)
        form.addRow("Brush diameter (px)", self.brush)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 0.99)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        form.addRow("Mask threshold", self.threshold)
        self.expansion = QSpinBox()
        self.expansion.setRange(-20, 30)
        self.expansion.setValue(self.settings["expansion"])
        form.addRow("Expand / contract (px)", self.expansion)
        self.overlay = QCheckBox("Show mask overlay")
        self.overlay.setChecked(True)
        controls.addWidget(self.overlay)
        self.generate = QPushButton("Generate Mask")
        self.generate.clicked.connect(self.generate_mask)
        controls.addWidget(self.generate)
        clear = QPushButton("Clear Mask / Prompts")
        clear.clicked.connect(self.clear_mask)
        controls.addWidget(clear)
        undo = QPushButton("Undo Brush Stroke")
        undo.clicked.connect(self.undo_stroke)
        controls.addWidget(undo)
        self.edit_buttons = [clear, undo]
        self.run = QPushButton("Run LaMa")
        self.run.clicked.connect(self.run_lama)
        controls.addWidget(self.run)
        self.preview = QCheckBox("Show repaired preview")
        self.preview.toggled.connect(self.redraw)
        controls.addWidget(self.preview)
        hint = QLabel(
            "Brush / eraser edits the mask.\nSAM 2: draw a tight box or add points,\nthen Generate Mask.\nWheel zooms; scrollbars pan.\nExpansion uses original-image pixels."
        )
        controls.addWidget(hint)
        controls.addStretch()
        self.status = QLabel("Generate or paint a mask, then run LaMa.")
        outer.addWidget(self.status)
        buttons = QHBoxLayout()
        outer.addLayout(buttons)
        self.accept_button = QPushButton("Accept Repair")
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Discard")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.cancel_button)
        self.threshold.valueChanged.connect(self.invalidate)
        self.expansion.valueChanged.connect(self.invalidate)
        self.overlay.toggled.connect(self.redraw)
        self.backend.currentTextChanged.connect(self.backend_changed)
        self.redraw()
        self.view.setSceneRect(0, 0, self.image.width, self.image.height)
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def backend_changed(self):
        self.clear_mask()

    def mask(self):
        import numpy as np

        if self.logits is None:
            mask = Image.new("L", self.image.size)
        else:
            threshold = math.log(self.threshold.value() / (1 - self.threshold.value()))
            mask = Image.fromarray((self.logits > threshold).astype("uint8") * 255)
        n = self.expansion.value()
        if n:
            mask = mask.filter(ImageFilter.MaxFilter(2 * n + 1) if n > 0 else ImageFilter.MinFilter(-2 * n + 1))
        values = np.array(mask)
        values[np.array(self.additions) > 0] = 255
        values[np.array(self.erasures) > 0] = 0
        return Image.fromarray(values)

    def invalidate(self):
        self.result = None
        self.accept_button.setEnabled(False)
        self.preview.setChecked(False)
        self.redraw()

    def redraw(self):
        base = self.result if self.preview.isChecked() and self.result is not None else self.image
        shown = base.convert("RGB")
        if self.overlay.isChecked() and not (self.preview.isChecked() and self.result is not None):
            tinted = Image.blend(shown, Image.new("RGB", shown.size, (255, 50, 70)), 0.5)
            shown = Image.composite(tinted, shown, self.mask())
        self.view.item.setPixmap(QPixmap.fromImage(qimage(shown)))
        for item in self.view.decorations:
            self.view.scene().removeItem(item)
        self.view.decorations = []
        if not self.preview.isChecked():
            if self.box:
                x0, y0, x1, y1 = self.box
                pen = QPen(QColor("#00aaff"), 2)
                pen.setCosmetic(True)
                self.view.decorations.append(self.view.scene().addRect(x0, y0, x1 - x0, y1 - y0, pen))
            for (x, y), label in zip(self.points, self.labels):
                pen = QPen(QColor("#00ff00" if label else "#ff3333"), 2)
                pen.setCosmetic(True)
                self.view.decorations.append(self.view.scene().addEllipse(x - 3, y - 3, 6, 6, pen))

    def clear_mask(self):
        self.logits = None
        self.points = []
        self.labels = []
        self.box = None
        self.additions = Image.new("L", self.image.size)
        self.erasures = Image.new("L", self.image.size)
        self.history = []
        self.mask_metadata = {"backend": "Manual"}
        self.invalidate()

    def undo_stroke(self):
        if self.history:
            self.additions, self.erasures = self.history.pop()
            self.invalidate()

    def task(self, title, fn, receive):
        self.busy = True
        self.status.setText(title)
        self.view.setEnabled(False)
        for widget in (
            *self.edit_buttons,
            self.backend,
            self.tool,
            self.threshold,
            self.expansion,
            self.generate,
            self.run,
            self.cancel_button,
            self.accept_button,
        ):
            widget.setEnabled(False)
        job = Job(lambda signals: fn())

        def finish(result=None, error=None):
            self.busy = False
            self.view.setEnabled(True)
            for widget in (
                *self.edit_buttons,
                self.backend,
                self.tool,
                self.threshold,
                self.expansion,
                self.generate,
                self.run,
                self.cancel_button,
            ):
                widget.setEnabled(True)
            if error:
                self.status.setText(error)
                QMessageBox.warning(self, "Inference failed", error)
            else:
                receive(result)

        job.connect(lambda result: finish(result=result), lambda error: finish(error=error))
        QThreadPool.globalInstance().start(job)

    def generate_mask(self):
        if self.backend.currentText() == "Manual":
            self.status.setText("Paint a mask with Brush / Eraser.")
            return
        self.invalidate()
        settings = {**self.settings, "mask_backend": self.backend.currentText()}
        box = copy.deepcopy(self.box)
        points = copy.deepcopy(self.points)
        labels = list(self.labels)

        def receive(result):
            self.logits, self.mask_metadata = result
            self.invalidate()
            self.status.setText(f"Mask ready — {self.mask_metadata['device']}; inspect before running LaMa")

        self.task(
            "Loading / generating mask… (first use may download model weights)",
            lambda: self.engine.segment(self.image, settings, box, points, labels),
            receive,
        )

    def run_lama(self):
        mask = self.mask()
        if not mask.getbbox():
            self.status.setText("The mask is empty. Generate or paint it first.")
            return
        x, y, w, h = self.rect
        n = self.settings["context"]
        left, top = max(0, x - n), max(0, y - n)
        right, bottom = min(self.full_image.width, x + w + n), min(self.full_image.height, y + h + n)
        context = self.full_image.crop((left, top, right, bottom))
        context_mask = Image.new("L", context.size)
        context_mask.paste(mask, (x - left, y - top))
        self.result = None

        def receive(result):
            image, self.lama_metadata = result
            self.result = image.crop((x - left, y - top, x - left + w, y - top + h))
            self.accepted_mask = mask
            self.preview.setChecked(True)
            self.accept_button.setEnabled(True)
            self.status.setText(f"Repair ready — {self.lama_metadata['device']}; Accept saves it permanently")
            self.redraw()

        self.task(
            "Loading / running LaMa… (first use may download model weights)",
            lambda: self.engine.inpaint(context, context_mask, self.settings),
            receive,
        )

    def reject(self):
        if not self.busy:
            super().reject()

    def closeEvent(self, event):
        if self.busy:
            event.ignore()
        else:
            super().closeEvent(event)
