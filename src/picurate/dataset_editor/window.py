"""The editor's main window: project handling, navigation, edit history, auto-save and export."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..common.images import PathFilter
from ..common.qt import Job
from ..project.project import Project, check_image_size, validate_source
from .canvas import Canvas, ThumbnailDelegate, Thumbnails
from .caption_editor import CaptionEditor
from .mask_editor import MaskEditor
from .render import export_project, load_image, render, save_repair
from .repair import MODEL_DEFAULTS, InferenceEngine
from .tags import TagIndex

# Automatic: excluded when a decided duplicate group does not keep the image.
EXCLUSION_CHOICES = ("automatic", "force keep", "force exclude")


class Window(QMainWindow):
    """One project; captions and edits auto-save 500 ms after a change, before navigation and on close."""

    def __init__(self, project_dir=None):
        super().__init__()
        self.setWindowTitle("Dataset Editor")
        self.resize(1450, 950)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self.project = None
        self.op = None
        self.inference_engine = InferenceEngine()
        self.state = None
        self.im = None
        self.paths = []
        self.history, self.future = [], []
        self.generation = 0
        self.saving = QTimer(self)
        self.saving.setSingleShot(True)
        self.saving.setInterval(500)
        self.saving.timeout.connect(self.save)
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        self.action(toolbar, "New Project", self.new_project, "Ctrl+N")
        self.action(toolbar, "Open Project", self.open_project, "Ctrl+O")
        self.action(toolbar, "Import Images", self.import_images)
        self.action(toolbar, "Relink Sources", self.relink_sources)
        self.action(toolbar, "Save", self.save, "Ctrl+S")
        self.action(toolbar, "Settings", self.project_settings)
        toolbar.addSeparator()
        self.edit_mode = "crop"
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions = {}
        for text, mode in [("Crop (A)", "crop"), ("Watermark Fill (S)", "fill")]:
            action = self.action(toolbar, text, lambda checked=False, m=mode: self.mode(m))
            action.setCheckable(True)
            self.tool_group.addAction(action)
            self.tool_actions[mode] = action
        self.tool_actions["crop"].setChecked(True)
        self.preview_action = self.action(toolbar, "Preview (D)", self.toggle_preview)
        self.preview_action.setCheckable(True)
        self.preview_action.setToolTip("Toggle output preview; click again to return to editing")
        self.action(toolbar, "Fit", lambda: self.canvas.fit())
        self.action(toolbar, "100%", lambda: self.canvas.resetTransform())
        self.action(toolbar, "Previous (Z)", lambda: self.navigate(-1), "Alt+Left")
        self.action(toolbar, "Next (X)", lambda: self.navigate(1), "Alt+Right")
        self.action(toolbar, "Export…", self.export)
        toolbar.addSeparator()
        self.action(toolbar, "Help", self.show_help, "F1")
        self.thumbs = QListView()
        self.thumbs.setItemDelegate(ThumbnailDelegate(self.thumbs))
        self.thumbs.setViewMode(QListView.ViewMode.IconMode)
        self.thumbs.setFlow(QListView.Flow.TopToBottom)
        self.thumbs.setWrapping(False)
        self.thumbs.setMovement(QListView.Movement.Static)
        self.thumbs.setResizeMode(QListView.ResizeMode.Adjust)
        self.thumbs.setIconSize(QSize(140, 130))
        self.thumbs.setMinimumWidth(160)
        self.thumbs.setMaximumWidth(300)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setWordWrap(True)
        self.canvas = Canvas()
        self.canvas.rectangle.connect(self.rectangle)
        self.canvas.navigate_requested.connect(self.navigate)
        self.caption = CaptionEditor()
        self.caption.exit_requested.connect(self.canvas.setFocus)
        self.caption.setPlaceholderText("Enter captions or tags freely\nTab completes · Enter inserts a newline · Ctrl+R searches")
        self.caption.textChanged.connect(self.caption_changed)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.filename = QLabel("Create or open a project to start")
        self.filename.setWordWrap(True)
        layout.addWidget(self.filename)
        layout.addWidget(QLabel("Caption / tags"))
        layout.addWidget(self.caption, 1)
        self.spaces = QCheckBox("Completion: underscores to spaces")
        self.escape = QCheckBox("Completion: escape parentheses")
        self.spaces.setChecked(True)
        for widget in (self.spaces, self.escape):
            widget.toggled.connect(self.settings)
            layout.addWidget(widget)
        self.index_label = QLabel("Loading tag index…")
        layout.addWidget(self.index_label)
        form = QFormLayout()
        self.ratio = QComboBox()
        self.ratio.addItems(["Free", "Original", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"])
        self.ratio.currentTextChanged.connect(self.set_ratio)
        form.addRow("Crop ratio", self.ratio)
        self.fill_method = QComboBox()
        self.fill_method.addItems(["Mean Color", "LaMa"])
        self.fill_method.currentTextChanged.connect(self.fill_method_changed)
        form.addRow("Fill method", self.fill_method)
        self.coords = []
        for name in ("X", "Y", "W", "H"):
            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.editingFinished.connect(self.numeric_crop)
            self.coords.append(spin)
            form.addRow(name, spin)
        self.ring = QSpinBox()
        self.ring.setRange(1, 100)
        self.ring.setValue(4)
        form.addRow("New fill ring (px)", self.ring)
        layout.addLayout(form)
        self.fills = QListWidget()
        self.fills.currentRowChanged.connect(self.select_fill)
        self.fills.setMaximumHeight(95)
        layout.addWidget(self.fills)
        buttons = QHBoxLayout()
        for label, fn in [
            ("Delete Fill", self.delete_fill),
            ("Reset Crop", self.clear_crop),
            ("Undo Image", self.undo),
            ("Redo", self.redo),
        ]:
            button = QPushButton(label)
            button.clicked.connect(fn)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.review = QComboBox()
        self.review.addItems(["unreviewed", "approved", "skip"])
        self.review.currentTextChanged.connect(self.review_changed)
        layout.addWidget(QLabel("Review status (C to cycle)"))
        layout.addWidget(self.review)
        self.exclusion = QComboBox()
        self.exclusion.addItems(EXCLUSION_CHOICES)
        self.exclusion.setToolTip(
            "Automatic: excluded when a decided duplicate group does not keep it (picurate-curate). "
            "Force keep / force exclude override that; excluded images are not exported."
        )
        self.exclusion.currentTextChanged.connect(self.exclusion_changed)
        self.exclusion_reason = QLineEdit()
        self.exclusion_reason.setPlaceholderText("Reason (optional)")
        self.exclusion_reason.editingFinished.connect(self.exclusion_reason_edited)
        self.exclusion_label = QLabel()
        self.exclusion_label.setWordWrap(True)
        layout.addWidget(QLabel("Exclusion (E to cycle)"))
        row = QHBoxLayout()
        row.addWidget(self.exclusion)
        row.addWidget(self.exclusion_reason, 1)
        layout.addLayout(row)
        layout.addWidget(self.exclusion_label)
        QApplication.instance().installEventFilter(self)
        split = QSplitter()
        split.addWidget(self.thumbs)
        split.addWidget(self.canvas)
        split.addWidget(panel)
        split.setSizes([180, 850, 420])
        self.setCentralWidget(split)
        self.image_position = QLabel("Image 0 / 0")
        self.image_position.setContentsMargins(6, 0, 12, 0)
        self.status_message = QLabel()
        self.statusBar().addPermanentWidget(self.image_position)
        self.statusBar().addPermanentWidget(self.status_message, 1)
        self.status_message.setText("Ready · Source images are never overwritten")
        self.pool.start(
            Job(lambda signals: TagIndex()).connect(
                self.index_ready, lambda error: self.index_label.setText("Tag index failed: " + error)
            )
        )
        if project_dir:
            QTimer.singleShot(0, lambda: self.start_project(Path(project_dir)))

    def action(self, toolbar, label, fn, shortcut=None):
        action = QAction(label, self)
        action.triggered.connect(fn)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        toolbar.addAction(action)
        return action

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and QApplication.activeWindow() is self:
            focus = QApplication.focusWidget()
            widget = focus
            while widget is not None:
                if isinstance(widget, (QPlainTextEdit, QTextEdit, QLineEdit, QAbstractSpinBox)):
                    return super().eventFilter(obj, event)
                widget = widget.parentWidget()
            if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                if event.key() == Qt.Key.Key_D:
                    if not event.isAutoRepeat():
                        self.preview_action.trigger()
                    return True
                if event.key() == Qt.Key.Key_F:
                    if not event.isAutoRepeat():
                        self.caption.setFocus()
                    return True
                if event.key() == Qt.Key.Key_Z:
                    self.navigate(-1)
                    return True
                if event.key() == Qt.Key.Key_X:
                    self.navigate(1)
                    return True
                if event.key() in (Qt.Key.Key_A, Qt.Key.Key_S):
                    self.mode("crop" if event.key() == Qt.Key.Key_A else "fill")
                    return True
                if event.key() == Qt.Key.Key_C:
                    if self.state and not event.isAutoRepeat():
                        self.review.setCurrentIndex((self.review.currentIndex() + 1) % self.review.count())
                    return True
                if event.key() == Qt.Key.Key_E:
                    if self.state and not event.isAutoRepeat():
                        self.exclusion.setCurrentIndex((self.exclusion.currentIndex() + 1) % self.exclusion.count())
                    return True
        return super().eventFilter(obj, event)

    def toggle_preview(self, checked):
        self.mode("preview" if checked else self.edit_mode)

    def show_help(self):
        if not hasattr(self, "help_dialog"):
            self.help_dialog = QDialog(self)
            self.help_dialog.setWindowTitle("Help — Keyboard Shortcuts")
            self.help_dialog.resize(680, 640)
            layout = QVBoxLayout(self.help_dialog)
            browser = QTextBrowser()
            browser.setHtml("""
                <h2>Keyboard Shortcuts</h2>
                <h3>General</h3>
                <table cellspacing="8">
                <tr><td><b>F1</b></td><td>Open this help</td></tr>
                <tr><td><b>Ctrl+N</b></td><td>New project</td></tr>
                <tr><td><b>Ctrl+O</b></td><td>Open project</td></tr>
                <tr><td><b>Ctrl+S</b></td><td>Save edit records</td></tr>
                <tr><td><b>Z / X</b></td><td>Previous / next image</td></tr>
                <tr><td><b>Alt+Left / Alt+Right</b></td><td>Previous / next image (also while editing text)</td></tr>
                <tr><td><b>C</b></td><td>Cycle Unreviewed → Approved → Skipped</td></tr>
                <tr><td><b>E</b></td><td>Cycle exclusion: automatic → force keep → force exclude</td></tr>
                <tr><td><b>A / S</b></td><td>Crop / Watermark Fill mode; exit Preview</td></tr>
                <tr><td><b>D</b></td><td>Toggle Preview</td></tr>
                <tr><td><b>F</b></td><td>Focus the caption editor</td></tr>
                </table>
                <p>Z, X, C, A, S, D and F are disabled in text and numeric inputs.</p>
                <h3>Image canvas</h3>
                <table cellspacing="8">
                <tr><td><b>Wheel up / down</b></td><td>Previous / next image</td></tr>
                <tr><td><b>Ctrl+wheel</b></td><td>Zoom in / out</td></tr>
                <tr><td><b>Space+left drag</b></td><td>Pan (middle-button drag also works)</td></tr>
                <tr><td><b>Arrow keys</b></td><td>Move the crop by 1 pixel in Crop mode</td></tr>
                <tr><td><b>Shift+arrow keys</b></td><td>Move the crop by 10 pixels</td></tr>
                </table>
                <p>Click Preview to toggle the output view. Click it again to return to the previous tool.
                Image undo/redo is available through the Undo Image / Redo buttons.</p>
                <h3>Caption editor</h3>
                <table cellspacing="8">
                <tr><td><b>Ctrl+R</b></td><td>Show completion candidates</td></tr>
                <tr><td><b>Tab</b></td><td>Accept the selected completion when the list is open</td></tr>
                <tr><td><b>Up / Down</b></td><td>Select a candidate when the list is open</td></tr>
                <tr><td><b>Enter</b></td><td>Insert a newline; never accept a completion</td></tr>
                <tr><td><b>Comma</b></td><td>Insert a comma and end completion</td></tr>
                <tr><td><b>Esc</b></td><td>Dismiss completion; if already closed, return focus to the image canvas</td></tr>
                <tr><td><b>Ctrl+W</b></td><td>Delete the preceding word</td></tr>
                <tr><td><b>Ctrl+U / Ctrl+K</b></td><td>Delete to the start / end of the line</td></tr>
                <tr><td><b>Ctrl+Z</b></td><td>Undo a text edit</td></tr>
                <tr><td><b>Ctrl+A / C / X / V</b></td><td>Select all / copy / cut / paste</td></tr>
                </table>
                <p>Ctrl+W/U/K delete the selection if there is one. Selecting a tag or placing the caret
                inside it highlights its other duplicates in yellow. Manual text is never auto-corrected.</p>
            """)
            layout.addWidget(browser)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.help_dialog.reject)
            layout.addWidget(buttons)
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def error(self, message):
        QMessageBox.warning(self, "Unable to complete", str(message))

    def index_ready(self, index):
        self.caption.index = index
        self.index_label.setText(f"{len(index.tags):,} tags · Ctrl+R to search")

    def settings(self):
        self.caption.spaces, self.caption.escape = self.spaces.isChecked(), self.escape.isChecked()
        if self.project:
            self.project.settings.update(spaces=self.caption.spaces, escape=self.caption.escape)
            self.project.save_settings()

    def fill_method_changed(self, value):
        if self.project:
            self.project.settings["fill_method"] = value
            self.project.save_settings()

    def project_settings(self):
        if not self.project:
            self.error("Create or open a project first")
            return
        current = {**MODEL_DEFAULTS, **self.project.settings.get("inference", {})}
        dialog = QDialog(self)
        dialog.setWindowTitle("Project Settings — Inference")
        form = QFormLayout(dialog)
        backend = QComboBox()
        backend.addItems(["Hi-SAM", "SAM-TS", "SAM 2.1", "Manual"])
        backend.setCurrentText(current["mask_backend"])
        form.addRow("Default mask backend", backend)
        devices = []
        for key, label in (("mask_device", "Mask device"), ("lama_device", "LaMa device")):
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(["Auto", "CPU", "cuda:0", "cuda:1"])
            combo.setCurrentText(current[key])
            form.addRow(label, combo)
            devices.append(combo)
        expansion = QSpinBox()
        expansion.setRange(-20, 30)
        expansion.setValue(current["expansion"])
        context = QSpinBox()
        context.setRange(0, 2048)
        context.setValue(current["context"])
        offline = QCheckBox("Use cached models only (no downloads)")
        offline.setChecked(current["offline"])
        form.addRow("Default mask expansion (px)", expansion)
        form.addRow("LaMa context padding (px)", context)
        form.addRow(offline)
        form.addRow(
            QLabel(
                "Weights use Hugging Face's shared cache. Models load on first use.\nExplicit CUDA never silently falls back to CPU. Existing repairs stay unchanged."
            )
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            values = [widget.currentText().strip() for widget in devices]
            if any(value not in ("Auto", "CPU") and not (value.startswith("cuda:") and value[5:].isdigit()) for value in values):
                self.error("Device must be Auto, CPU, or cuda:N")
                return
            self.project.settings["inference"] = dict(
                mask_backend=backend.currentText(),
                mask_device=values[0],
                lama_device=values[1],
                expansion=expansion.value(),
                context=context.value(),
                offline=offline.isChecked(),
            )
            self.project.save_settings()

    def new_project(self):
        directory = QFileDialog.getExistingDirectory(self, "Select an empty project directory")
        if directory:
            self.start_project(directory, create=True)

    def open_project(self):
        directory = QFileDialog.getExistingDirectory(self, "Open project directory")
        if directory:
            self.start_project(directory)

    def start_project(self, directory, create=False):
        if not self.save():
            return
        try:
            project = Project(directory, create=create)
        except Exception as error:
            self.error(error)
            return
        try:
            project.lock()
        except ValueError as error:
            project.close()
            self.error(error)
            return
        if self.project:
            self.project.close()
        self.project = project
        self.op = None
        self.state = None
        self.dirty = False
        self.setWindowTitle(f"{project.directory.name} — Dataset Editor")
        self.spaces.blockSignals(True)
        self.escape.blockSignals(True)
        self.spaces.setChecked(project.settings["spaces"])
        self.escape.setChecked(project.settings["escape"])
        self.spaces.blockSignals(False)
        self.escape.blockSignals(False)
        self.settings()
        self.fill_method.setCurrentText(project.settings.get("fill_method", "Mean Color"))
        self.rebuild_items()

    def rebuild_items(self):
        self.generation += 1
        self.state = None
        self.fills.clear()
        self.im = None
        self.caption.blockSignals(True)
        self.caption.setPlainText("")
        self.caption.blockSignals(False)
        self.caption.setEnabled(False)
        self.canvas.pix.setPixmap(QPixmap())
        self.canvas.crop, self.canvas.fills = None, []
        self.canvas.draw()
        self.canvas.setEnabled(False)
        self.item_ids = list(self.project.items)
        self.paths = [self.project.source_path(key) for key in self.item_ids]
        self.image_position.setText(f"Image 0 / {len(self.paths):,}")
        self.model = Thumbnails(self.paths, self.pool)
        for i, state in enumerate(self.project.states()):
            self.model.status[i] = state["review_status"]
            if state["excluded"]:
                self.model.excluded[i] = self.exclusion_text(state)
            if not self.paths[i].is_file():
                self.model.issues[i] = "Missing source — use Relink Sources"
        self.thumbs.setModel(self.model)
        self.thumbs.selectionModel().currentChanged.connect(self.select)
        if self.paths:
            self.thumbs.setCurrentIndex(self.model.index(0))
        else:
            self.filename.setText("No imported images")
            self.status_message.setText("Project ready — use Import Images to add image/caption pairs")

    def project_job(self, title, function):
        progress = QProgressDialog(title, "", 0, 0, self)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        job = Job(lambda signals: function())

        def done(result):
            progress.close()
            self.rebuild_items()
            self.status_message.setText(str(result))

        def failed(error):
            progress.close()
            self.error(error)

        self.pool.start(job.connect(done, failed))

    def import_images(self):
        if not self.project:
            self.error("Create or open a project first")
            return
        if not self.save():
            return
        choice = QMessageBox(self)
        choice.setWindowTitle("Import Images")
        choice.setText("Import external images and copy their current captions into this project.")
        subfolders = QCheckBox("Include subfolders")
        choice.setCheckBox(subfolders)
        folder = choice.addButton("Folder…", QMessageBox.ButtonRole.AcceptRole)
        files = choice.addButton("Image files…", QMessageBox.ButtonRole.AcceptRole)
        choice.addButton(QMessageBox.StandardButton.Cancel)
        choice.exec()
        filters = []
        if choice.clickedButton() == folder:
            directory = QFileDialog.getExistingDirectory(self, "Import image folder")
            paths = [directory] if directory else []
            while paths:
                text, ok = QInputDialog.getMultiLineText(
                    self,
                    "Import filters",
                    "Optional rsync-style rules, one per line, first match wins:\n"
                    "- *_draft.png    exclude by name\n+ */    then + *.png    then - *    only PNG files\n"
                    "- /raw/    a directory at the top of the folder",
                )
                if not ok:
                    return
                try:
                    filters = [PathFilter.parse(line) for line in text.splitlines() if line.strip()]
                    break
                except ValueError as error:
                    self.error(error)
        elif choice.clickedButton() == files:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Import images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.PNG *.JPG *.JPEG *.WEBP *.BMP)"
            )
        else:
            return
        if paths:
            project = self.project
            recursive = subfolders.isChecked()

            def run():
                imported, skipped = project.import_images(paths, recursive=recursive, filters=filters)
                return f"Imported {imported} images; skipped {skipped} existing references"

            self.project_job("Importing images and captions…", run)

    def relink_sources(self):
        if not self.project or not self.save():
            return
        sources = self.project.sources
        labels = [f"{key.hex()[:8]} — {path}" for key, path in sources.items()]
        if not labels:
            self.error("No imported sources to relink")
            return
        label, ok = QInputDialog.getItem(self, "Relink Source", "Source folder", labels, 0, False)
        if not ok:
            return
        root = QFileDialog.getExistingDirectory(self, "Choose replacement source folder")
        if root:
            source_id = list(sources)[labels.index(label)]
            project = self.project

            def run():
                project.relink(source_id, root)
                return "Source folder relinked; every image found with its recorded file size"

            self.project_job("Verifying replacement images…", run)

    def navigate(self, delta):
        if self.paths:
            row = max(0, min(len(self.paths) - 1, self.thumbs.currentIndex().row() + delta))
            self.thumbs.setCurrentIndex(self.model.index(row))

    def select(self, index, previous):
        if not self.save():
            selection = self.thumbs.selectionModel()
            selection.blockSignals(True)
            self.thumbs.setCurrentIndex(previous)
            selection.blockSignals(False)
            return
        self.generation += 1
        generation = self.generation
        self.state = None
        self.op = None  # each visit to an image is one operation in the history
        self.im = None
        self.caption.setEnabled(False)
        self.canvas.setEnabled(False)
        self.image_position.setText(f"Image {index.row() + 1:,} / {len(self.paths):,}")
        path = self.paths[index.row()]
        self.filename.setText(path.name + " · Loading")
        state = self.project.state(self.item_ids[index.row()])
        self.loaded((None, state), generation)
        self.canvas.pix.setPixmap(QPixmap())
        self.canvas.crop, self.canvas.fills = None, []
        self.canvas.draw()
        self.canvas.setEnabled(False)

        def read(signals):
            warning = validate_source(path, state)
            image = load_image(path)
            check_image_size(image.size, state)
            return image, None, warning

        job = Job(read).connect(lambda result: self.loaded(result, generation), lambda error: self.load_error(error, generation))
        self.pool.start(job)

    def load_error(self, error, generation):
        if generation == self.generation:
            row = self.thumbs.currentIndex().row()
            self.model.issues[row] = error
            self.model.dataChanged.emit(self.model.index(row), self.model.index(row))
            self.filename.setText("Image not editable: " + error)
            self.status_message.setText("Source missing or changed — caption and review state remain editable")

    def loaded(self, result, generation):
        if generation != self.generation:
            return
        image, state, *warning = result
        self.im = image
        if state is None:
            self.canvas.setEnabled(True)
            self.filename.setText(f"{self.state['image']}\n{self.im.width} × {self.im.height}")
            if warning and warning[0]:
                self.status_message.setText(warning[0])
            self.set_ratio(self.ratio.currentText())
            self.refresh(True)
            return
        self.state = state
        self.fills.clear()
        self.dirty = False
        self.history, self.future = [], []
        self.caption.setEnabled(True)
        self.canvas.setEnabled(self.im is not None)
        self.caption.blockSignals(True)
        self.caption.setPlainText(self.state["caption"])
        self.caption.blockSignals(False)
        self.review.blockSignals(True)
        self.review.setCurrentText(self.state["review_status"])
        self.review.blockSignals(False)
        self.show_exclusion()
        self.filename.setText(self.state["image"])
        self.set_ratio(self.ratio.currentText())
        self.refresh(True)

    def save(self):
        self.saving.stop()
        if self.project and self.state and getattr(self, "dirty", False):
            try:
                if self.im is not None:
                    edited = self.state["crop"] or self.state["fills"]
                    self.state["image_size"] = [self.im.width, self.im.height] if edited else None
                if self.op is None:
                    self.op = self.project.begin_op("edit", self.state["image"])
                self.project.save(self.state, self.op)
                self.dirty = False
                self.status_message.setText("Edits saved")
            except Exception as error:
                self.error(error)
                return False
        return True

    def changed(self):
        self.dirty = True
        self.saving.start()
        row = self.thumbs.currentIndex().row()
        self.model.status[row] = self.state["review_status"]
        if self.state["excluded"]:
            self.model.excluded[row] = self.exclusion_text(self.state)
        else:
            self.model.excluded.pop(row, None)
        self.model.dataChanged.emit(self.model.index(row), self.model.index(row))
        self.status_message.setText("Unsaved changes…")

    def caption_changed(self):
        if self.state:
            self.state["caption"] = self.caption.toPlainText()
            self.changed()

    def review_changed(self, value):
        if self.state:
            self.state["review_status"] = value
            self.changed()

    def exclusion_text(self, state):
        by = state["excluded_by"]
        if by is None:
            return "Kept by hand" if (state["exclusion"] or {}).get("decision") == "keep" else "Included"
        if by["by"] == "human":
            return "Excluded by hand" + (f": {by['reason']}" if by["reason"] else "")
        kept = [self.project.items[bytes.fromhex(k)]["path"] for k in by["kept"] if bytes.fromhex(k) in self.project.items]
        if not kept:
            return f"Excluded: duplicate group {by['group']} discarded whole"
        return f"Excluded: duplicate of {', '.join(kept)} (group {by['group']})"

    def show_exclusion(self):
        exclusion = self.state["exclusion"] or {}
        self.exclusion.blockSignals(True)
        self.exclusion.setCurrentIndex({None: 0, "keep": 1, "exclude": 2}[exclusion.get("decision")])
        self.exclusion.blockSignals(False)
        self.exclusion_reason.blockSignals(True)
        self.exclusion_reason.setText(exclusion.get("reason", ""))
        self.exclusion_reason.blockSignals(False)
        self.exclusion_reason.setEnabled(bool(exclusion))
        self.exclusion_label.setText(self.exclusion_text(self.state))

    def exclusion_changed(self, value):
        if self.state:
            decision = {EXCLUSION_CHOICES[1]: "keep", EXCLUSION_CHOICES[2]: "exclude"}.get(value)
            reason = self.exclusion_reason.text().strip()
            self.state["exclusion"] = None if decision is None else {"decision": decision, "reason": reason}
            self.project.update_exclusion(self.state)
            self.show_exclusion()
            self.changed()

    def exclusion_reason_edited(self):
        if self.state and self.state["exclusion"] and self.exclusion_reason.text().strip() != self.state["exclusion"]["reason"]:
            self.state["exclusion"] = {**self.state["exclusion"], "reason": self.exclusion_reason.text().strip()}
            self.project.update_exclusion(self.state)
            self.show_exclusion()
            self.changed()

    def checkpoint(self):
        self.history.append(copy.deepcopy((self.state["crop"], self.state["fills"])))
        self.future.clear()

    def undo(self):
        if self.state and self.history:
            self.future.append(copy.deepcopy((self.state["crop"], self.state["fills"])))
            self.state["crop"], self.state["fills"] = self.history.pop()
            self.changed()
            self.refresh()

    def redo(self):
        if self.state and self.future:
            self.history.append(copy.deepcopy((self.state["crop"], self.state["fills"])))
            self.state["crop"], self.state["fills"] = self.future.pop()
            self.changed()
            self.refresh()

    def rectangle(self, rect):
        if not self.state or self.im is None:
            return
        if self.canvas.mode == "fill" and self.fill_method.currentText() == "LaMa":
            try:
                base = render(self.im, self.state, False, self.project.directory)
                dialog = MaskEditor(base, rect, self.project.settings.get("inference", {}), self.inference_engine, self)
                if dialog.exec() and dialog.result is not None:
                    metadata = {
                        "mask_generator": dialog.mask_metadata,
                        "inpainting": dialog.lama_metadata,
                        "threshold": dialog.threshold.value(),
                        "expansion": dialog.expansion.value(),
                        "context": dialog.settings["context"],
                        "box": dialog.box,
                        "points": dialog.points,
                        "point_labels": dialog.labels,
                    }
                    operation = save_repair(
                        self.project.directory, self.state["item_id"], base, rect, dialog.accepted_mask, dialog.result, metadata
                    )
                    self.checkpoint()
                    self.state["fills"].append(operation)
                    self.changed()
                    self.save()
            except Exception as error:
                self.error(error)
            self.refresh()
            return
        self.checkpoint()
        if self.canvas.mode == "crop":
            self.state["crop"] = rect
        else:
            self.state["fills"].append({"rect": rect, "ring_px": self.ring.value(), "method": "outer_ring_mean"})
        try:
            render(self.im, self.state, False, self.project.directory)
        except Exception as error:
            self.state["crop"], self.state["fills"] = self.history.pop()
            self.error(error)
        self.changed()
        self.refresh()

    def numeric_crop(self):
        if not self.state or self.im is None or self.canvas.mode != "crop":
            return
        x, y, w, h = [spin.value() for spin in self.coords]
        w, h = min(w, self.im.width - x), min(h, self.im.height - y)
        if min(x, y) >= 0 and min(w, h) > 0 and [x, y, w, h] != self.state["crop"]:
            self.rectangle([x, y, w, h])

    def clear_crop(self):
        if self.state:
            self.checkpoint()
            self.state["crop"] = None
            self.changed()
            self.refresh()

    def select_fill(self, row):
        self.canvas.selected_fill = row
        self.canvas.draw()

    def delete_fill(self):
        row = self.fills.currentRow()
        if self.state and row >= 0:
            self.checkpoint()
            self.state["fills"].pop(row)
            self.changed()
            self.refresh()

    def set_ratio(self, value):
        self.canvas.ratio = (
            (self.im.width / self.im.height if self.im else 0)
            if value == "Original"
            else (0 if value == "Free" else int(value.split(":")[0]) / int(value.split(":")[1]))
        )

    def mode(self, mode):
        if mode != "preview":
            self.edit_mode = mode
            self.tool_actions[mode].setChecked(True)
        self.preview_action.setChecked(mode == "preview")
        self.canvas.mode = mode
        self.refresh(True)
        self.status_message.setText(
            {
                "crop": "Crop: drag to create, resize corners, or move inside",
                "fill": "Fill: drag to add a rectangle",
                "preview": "Preview",
            }[mode]
        )

    def refresh(self, fit=False):
        if self.im is None or not self.state:
            return
        self.canvas.crop = copy.deepcopy(self.state["crop"])
        self.canvas.fills = copy.deepcopy(self.state["fills"])
        for spin, value in zip(self.coords, self.state["crop"] or [0, 0, self.im.width, self.im.height]):
            spin.setValue(value)
        selected = self.fills.currentItem()
        selected_fill = selected.data(Qt.ItemDataRole.UserRole) if selected else None
        self.fills.blockSignals(True)
        self.fills.clear()
        selected_row = -1
        for i, fill in enumerate(self.state["fills"]):
            description = "LaMa" if fill.get("method") == "lama" else f"ring {fill['ring_px']}px"
            self.fills.addItem(f"{i+1}: {fill['rect']} · {description}")
            self.fills.item(i).setData(Qt.ItemDataRole.UserRole, copy.deepcopy(fill))
            if selected_row == -1 and fill == selected_fill:
                selected_row = i
        self.fills.setCurrentRow(selected_row)
        self.fills.blockSignals(False)
        self.canvas.selected_fill = selected_row
        try:
            self.canvas.set_image(render(self.im, self.state, self.canvas.mode == "preview", self.project.directory), fit)
            self.filename.setText(f"{self.state['image']}\n{self.im.width} × {self.im.height}")
        except Exception as error:
            self.canvas.pix.setPixmap(QPixmap())
            self.canvas.draw()
            self.status_message.setText(str(error))
            self.filename.setText("Repair unavailable — remove stale/missing repair entries or Undo")

    def export(self):
        if not self.project or not self.save():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Export")
        form = QFormLayout(dialog)
        form.addRow("Output directory", QLabel(str(self.project.directory / "output")))
        fmt = QComboBox()
        fmt.addItems(["PNG", "WEBP"])
        fmt.setCurrentText(self.project.settings["format"])
        form.addRow("Format (lossless)", fmt)
        scope = QComboBox()
        scope.addItems(["All (except skip)", "Approved only", "Current image"])
        form.addRow("Scope", scope)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if not dialog.exec():
            return
        states = [state for state in self.project.states() if not state["excluded"]]
        if scope.currentIndex() == 2:
            states = [state for state in states if self.state and state["item_id"] == self.state["item_id"]]
        elif scope.currentIndex() == 1:
            states = [state for state in states if state["review_status"] == "approved"]
        else:
            states = [state for state in states if state["review_status"] != "skip"]
        if not states:
            self.error("No images match the selected export scope")
            return
        self.project.settings["format"] = fmt.currentText()
        self.project.save_settings()
        snapshot = self.project.export_snapshot(states, fmt.currentText())
        directory = self.project.directory
        progress = QProgressDialog("Building output snapshot…", "Cancel", 0, len(states), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        cancelled = [False]
        progress.canceled.connect(lambda: cancelled.__setitem__(0, True))

        def run(signals):
            return export_project(directory, snapshot, lambda: cancelled[0], signals.progress.emit)

        job = Job(run)
        job.signals.progress.connect(progress.setValue, Qt.ConnectionType.QueuedConnection)

        def done(result):
            progress.close()
            count, was_cancelled = result
            QMessageBox.information(
                self,
                "Export Results",
                "Cancelled — previous output preserved" if was_cancelled else f"Exported {count} images to {directory / 'output'}",
            )

        def failed(error):
            progress.close()
            self.error(f"Export failed; previous output preserved.\n{error}")

        self.pool.start(job.connect(done, failed))

    def closeEvent(self, event):
        if not self.save():
            event.ignore()
            return
        QApplication.instance().removeEventFilter(self)
        self.caption.dismiss()
        self.pool.waitForDone()
        if self.project:
            self.project.close()
        event.accept()
