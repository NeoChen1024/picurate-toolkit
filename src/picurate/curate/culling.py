"""The Culling tab: select images by similarity to example images, by text queries and by tag conditions, adjust by
hand, and exclude the selection as one operation of human exclusion decisions in ``project.sqlite``.

Everything is loaded when the tab is first shown: the project's embeddings (see :mod:`picurate.curate.scores`), what
excludes each image now, and the tag vocabulary. Scores are computed once per criterion; thresholds apply live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt, QThreadPool
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..common.siglip_text import IMAGE_TOWER
from ..dedupe.db_store import ModelMismatchError
from ..project.captions import TAGS
from ..project.project import EMBEDDINGS, Project
from ..tagger.tag_db import TagDB, TagDBMismatchError
from .canvas import ImageView
from .jobs import LRU, Job, load_image, qimage, thumbnail
from .scores import EmbeddingIndex

logger = logging.getLogger(__name__)

THUMB = (160, 160)
CELL = QSize(THUMB[0] + 12, THUMB[1] + 28)
# Starting thresholds, on the scale of each kind of score. Measured on SigLIP2 so400m (library images): a text query
# scores 0.10-0.18 on images that match it and 0.02-0.06 on the rest; two unrelated images score about 0.7 (99.9th
# percentile 0.91), since the stored embeddings share a large common component.
DEFAULT_THRESHOLDS = {"image": 0.90, "text": 0.12}
SLIDER_RANGES = {"image": (0.5, 1.0), "text": (-0.05, 0.30), "tag": (0.0, 1.0)}
BADGES = {"trash": ("T", "#b0b0b0"), "human": ("H", "#ef8888"), "dedupe": ("D", "#d8b878"), "keep": ("K", "#78d899")}
MANUAL_REASON = "culling: picked by hand"


@dataclass(eq=False)
class Criterion:
    """One example image, text query or tag condition. ``scores`` is None until computed; NaN means no score (no
    embedding, not tagged)."""

    kind: str  # image, text, tag
    label: str  # image path, query text, tag name
    threshold: float
    vector: np.ndarray | None = None
    tag: int | None = None
    floor: float = 0.0  # tags: scores below the database's floor are not stored
    scores: np.ndarray | None = field(default=None, repr=False)
    enabled: bool = True

    def describe(self) -> str:
        if self.kind == "text":
            return f'text "{self.label}" >= {self.threshold:.3f}'
        if self.kind == "image":
            return f"like {self.label} >= {self.threshold:.3f}"
        return f"tag {self.label} >= {self.threshold:.2f}"

    def matches(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return self.scores >= self.threshold


class QueryEncoders:
    """The models that turn query texts and external example images into embeddings, loaded on first use (on the GPU
    when there is one). Used from one worker thread."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.text = None
        self.image = None

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def encode_text(self, text: str) -> np.ndarray:
        if self.text is None:
            import torch

            from ..common.siglip_text import load_text_encoder

            cuda = self.device == "cuda"
            self.text = load_text_encoder(device=self.device, dtype=torch.float16 if cuda else torch.float32)
        return self.text.encode([text])[0].numpy()

    def encode_image(self, path: str) -> np.ndarray:
        if self.image is None:
            from ..dedupe.encoder import CLIPImageEncoder

            name, _, seq_len = self.model_id.partition("@")
            kwargs = {"seq_len": int(seq_len)} if seq_len else {}
            encoder = CLIPImageEncoder(name, self.device, **kwargs)
            if encoder.model_id != self.model_id:
                raise ModelMismatchError(f"The encoder identifies as {encoder.model_id}, the database as {self.model_id}")
            self.image = encoder
        return self.image.encode_pil_images([load_image(path)])[0]


class GridModel(QAbstractListModel):
    """Rows are item indices (``order``); the delegate paints them from the tab's state."""

    def __init__(self, tooltip):
        super().__init__()
        self.order = np.empty(0, dtype=np.int64)
        self.tooltip = tooltip

    def set_order(self, order: np.ndarray) -> None:
        self.beginResetModel()
        self.order = order
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802  # type: ignore[override]
        return 0 if parent.isValid() else len(self.order)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role == Qt.ItemDataRole.SizeHintRole:
            return CELL
        if role == Qt.ItemDataRole.UserRole and index.isValid():
            return int(self.order[index.row()])
        if role == Qt.ItemDataRole.ToolTipRole and index.isValid():
            return self.tooltip(int(self.order[index.row()]))
        return None


class GridDelegate(QStyledItemDelegate):
    def __init__(self, tab: CullingTab):
        super().__init__()
        self.tab = tab

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return CELL

    def paint(self, painter: QPainter, option, index) -> None:
        tab = self.tab
        i = index.data(Qt.ItemDataRole.UserRole)
        rect = option.rect.adjusted(2, 2, -2, -2)
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        box = QRect(rect.left() + 4, rect.top() + 4, THUMB[0], THUMB[1])
        pixmap = tab.thumbs.get(i)
        if pixmap is None:
            tab.request_thumbnail(i)
            painter.fillRect(box, QColor("#303030"))
        elif not pixmap.isNull():
            size = pixmap.size().scaled(box.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(0, 0, size.width(), size.height())
            target.moveCenter(box.center())
            painter.drawPixmap(target, pixmap)
        if tab.selection[i]:
            painter.setPen(QPen(QColor("#e04848"), 3))
            painter.drawRect(box.adjusted(1, 1, -1, -1))
        text = tab.score_text(i)
        if tab.manual_in[i]:
            text = "+ " + text
        elif tab.manual_out[i]:
            text = "− " + text
        painter.setPen(option.palette.color(option.palette.ColorRole.Text))
        painter.drawText(
            QRect(rect.left(), box.bottom() + 2, rect.width(), rect.bottom() - box.bottom() - 2), Qt.AlignmentFlag.AlignCenter, text
        )
        badge = tab.badge(i)
        if badge is not None:
            letter, colour = BADGES[badge]
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            r = QRect(box.right() - 20, box.top() + 2, 18, 18)
            painter.setPen(QPen(QColor("#253038"), 1))
            painter.setBrush(QColor(colour))
            painter.drawRoundedRect(r, 4, 4)
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, letter)
        painter.restore()


class CullingTab(QWidget):
    """Culling for one project. Writes human exclusion decisions, holding the project lock only while it does."""

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.loaded = False
        self.loading = False
        self.items: list[bytes] = []
        self.position: dict[bytes, int] = {}
        self.index: EmbeddingIndex | None = None
        self.encoders: QueryEncoders | None = None
        self.criteria: list[Criterion] = []
        self.tag_ids: dict[str, int] = {}
        self.tag_thresholds: list[float] = []
        self.tag_floor = 0.0
        self.tagged = np.zeros(0, dtype=bool)
        self._set_size(0)
        self.sorted = np.empty(0, dtype=np.int64)
        self.sort_key: Criterion | None = None
        self.last_op: int | None = None
        self.thumbs: LRU = LRU(2000)  # item index -> QPixmap
        self.thumb_jobs: dict[int, Job] = {}
        self.preview_job: Job | None = None

        self.pool = QThreadPool(self)  # thumbnails and the preview
        self.pool.setMaxThreadCount(4)
        self.compute_pool = QThreadPool(self)  # loading, scores, model inference: one at a time, in order
        self.compute_pool.setMaxThreadCount(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        splitter.addWidget(self._build_criteria())
        splitter.addWidget(self._build_grid())
        splitter.addWidget(self._build_preview())
        splitter.setSizes([340, 900, 460])
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 4)
        self.counts_label = QLabel()
        self.message_label = QLabel()
        row.addWidget(self.counts_label)
        row.addSpacing(12)
        row.addWidget(self.message_label, 1)
        layout.addLayout(row)
        for key, function in (
            ("Space", self.toggle_selected),
            ("T", self.threshold_at_current),
            ("E", self.add_current_example),
            ("F1", self.show_help),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(function)
        self._criterion_selected(-1)
        self.setEnabled(False)
        self.message("Not loaded yet")

    def _set_size(self, n: int) -> None:
        self.manual_in = np.zeros(n, dtype=bool)
        self.manual_out = np.zeros(n, dtype=bool)
        self.match = np.zeros(n, dtype=bool)
        self.selection = np.zeros(n, dtype=bool)
        self.excluded_by: list[dict | None] = [None] * n
        self.excluded = np.zeros(n, dtype=bool)
        self.decided = np.zeros(n, dtype=bool)  # a person decided (force keep or force exclude): Apply leaves it alone
        self.kept = np.zeros(n, dtype=bool)

    # -- construction ------------------------------------------------------------------------------------------------

    def _build_criteria(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(6, 6, 6, 6)
        add = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(["Text", "Tag"])
        self.kind.currentTextChanged.connect(self._kind_changed)
        self.query = QLineEdit()
        self.query.setPlaceholderText("a text query")
        self.query.returnPressed.connect(self.add_query)
        self.completer = QCompleter([])
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        add.addWidget(self.kind)
        add.addWidget(self.query, 1)
        box.addLayout(add)
        examples = QHBoxLayout()
        self.example_button = QPushButton("Current image as example (E)")
        self.example_button.clicked.connect(self.add_current_example)
        self.file_button = QPushButton("Example files…")
        self.file_button.clicked.connect(self.add_example_files)
        examples.addWidget(self.example_button)
        examples.addWidget(self.file_button)
        box.addLayout(examples)

        self.criteria_list = QListWidget()
        self.criteria_list.currentRowChanged.connect(self._criterion_selected)
        self.criteria_list.itemChanged.connect(self._criterion_toggled)
        box.addWidget(self.criteria_list, 1)

        threshold = QHBoxLayout()
        threshold.addWidget(QLabel("Threshold"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.005)
        self.spin.setRange(-1.0, 1.0)
        self.spin.valueChanged.connect(self._threshold_changed)
        threshold.addWidget(self.slider, 1)
        threshold.addWidget(self.spin)
        box.addLayout(threshold)

        combine = QHBoxLayout()
        combine.addWidget(QLabel("Select images matching"))
        self.combine = QComboBox()
        self.combine.addItems(["any criterion", "all criteria"])
        self.combine.currentIndexChanged.connect(lambda _: self.recompute())
        combine.addWidget(self.combine, 1)
        box.addLayout(combine)

        buttons = QHBoxLayout()
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_criterion)
        clear = QPushButton("Clear all")
        clear.clicked.connect(self.clear)
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        box.addLayout(buttons)
        apply = QHBoxLayout()
        self.apply_button = QPushButton("Exclude selection…")
        self.apply_button.clicked.connect(self.apply)
        self.undo_button = QPushButton("Undo exclusion")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo)
        apply.addWidget(self.apply_button)
        apply.addWidget(self.undo_button)
        box.addLayout(apply)
        return panel

    def _build_grid(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 6, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Show"))
        self.show_mode = QComboBox()
        self.show_mode.addItems(["selection", "all images", "left out by hand"])
        self.show_mode.currentIndexChanged.connect(lambda _: self.refresh_view())
        bar.addWidget(self.show_mode)
        self.hide_excluded = QCheckBox("Hide excluded")
        self.hide_excluded.toggled.connect(lambda _: self.refresh_view())
        bar.addWidget(self.hide_excluded)
        bar.addStretch(1)
        self.sort_label = QLabel()
        bar.addWidget(self.sort_label)
        box.addLayout(bar)
        self.model = GridModel(self.tooltip)
        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setItemDelegate(GridDelegate(self))
        self.view.setViewMode(QListView.ViewMode.ListMode)
        self.view.setFlow(QListView.Flow.LeftToRight)
        self.view.setWrapping(True)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setUniformItemSizes(True)
        self.view.setLayoutMode(QListView.LayoutMode.Batched)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.selectionModel().currentChanged.connect(lambda current, _: self._current_changed(current))
        self.view.verticalScrollBar().valueChanged.connect(lambda _: self._cancel_thumbnails())
        box.addWidget(self.view, 1)
        return panel

    def _build_preview(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 6, 6, 0)
        self.preview = ImageView()
        box.addWidget(self.preview, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.info)
        return panel

    # -- loading -----------------------------------------------------------------------------------------------------

    def activate(self) -> None:
        """Called whenever the tab is shown: loads everything the first time, then refreshes what excludes each
        image (the Duplicates tab and the editor change it)."""
        if self.loaded:
            self.reload_exclusions()
        elif not self.loading:
            self.loading = True
            self.message("Loading the project's embeddings and tags…")
            project = self.project
            items = list(project.items)

            def load(signals):
                result = {"items": items, "states": project.states(captions=False), "index": None, "notes": []}
                path = project.directory / EMBEDDINGS
                if path.exists():
                    try:
                        result["index"] = EmbeddingIndex.load(path, items)
                    except ModelMismatchError as error:
                        result["notes"].append(f"{error}: no similarity search")
                else:
                    result["notes"].append("no embeddings yet (picurate-dedupe update-db): no similarity search")
                path = project.directory / TAGS
                result["tags"] = None
                if path.exists():
                    try:
                        db = TagDB(str(path), readonly=True)
                    except TagDBMismatchError as error:
                        result["notes"].append(f"{error}: no tag conditions")
                    else:
                        try:
                            if db.get_meta("model_id") is not None:
                                vocab = db.vocabulary()
                                thresholds = [0.35] * len(vocab.tags)
                                for category, part in vocab.slices.items():
                                    for t in range(*part.indices(len(vocab.tags))):
                                        thresholds[t] = vocab.thresholds.get(category, 0.35)
                                result["tags"] = (list(vocab.tags), thresholds, db.score_floor or 0.0, set(db.load_index()))
                        finally:
                            db.conn.close()
                else:
                    result["notes"].append("no tag scores yet (picurate-tag update): no tag conditions")
                return result

            self.compute_pool.start(Job(load).connect(self._loaded, self._load_failed))

    def _load_failed(self, error: str) -> None:
        self.loading = False
        self.message(f"Loading failed: {error}")

    def _loaded(self, result: dict) -> None:
        self.loading = False
        self.loaded = True
        self.items = result["items"]
        self.position = {item: i for i, item in enumerate(self.items)}
        self._set_size(len(self.items))
        self._set_states(result["states"])
        self.index = result["index"]
        if self.index is not None:
            self.encoders = QueryEncoders(self.index.model_id)
        if result["tags"] is not None:
            names, self.tag_thresholds, self.tag_floor, tagged = result["tags"]
            self.tag_ids = {name: t for t, name in enumerate(names)}
            self.tagged = np.fromiter((item in tagged for item in self.items), dtype=bool, count=len(self.items))
            self.completer.setModel(None)
            self.completer = QCompleter(names, self)
            self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.completer.setMaxVisibleItems(15)
        self.kind.clear()
        kinds = (["Text"] if self.text_available() else []) + (["Tag"] if self.tag_ids else [])
        self.kind.addItems(kinds)
        self.query.setEnabled(bool(kinds))
        self.example_button.setEnabled(self.index is not None)
        self.file_button.setEnabled(self.index is not None)
        self.setEnabled(True)
        self.sorted = np.arange(len(self.items))
        self.recompute()
        notes = list(result["notes"])
        if self.index is not None and not self.text_available():
            notes.append(f"text queries need {IMAGE_TOWER} embeddings, these are {self.index.model_id}")
        if self.index is not None:
            missing = int(np.isnan(self.index.norms).sum())
            if missing:
                notes.append(f"{missing} images have no embedding")
        self.message(f"{len(self.items)} images" + "".join(f" · {note}" for note in notes))

    def text_available(self) -> bool:
        return self.index is not None and self.index.model_id.partition("@")[0] == IMAGE_TOWER

    def _set_states(self, states: list[dict]) -> None:
        for state in states:
            i = self.position.get(state["item_id"])
            if i is None:
                continue
            self.excluded_by[i] = state["excluded_by"]
            self.excluded[i] = state["excluded"]
            decision = (state["exclusion"] or {}).get("decision")
            self.decided[i] = decision is not None
            self.kept[i] = decision == "keep"

    def reload_exclusions(self) -> None:
        project = self.project
        self.compute_pool.start(
            Job(lambda signals: project.states(captions=False)).connect(
                lambda states: (self._set_states(states), self.recompute()),
                lambda error: self.message(f"Cannot read the project: {error}"),
            )
        )

    # -- criteria ----------------------------------------------------------------------------------------------------

    def _kind_changed(self, kind: str) -> None:
        if kind == "Tag":
            self.query.setPlaceholderText("tag, or tag >= probability")
            self.query.setCompleter(self.completer)
        else:
            self.query.setPlaceholderText("a text query")
            self.query.setCompleter(None)

    def add_query(self) -> None:
        text = self.query.text().strip()
        if not text:
            return
        if self.kind.currentText() == "Tag":
            name, _, value = text.partition(">=")
            name = name.strip().lower().replace(" ", "_")
            tag = self.tag_ids.get(name)
            if tag is None:
                self.message(f"No tag {name!r} in the tagger's vocabulary")
                return
            try:
                threshold = float(value) if value.strip() else self.tag_thresholds[tag]
            except ValueError:
                self.message(f"Not a probability: {value.strip()!r}")
                return
            criterion = Criterion("tag", name, max(threshold, self.tag_floor), tag=tag, floor=self.tag_floor)
            self.query.clear()
            self._add(criterion)
            tagged, position = self.tagged, self.position
            path = self.project.directory / TAGS

            def compute(signals):
                db = TagDB(str(path), readonly=True)
                try:
                    rows = db.tag_scores(tag)
                finally:
                    db.conn.close()
                scores = np.where(tagged, np.float32(0), np.float32(np.nan))
                for item, prob in rows:
                    i = position.get(item)
                    if i is not None:
                        scores[i] = prob
                return scores

            self._compute(criterion, compute)
            return
        criterion = Criterion("text", text, DEFAULT_THRESHOLDS["text"])
        self.query.clear()
        self._add(criterion)
        encoders, index = self.encoders, self.index
        assert encoders is not None and index is not None
        self.message("Encoding the query (the text model loads on first use)…")

        def compute(signals):
            criterion.vector = encoders.encode_text(text)
            return index.cosine(criterion.vector[None])[:, 0]

        self._compute(criterion, compute)

    def add_current_example(self) -> None:
        i = self.current_item()
        if i is None or self.index is None:
            return
        vector = self.index.vector(i)
        if vector is None:
            self.message("This image has no embedding")
            return
        criterion = Criterion("image", self.items_path(i), DEFAULT_THRESHOLDS["image"], vector=vector)
        self._add(criterion)
        index = self.index
        self._compute(criterion, lambda signals: index.cosine(vector[None])[:, 0])

    def add_example_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Example images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif)")
        encoders, index = self.encoders, self.index
        if not paths or encoders is None or index is None:
            return
        self.message("Encoding the examples (the image model loads on first use)…")
        for path in paths:
            criterion = Criterion("image", path, DEFAULT_THRESHOLDS["image"])
            self._add(criterion)

            def compute(signals, criterion=criterion, path=path):
                criterion.vector = encoders.encode_image(path)
                return index.cosine(criterion.vector[None])[:, 0]

            self._compute(criterion, compute)

    def _add(self, criterion: Criterion) -> None:
        self.criteria.append(criterion)
        item = QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.criteria_list.blockSignals(True)
        self.criteria_list.addItem(item)
        self.criteria_list.blockSignals(False)
        self.criteria_list.setCurrentRow(len(self.criteria) - 1)
        self._update_labels()

    def _compute(self, criterion: Criterion, function) -> None:
        def done(scores) -> None:
            criterion.scores = scores
            if criterion in self.criteria:
                self.message(f"Scored {criterion.describe()}")
                if criterion is self.current_criterion():
                    self._resort()
                self.recompute()

        def failed(error: str) -> None:
            self.message(f"{criterion.describe()}: {error}")
            if criterion in self.criteria:
                self._remove(self.criteria.index(criterion))

        self.compute_pool.start(Job(function).connect(done, failed))

    def current_criterion(self) -> Criterion | None:
        row = self.criteria_list.currentRow()
        return self.criteria[row] if 0 <= row < len(self.criteria) else None

    def _criterion_selected(self, row: int) -> None:
        criterion = self.current_criterion()
        for widget in (self.slider, self.spin):
            widget.setEnabled(criterion is not None)
        if criterion is not None:
            self.spin.blockSignals(True)
            self.spin.setValue(criterion.threshold)
            self.spin.blockSignals(False)
            self._sync_slider()
        self._resort()

    def _criterion_toggled(self, item: QListWidgetItem) -> None:
        row = self.criteria_list.row(item)
        if 0 <= row < len(self.criteria):
            self.criteria[row].enabled = item.checkState() == Qt.CheckState.Checked
            self.recompute()

    def remove_criterion(self) -> None:
        row = self.criteria_list.currentRow()
        if 0 <= row < len(self.criteria):
            self._remove(row)

    def _remove(self, row: int) -> None:
        del self.criteria[row]
        self.criteria_list.blockSignals(True)
        self.criteria_list.takeItem(row)
        self.criteria_list.blockSignals(False)
        self._criterion_selected(self.criteria_list.currentRow())
        self.recompute()

    def clear(self) -> None:
        if self.manual_in.any() or self.manual_out.any() or self.criteria:
            if (
                QMessageBox.question(self, "Clear", "Remove every criterion and every choice made by hand?")
                != QMessageBox.StandardButton.Yes
            ):
                return
        self.criteria.clear()
        self.criteria_list.clear()
        self.manual_in[:] = False
        self.manual_out[:] = False
        self._criterion_selected(-1)
        self.recompute()

    # -- thresholds --------------------------------------------------------------------------------------------------

    def _sync_slider(self) -> None:
        criterion = self.current_criterion()
        if criterion is None:
            return
        low, high = SLIDER_RANGES[criterion.kind]
        self.slider.blockSignals(True)
        self.slider.setValue(round((min(max(criterion.threshold, low), high) - low) / (high - low) * 1000))
        self.slider.blockSignals(False)

    def _slider_moved(self, value: int) -> None:
        criterion = self.current_criterion()
        if criterion is not None:
            low, high = SLIDER_RANGES[criterion.kind]
            self.spin.setValue(low + (high - low) * value / 1000)

    def _threshold_changed(self, value: float) -> None:
        criterion = self.current_criterion()
        if criterion is None:
            return
        criterion.threshold = max(value, criterion.floor) if criterion.kind == "tag" else value
        self._sync_slider()
        self.recompute()

    def threshold_at_current(self) -> None:
        """Move the threshold to the current image's score, so it is the last one that matches."""
        criterion, i = self.current_criterion(), self.current_item()
        if criterion is None or i is None or criterion.scores is None or np.isnan(criterion.scores[i]):
            return
        self.spin.setValue(float(np.floor(criterion.scores[i] * 1000) / 1000))

    # -- selection ---------------------------------------------------------------------------------------------------

    def recompute(self) -> None:
        active = [c for c in self.criteria if c.enabled and c.scores is not None]
        if active:
            combine = np.logical_or if self.combine.currentIndex() == 0 else np.logical_and
            self.match = combine.reduce([c.matches() for c in active])
        else:
            self.match = np.zeros(len(self.items), dtype=bool)
        self.selection = (self.match | self.manual_in) & ~self.manual_out
        self._update_labels()
        self.refresh_view()

    def _resort(self) -> None:
        criterion = self.current_criterion()
        if criterion is not None and criterion.scores is not None:
            key = np.where(np.isnan(criterion.scores), np.inf, -criterion.scores)
            self.sorted = np.argsort(key, kind="stable")
            self.sort_label.setText(f"sorted by {criterion.describe().rpartition(' >=')[0]}")
        else:
            self.sorted = np.arange(len(self.items))
            self.sort_label.setText("in project order")
        self.sort_key = criterion
        self.refresh_view(force=True)

    def refresh_view(self, force: bool = False) -> None:
        mode = self.show_mode.currentIndex()
        order = self.sorted
        if mode == 0:
            order = order[self.selection[order]]
        elif mode == 2:
            order = order[self.manual_out[order]]
        if self.hide_excluded.isChecked():
            order = order[~self.excluded[order]]
        if not force and np.array_equal(order, self.model.order):
            self.view.viewport().update()
            return
        current = self.current_item()
        self.model.set_order(order)
        if current is not None:
            rows = np.flatnonzero(order == current)
            if len(rows):
                index = self.model.index(int(rows[0]))
                self.view.setCurrentIndex(index)
                self.view.scrollTo(index)

    def _update_labels(self) -> None:
        self.criteria_list.blockSignals(True)
        for row, criterion in enumerate(self.criteria):
            count = "…" if criterion.scores is None else str(int(np.count_nonzero(criterion.matches())))
            self.criteria_list.item(row).setText(f"{criterion.describe()} · {count}")
        self.criteria_list.blockSignals(False)
        selected = int(self.selection.sum())
        writes = int((self.selection & ~self.decided).sum())
        self.counts_label.setText(
            f"Selected {selected} (by hand +{int(self.manual_in.sum())} −{int(self.manual_out.sum())})"
            f" · {int((self.selection & self.excluded).sum())} of them already excluded"
        )
        self.apply_button.setText(f"Exclude {writes} images…")
        self.apply_button.setEnabled(writes > 0)

    def toggle_selected(self) -> None:
        """Space: take the selected cells out of the selection, or put them in, by hand."""
        rows = sorted({index.row() for index in self.view.selectionModel().selectedIndexes()})
        if not rows:
            i = self.current_item()
            if i is None:
                return
            items = np.array([i])
        else:
            items = self.model.order[rows]
        include = not self.selection[items].all()
        for i in items:
            self.manual_in[i] = include and not self.match[i]
            self.manual_out[i] = not include and self.match[i]
        self.selection = (self.match | self.manual_in) & ~self.manual_out
        self._update_labels()
        self.view.viewport().update()
        self._show_info()

    # -- current image -----------------------------------------------------------------------------------------------

    def current_item(self) -> int | None:
        index = self.view.currentIndex()
        return index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None

    def items_path(self, i: int) -> str:
        return self.project.items[self.items[i]]["path"]

    def _current_changed(self, index) -> None:
        self._show_info()
        if not index.isValid():
            self.preview.set_image(None)
            return
        i = index.data(Qt.ItemDataRole.UserRole)
        path = str(self.project.source_path(self.items[i]))
        if self.preview_job is not None:
            self.preview_job.cancel()
        self.preview_job = Job(lambda signals: (i, qimage(load_image(path))))
        self.preview_job.connect(self._preview_ready, lambda error: self._preview_failed(i, error))
        self.pool.start(self.preview_job)

    def _preview_ready(self, result) -> None:
        i, image = result
        if i == self.current_item():
            self.preview.set_image(image)
            self.preview.fit()

    def _preview_failed(self, i: int, error: str) -> None:
        if i == self.current_item():  # a stale job's failure must not blank the newer preview
            self.preview.set_image(None)
            self.message(f"Cannot read {self.items_path(i)}: {error}")

    def _show_info(self) -> None:
        i = self.current_item()
        if i is None:
            self.info.clear()
            return
        lines = [self.items_path(i)]
        for criterion in self.criteria:
            if criterion.scores is not None:
                score = criterion.scores[i]
                lines.append(f"{'✓' if criterion.matches()[i] else '·'} {criterion.describe()}: {score:.3f}")
        lines.append(self.exclusion_text(i))
        if self.manual_in[i]:
            lines.append("Put into the selection by hand")
        elif self.manual_out[i]:
            lines.append("Left out of the selection by hand")
        self.info.setText("\n".join(lines))

    def exclusion_text(self, i: int) -> str:
        by = self.excluded_by[i]
        if by is None:
            return "Kept by hand: Exclude leaves it alone" if self.kept[i] else "Included"
        if by["by"] == "trash":
            return "Excluded: in the trash"
        if by["by"] == "human":
            return "Excluded by hand" + (f": {by['reason']}" if by.get("reason") else "")
        return f"Excluded as a duplicate (group {by['group']})"

    def badge(self, i: int) -> str | None:
        by = self.excluded_by[i]
        return by["by"] if by is not None else "keep" if self.kept[i] else None

    def score_text(self, i: int) -> str:
        criterion = self.sort_key
        if criterion is None or criterion.scores is None:
            return ""
        score = criterion.scores[i]
        return "—" if np.isnan(score) else f"{score:.3f}"

    def tooltip(self, i: int) -> str:
        return f"{self.items_path(i)}\n{self.exclusion_text(i)}"

    # -- thumbnails --------------------------------------------------------------------------------------------------

    def request_thumbnail(self, i: int) -> None:
        if i in self.thumb_jobs:
            return
        path = str(self.project.source_path(self.items[i]))
        job = Job(lambda signals: (i, thumbnail(path, THUMB)))
        job.connect(self._thumbnail_ready, lambda error: self._thumbnail_ready((i, None)))
        self.thumb_jobs[i] = job
        self.pool.start(job)

    def _thumbnail_ready(self, result) -> None:
        i, image = result
        self.thumb_jobs.pop(i, None)
        self.thumbs.put(i, QPixmap() if image is None else QPixmap.fromImage(image))
        self.view.viewport().update()

    def _cancel_thumbnails(self) -> None:
        """Scrolling: drop the requests of cells no longer shown; the visible ones ask again when painted."""
        for i, job in list(self.thumb_jobs.items()):
            if job.cancel():
                del self.thumb_jobs[i]

    # -- applying ----------------------------------------------------------------------------------------------------

    def query_description(self) -> str:
        active = [c for c in self.criteria if c.enabled and c.scores is not None]
        if not active:
            return MANUAL_REASON
        joined = "; ".join(c.describe() for c in active)
        if len(active) == 1:
            return f"culling: {joined}"
        return f"culling: {'any' if self.combine.currentIndex() == 0 else 'all'} of {joined}"

    def apply(self) -> None:
        targets = np.flatnonzero(self.selection & ~self.decided)
        if not len(targets):
            return
        query = self.query_description()
        reasons = {self.items[i]: (query if self.match[i] else MANUAL_REASON) for i in targets}
        skipped = int((self.selection & self.decided).sum())
        text = f"Exclude {len(targets)} images with the reason\n\n{query}\n\nas one operation?"
        if skipped:
            text += f"\n\n{skipped} selected images already have a decision by hand and stay as they are."
        if QMessageBox.question(self, "Exclude", text) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.project.lock("project")
        except ValueError as error:
            QMessageBox.warning(self, "Exclude", f"{error}: close it first.")
            return
        try:
            op, count = self.project.exclude(reasons, "culling", query.removeprefix("culling: "))
        except Exception as error:  # noqa: BLE001 - reported to the user
            QMessageBox.critical(self, "Exclude", f"Nothing was excluded: {error}")
            return
        finally:
            self.project.unlock("project")
        self.manual_in[:] = False
        self.manual_out[:] = False
        # The project decides what is still undecided: what this tab knew may be behind the editor or another process.
        changed = (
            f"; {len(targets) - count} were decided by hand since this tab loaded and stay as they are"
            if count < len(targets)
            else ""
        )
        if op is None:
            self.message(f"Nothing excluded: every selected image was decided by hand since this tab loaded")
        else:
            self.last_op = op
            self.undo_button.setEnabled(True)
            self.message(f"Excluded {count} images as operation {op} (picurate-project rollback {op} undoes it too){changed}")
        self.reload_exclusions()

    def undo(self) -> None:
        if self.last_op is None:
            return
        try:
            self.project.lock("project")
        except ValueError as error:
            QMessageBox.warning(self, "Undo", f"{error}: close it first.")
            return
        try:
            _, skipped, restored, _ = self.project.rollback(self.last_op)
        except Exception as error:  # noqa: BLE001 - reported to the user
            QMessageBox.critical(self, "Undo", str(error))
            return
        finally:
            self.project.unlock("project")
        self.message(
            f"Rolled back operation {self.last_op}: {restored} exclusions removed"
            + (f", {len(skipped)} changed since and left alone" if skipped else "")
        )
        self.last_op = None
        self.undo_button.setEnabled(False)
        self.reload_exclusions()

    # -- misc --------------------------------------------------------------------------------------------------------

    def message(self, text: str) -> None:
        self.message_label.setText(text)
        logger.info("%s", text)

    def show_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Culling")
        dialog.resize(640, 600)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setHtml(HELP_HTML)
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.show()

    def shutdown(self) -> None:
        """Stop background work. A running inference cannot be cancelled: it finishes first, then the models and the
        embeddings are released."""
        self._cancel_thumbnails()
        if self.preview_job is not None:
            self.preview_job.cancel()
        self.pool.waitForDone()
        self.compute_pool.waitForDone()
        self.encoders = None
        self.index = None


HELP_HTML = """
<h2>Culling</h2>
<p>Add criteria on the left: a <b>text query</b> (scored by SigLIP2's text tower), a <b>tag condition</b>
(<code>tag</code> or <code>tag &gt;= probability</code>, from the project's tag scores; without a probability the
tagger's threshold for the tag's category) or an <b>example image</b> (the current image, or files from anywhere).
Each criterion has its own threshold; the selection is the images matching any (or all) of the ticked criteria,
changed by hand.</p>
<p>The grid is sorted by the score of the criterion selected in the list. A red frame marks the selection; + and −
mark images put in or left out by hand. Badges say what excludes an image already: <b>T</b> trash, <b>H</b> by hand,
<b>D</b> a duplicate group; <b>K</b> marks an image kept by hand.</p>
<p><b>Exclude</b> writes a force exclude decision for every selected image no person decided about yet, with the
query as the reason, as one operation of the project's history. Undo, or <code>picurate-project rollback</code>,
takes it back. Criteria and choices by hand are not saved.</p>
<table cellspacing="6">
<tr><td><b>Space</b></td><td>Put the selected cells into the selection, or take them out</td></tr>
<tr><td><b>T</b></td><td>Set the threshold to the current image's score</td></tr>
<tr><td><b>E</b></td><td>Add the current image as an example</td></tr>
<tr><td><b>F1</b></td><td>This help</td></tr>
</table>
<p>Scores of example images and text queries are cosine similarities. A text query scores about 0.10-0.18 on images
that match it and 0.02-0.06 on others.</p>
"""
