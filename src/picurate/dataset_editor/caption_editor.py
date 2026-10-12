"""Caption text box with tag completion and duplicate-entry highlighting. Manual text is never auto-corrected."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QPlainTextEdit, QTextEdit

from .tags import CATEGORIES, normalize


class CaptionEditor(QPlainTextEdit):
    exit_requested = Signal()

    def __init__(self):
        super().__init__()
        self.index = None
        self.spaces = True
        self.escape = False
        self.popup = QListWidget(self)
        self.popup.setWindowFlags(Qt.WindowType.ToolTip)
        self.popup.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.popup.itemClicked.connect(self.accept)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.complete)
        self.textChanged.connect(self.schedule)
        self.cursorPositionChanged.connect(self.popup.hide)
        self.cursorPositionChanged.connect(self.highlight_duplicates)
        self.selectionChanged.connect(self.highlight_duplicates)
        self.textChanged.connect(self.highlight_duplicates)
        self.matches = []

    def highlight_duplicates(self):
        """Highlight matching whole comma/newline-delimited entries, not substrings."""
        text = self.toPlainText()
        cursor = self.textCursor()
        spans = []
        for match in re.finditer(r"[^,\n]+", text):
            raw = match[0]
            if not raw.strip():
                continue
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.end() - len(raw) + len(raw.rstrip())
            # Qt uses UTF-16 positions, including for astral characters.
            a = len(text[:start].encode("utf-16-le")) // 2
            b = len(text[:end].encode("utf-16-le")) // 2
            key = " ".join(normalize(raw).split())
            spans.append((a, b, key))
        active = next(
            (span for span in spans if (span[0] <= cursor.selectionStart() < span[1] and cursor.selectionEnd() <= span[1])), None
        )
        if active is None and not cursor.hasSelection():
            active = next((span for span in spans if span[1] == cursor.position()), None)
        selections = []
        if active:
            for start, end, key in spans:
                if key != active[2] or (start, end) == active[:2]:
                    continue
                selection = QTextEdit.ExtraSelection()
                selection.cursor = QTextCursor(self.document())
                selection.cursor.setPosition(start)
                selection.cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                selection.format.setBackground(QColor("#ffe066"))
                selection.format.setForeground(QColor("#202020"))
                selections.append(selection)
        self.setExtraSelections(selections)

    def schedule(self):
        self.timer.start()

    def dismiss(self):
        self.timer.stop()
        self.popup.hide()

    def formatted(self, tag):
        if self.spaces:
            tag = tag.replace("_", " ")
        if self.escape:
            tag = re.sub(r"(?<!\\)([()])", r"\\\1", tag)
        return tag

    def complete(self):
        if self.index is None or not self.hasFocus():
            self.popup.hide()
            return
        cursor = self.textCursor()
        # Qt cursor positions are UTF-16; select text using Qt instead of slicing with that offset.
        before = QTextCursor(cursor)
        before.setPosition(0, QTextCursor.MoveMode.KeepAnchor)
        text = before.selectedText().replace("\u2029", "\n")
        start = max(text.rfind(","), text.rfind("\n")) + 1
        segment = text[start:]
        self.matches = self.index.search(segment)
        self.popup.clear()
        self.end = cursor.position()
        for _, offset, name, kind, count in self.matches:
            replacement = self.formatted(name)
            preserved = segment[:offset].strip()
            label = f"{replacement}    {CATEGORIES.get(kind, str(kind))}    {count:,}"
            if preserved:
                label += f"\nKeep: {preserved}"
            item = QListWidgetItem(label)
            span = segment[offset:]
            item.setData(Qt.ItemDataRole.UserRole, (self.end - len(span.encode("utf-16-le")) // 2, replacement))
            self.popup.addItem(item)
        if not self.matches:
            self.popup.hide()
            return
        self.popup.setCurrentRow(0)
        self.popup.resize(max(420, self.width()), min(300, self.popup.sizeHintForRow(0) * len(self.matches) + 8))
        self.popup.move(self.mapToGlobal(self.cursorRect().bottomLeft()))
        self.popup.show()

    def accept(self, item=None):
        item = item or self.popup.currentItem()
        if item is None:
            return
        start, replacement = item.data(Qt.ItemDataRole.UserRole)
        cursor = self.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(self.end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(replacement)
        self.setTextCursor(cursor)
        self.dismiss()

    def keyPressEvent(self, event):
        key = event.key()
        ctrl = event.modifiers() == Qt.KeyboardModifier.ControlModifier
        if ctrl and key == Qt.Key.Key_R:
            self.complete()
            return
        if ctrl and key in (Qt.Key.Key_W, Qt.Key.Key_U, Qt.Key.Key_K):
            cursor = self.textCursor()
            if not cursor.hasSelection():
                if key == Qt.Key.Key_W:
                    before = QTextCursor(cursor)
                    before.setPosition(0, QTextCursor.MoveMode.KeepAnchor)
                    text = before.selectedText().replace("\u2029", "\n")
                    match = re.search(r"[^\s,]+[\s,]*$|[\s,]+$", text)
                    if match:
                        cursor.setPosition(
                            cursor.position() - len(match[0].encode("utf-16-le")) // 2, QTextCursor.MoveMode.KeepAnchor
                        )
                else:
                    cursor.movePosition(
                        QTextCursor.MoveOperation.StartOfBlock if key == Qt.Key.Key_U else QTextCursor.MoveOperation.EndOfBlock,
                        QTextCursor.MoveMode.KeepAnchor,
                    )
            cursor.removeSelectedText()
            self.setTextCursor(cursor)
            self.dismiss()
            return
        if self.popup.isVisible():
            if key == Qt.Key.Key_Tab:
                self.accept()
                return
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                self.popup.setCurrentRow((self.popup.currentRow() + (1 if key == Qt.Key.Key_Down else -1)) % self.popup.count())
                return
        if key == Qt.Key.Key_Escape:
            was_visible = self.popup.isVisible()
            self.dismiss()
            if not was_visible and not event.isAutoRepeat():
                self.exit_requested.emit()
            return
        super().keyPressEvent(event)
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Comma):
            self.dismiss()

    def inputMethodEvent(self, event):
        self.dismiss()
        super().inputMethodEvent(event)
        if event.preeditString():
            self.dismiss()

    def focusOutEvent(self, event):
        self.dismiss()
        super().focusOutEvent(event)
