"""Qt helpers shared by the GUIs. Imported only by GUI code: the command-line tools never import Qt."""

from __future__ import annotations

import threading
from collections.abc import Callable

import PIL.Image
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QImage


class Signals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class Job(QRunnable):
    """Run ``function(signals)`` in the pool; its return value is emitted on ``signals.result``.

    ``connect`` wires both outcomes to the GUI thread with queued connections and keeps the job alive until the
    result has been delivered: a lambda slot has no receiver object, so without this Qt would run it on the worker
    thread, and an auto-deleted runnable could take its ``Signals`` down before the queued event arrives.
    """

    _alive: set[Job] = set()

    def __init__(self, function: Callable[[Signals], object]):
        super().__init__()
        self.function = function
        self.signals = Signals()
        self.cancelled = False
        self.started = False
        self._state = threading.Lock()  # cancel() and run() race on the two flags from different threads

    def cancel(self) -> bool:
        """Make the job a no-op if it has not started yet; returns whether it was still waiting. (Qt deletes a
        runnable as soon as it has run, so taking it back out of the pool is not safe once it may have started; a
        flag is.) A job that already started finishes and delivers its result as usual; the answer is exact, so the
        caller can forget a job that returned True and keep waiting for one that returned False."""
        with self._state:
            self.cancelled = True
            return not self.started

    def connect(self, result: Callable[[object], object], error: Callable[[str], object]) -> Job:
        self.signals.result.connect(lambda r: (Job._alive.discard(self), result(r)), Qt.ConnectionType.QueuedConnection)
        self.signals.error.connect(lambda e: (Job._alive.discard(self), error(e)), Qt.ConnectionType.QueuedConnection)
        Job._alive.add(self)
        return self

    def run(self) -> None:
        with self._state:
            if self.cancelled:
                Job._alive.discard(self)
                return
            self.started = True
        try:
            self.signals.result.emit(self.function(self.signals))
        except Exception as error:  # noqa: BLE001 - reported to the GUI
            self.signals.error.emit(str(error))


def qimage(im: PIL.Image.Image) -> QImage:
    rgb = im.convert("RGB")
    return QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
