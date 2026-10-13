"""The decode-and-batch pipeline every model in the toolkit runs on: worker processes decode and fully preprocess
images, the main thread gathers batches and submits them to the model device, and results are collected one batch
behind so the GPU never waits for the decoders (or the other way round).
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing as mp
import os
import signal
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import TypeVar

import PIL.Image
import torch
from blake3 import blake3

from .images import ImageRecord, Item, Roots

logger = logging.getLogger(__name__)

A = TypeVar("A")  # a preprocessed image, whatever the model wants
P = TypeVar("P")  # a submitted batch


def _load_image(
    preprocessor: Callable[[PIL.Image.Image], A], image_path: str, item: Item
) -> tuple[str, tuple[ImageRecord, A] | Exception]:
    """Worker: decode one image and run the model's full preprocessing (resize, normalize, patchify, dtype cast).

    The preprocessor gets the image in its own mode (first frame if animated) and does any color conversion itself,
    so one that needs the alpha channel can have it. The main process gets back ready-to-stack arrays in the model's
    dtype, so all it has to do for the GPU is gather a batch and copy it over. Returns the record + arrays, or the
    exception.

    Any failure here means "skip this file", so the catch is deliberately broad: a corrupt file must not take the
    whole run down, and Pillow raises a wide variety of exception types for broken inputs.

    The file is read whole once, so its BLAKE3 comes along without a second read.
    """
    try:
        st = os.stat(image_path)
        with open(image_path, "rb") as stream:
            data = stream.read()
        digest = blake3(data).digest()
        with PIL.Image.open(io.BytesIO(data)) as img:
            fmt = img.format or os.path.splitext(item.path)[1].lstrip(".").upper()
            width, height = img.size
            arrays = preprocessor(img)
        record = ImageRecord(item.path, st.st_mtime, st.st_size, width, height, fmt, item.source_id, item.id, digest)
        return image_path, (record, arrays)
    except Exception as e:  # noqa: BLE001 - see docstring
        return image_path, e


@contextlib.contextmanager
def _sigint_ignored() -> Iterator[None]:
    """Ignore SIGINT while worker processes start: an ignored signal survives exec, and a Python started with SIGINT
    ignored never installs its KeyboardInterrupt handler, so a Ctrl-C during the workers' slow spawn bootstrap
    (re-importing the main module, before ``_init_worker`` runs) cannot kill them with a traceback each. Only the main
    thread may change signal handlers; elsewhere this does nothing."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _init_worker() -> None:
    torch.set_num_threads(1)
    # Ctrl-C is the main process's to handle (it cancels the queue and stops cleanly); without this every worker
    # would also die with its own KeyboardInterrupt traceback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def default_workers() -> int:
    return os.cpu_count() or 1


def run_pipeline(
    roots: Roots,
    candidates: Sequence[Item],
    preprocessor: Callable[[PIL.Image.Image], A],
    submit: Callable[[list[A]], P],
    finish: Callable[[list[ImageRecord], P], object],
    *,
    batch_size: int,
    workers: int | None = None,
    progress: Callable[[int], object] | None = None,
) -> int:
    """Decode ``candidates`` (found under ``roots``) in ``workers`` processes and feed them to a model in batches. Returns #images finished.

    Three stages overlap:

    * ``workers`` processes decode and fully preprocess images with ``preprocessor`` (picklable). The number of
      workers is independent of the batch size; it is a function of how fast the CPU can decode, the batch size of
      how much VRAM there is.
    * The main thread keeps ``2 * workers`` decode jobs in flight (so no worker ever waits for the main thread),
      gathers finished arrays into batches and hands each one to ``submit``, which should queue the forward pass on
      the device and return without waiting for it.
    * The device runs the previous batch while the main thread is back to collecting decoded images; ``finish`` gets
      that batch's records and ``submit`` result only when the next batch has been submitted (or at the end).

    Unreadable files are logged and skipped. On an exception (Ctrl-C included) the queued decodes are cancelled and
    the exception propagates; only batches already handed to ``finish`` count as done. Memory: at most
    ``2 * workers`` preprocessed arrays plus two batches live at any time.
    """
    if not candidates:
        return 0
    if workers is None:
        workers = default_workers()
    workers = max(1, workers)
    window = 2 * workers
    done_count = 0
    batch: list[tuple[ImageRecord, A]] = []
    in_flight: tuple[list[ImageRecord], P] | None = None

    def finish_in_flight() -> None:
        nonlocal done_count, in_flight
        if in_flight is None:
            return
        records, pending = in_flight
        in_flight = None
        finish(records, pending)
        done_count += len(records)

    def flush() -> None:
        nonlocal in_flight
        if not batch:
            return
        records = [rec for rec, _ in batch]
        pending = submit([arr for _, arr in batch])
        batch.clear()
        finish_in_flight()  # the previous batch has had the whole decode interval to complete
        in_flight = (records, pending)

    todo = iter(candidates)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
        pending_jobs: set[Future] = set()

        def refill() -> None:
            for item in todo:
                pending_jobs.add(pool.submit(_load_image, preprocessor, item.file(roots), item))
                if len(pending_jobs) >= window:
                    break

        try:
            with _sigint_ignored():
                refill()  # starts the worker processes
            while pending_jobs:
                done, pending_jobs = wait(pending_jobs, return_when=FIRST_COMPLETED)
                for future in done:
                    image_path, result = future.result()
                    if isinstance(result, Exception):
                        logger.warning("Skipping %s: %s", image_path, result)
                    else:
                        batch.append(result)
                        if len(batch) >= batch_size:
                            flush()
                    if progress is not None:
                        progress(1)
                refill()
        except BaseException:
            # Interrupted or failed: drop the queued decodes and wait only for the ones being decoded. Batches already
            # passed to ``finish`` stay done; the gathered and in-flight ones are lost. The wait is what makes the
            # cancel take: the manager thread does it, and only while the executor still exists (without it, the
            # executor is collected as the exception unwinds and every queued decode runs to the end).
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    flush()
    finish_in_flight()
    return done_count
