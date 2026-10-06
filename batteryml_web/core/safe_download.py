"""Dataset download that never leaves a broken file behind.

Upstream ``batteryml.preprocess.download.download_file``
(``batteryml/preprocess/download.py:50``) opens the target file *before* it
sends the request. Any failure after that — a refused connection, an HTTP
error, Ctrl+C, a transfer that stops early — leaves a 0-byte or truncated file
under the final name, and the next ``download`` run prints
``[INFO] <file> already exists. Skip it.`` and keeps it. Preprocessing then
fails far from the cause (h5py: ``OSError: Unable to synchronously open file
(file signature not found)``), which a fresh-environment replay of the README
reproduced [VERIFIED, 2026-09-27].

This module is the replacement. CLAUDE.md §4 forbids editing upstream, so
``batteryml_web/core/run_batteryml_cli.py`` calls :func:`install_safe_download`
before it hands over to ``bin/batteryml.py``. That script reads
``download_file`` from the ``batteryml.preprocess`` package at import time
(``from batteryml.preprocess import ... download_file``), so it receives this
function instead [STATIC: bin/batteryml.py:11-13, :96].

Guarantees:

* the request is made, and its HTTP status checked, before any file is
  created;
* data is written to ``<name>.part`` and renamed to ``<name>`` only after the
  byte count matches the expected size (``os.replace``, atomic on one
  filesystem). On any failure — including ``KeyboardInterrupt`` — the
  ``.part`` file is deleted and the exception propagates;
* an existing file is kept only when its size equals the expected size
  (from ``DOWNLOAD_LINKS`` when given there, else the server's
  ``Content-Length``). A 0-byte or short file is downloaded again.

Kept from upstream on purpose: the signature, and ``verify=False`` on the
request. Changing TLS verification could break a dataset host that works
today, and none of the hosts can be reached from the environment this was
written in, so that is left as a separate decision [ASSUMED: upstream chose
``verify=False`` for a host with a certificate problem; not checked].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

#: (connect, read) timeouts in seconds. Upstream sets none, so a stalled
#: connection hangs forever. Read is the longest gap between two chunks, not
#: the whole transfer.
REQUEST_TIMEOUT_S: tuple[float, float] = (30.0, 300.0)

PART_SUFFIX = ".part"


class DownloadError(RuntimeError):
    """A download failed; the message names the file, the URL and the fix."""


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def _unreachable_advice(url: str) -> str:
    host = _host(url)
    return (
        f"Check that this machine can reach {host} (for example, open "
        f"https://{host} in a browser on the same machine; a university or "
        "company proxy may block it), then run the same download command again."
    )


def _expected_size(response: Any, total_length: Optional[int]) -> Optional[int]:
    """The byte count a complete file must have, or None when unknown.

    ``total_length`` from ``DOWNLOAD_LINKS`` wins. Otherwise the server's
    ``Content-Length`` — but only for an unencoded body: with gzip transfer
    encoding the header counts compressed bytes while ``iter_content`` yields
    decompressed ones, so the two can never be compared.
    """
    if total_length is not None:
        return int(total_length)
    encoding = str(response.headers.get("Content-Encoding", "") or "").strip().lower()
    if encoding not in ("", "identity"):
        return None
    header = response.headers.get("Content-Length")
    if header is None:
        return None
    try:
        return int(header)
    except (TypeError, ValueError):
        return None


def _open(url: str, filename: Path, get: Callable[..., Any]) -> Any:
    """Send the request; raise DownloadError on any network or HTTP failure."""
    try:
        response = get(
            url,
            stream=True,
            verify=False,
            timeout=REQUEST_TIMEOUT_S,
            headers={"Accept-Encoding": "identity"},
        )
    except requests.RequestException as exc:
        raise DownloadError(
            f"Could not download {filename} from {url}: {type(exc).__name__}: {exc}. "
            f"Nothing was written to {filename}. {_unreachable_advice(url)}"
        ) from exc
    if response.status_code != 200:
        response.close()
        raise DownloadError(
            f"Could not download {filename}: the server answered HTTP "
            f"{response.status_code} for {url}. Nothing was written to {filename}. "
            "If this repeats, the dataset may have moved; see dataprepare.md for "
            "manual download instructions."
        )
    return response


def _stream_to_part(
    response: Any,
    part: Path,
    expected: Optional[int],
    chunk_size: int,
    label: str,
) -> int:
    """Write the body to ``part``; return the byte count written."""
    written = 0
    chunks: Iterator[bytes] = response.iter_content(chunk_size=chunk_size)
    with open(part, "wb") as handle, tqdm(
        total=expected, unit="B", unit_scale=True, desc=label
    ) as bar:
        for chunk in chunks:
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
            bar.update(len(chunk))
        handle.flush()
        os.fsync(handle.fileno())
    return written


def download_file(
    url: str,
    filename: os.PathLike[str] | str,
    update_interval: int = 500,
    chunk_size: int = 4096,
    total_length: Optional[int] = None,
    force: bool = False,
    *,
    get: Callable[..., Any] = requests.get,
) -> None:
    """Download ``url`` to ``filename`` without ever leaving a broken file.

    Same positional/keyword parameters as upstream's ``download_file`` so
    ``bin/batteryml.py`` can call it unchanged; ``update_interval`` is
    accepted and unused (tqdm paces its own updates). ``get`` exists so the
    behaviour can be tested without a network.

    Raises:
        DownloadError: the server could not be reached, answered with an
            error, or sent fewer/more bytes than expected. The final file is
            then exactly as it was before the call (absent, or the old one).
    """
    del update_interval  # kept for signature compatibility only
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + PART_SUFFIX)

    if part.exists():
        print(f"[INFO] Removing {part}, left by an earlier interrupted download.")
        part.unlink()

    response: Any = None
    if target.exists() and not force:
        size = target.stat().st_size
        if size == 0:
            # Removed now, not when the new copy lands: an empty file is never
            # worth keeping, and without it a failed retry leaves the clear
            # "Batch file not found" at preprocessing instead of h5py's
            # "file signature not found".
            print(
                f"[INFO] {target} is empty (left by an earlier failed download). "
                "Removing it and downloading it again."
            )
            target.unlink()
        elif total_length is not None and size == int(total_length):
            print(f"[INFO] {target} already exists and is complete ({size} bytes). Skip it.")
            return
        elif total_length is not None:
            print(
                f"[INFO] {target} has {size} bytes but should have {int(total_length)} "
                "(an earlier download was cut off). Downloading it again."
            )
        else:
            # Only the server knows the complete size.
            try:
                response = _open(url, target, get)
            except DownloadError as exc:
                raise DownloadError(
                    f"{target} already exists ({size} bytes), but whether it is "
                    f"complete cannot be checked: {exc} If the earlier download of "
                    "this file finished without any error, it is complete and you "
                    "can go on to preprocessing."
                ) from exc
            expected = _expected_size(response, None)
            if expected is None:
                response.close()
                print(
                    f"[INFO] {target} already exists ({size} bytes). The server does "
                    "not report a file size, so completeness cannot be checked; the "
                    "file is kept. If an earlier download of it was interrupted, "
                    f"delete {target} and run the download again."
                )
                return
            if expected == size:
                response.close()
                print(f"[INFO] {target} already exists and is complete ({size} bytes). Skip it.")
                return
            print(
                f"[INFO] {target} has {size} bytes but the server's copy has "
                f"{expected} (an earlier download was cut off). Downloading it again."
            )

    if response is None:
        response = _open(url, target, get)

    try:
        expected = _expected_size(response, total_length)
        try:
            written = _stream_to_part(response, part, expected, chunk_size, target.name)
        except requests.RequestException as exc:
            raise DownloadError(
                f"The download of {target} from {url} stopped part-way: "
                f"{type(exc).__name__}: {exc}. The incomplete data was deleted. "
                f"{_unreachable_advice(url)}"
            ) from exc
        if written == 0:
            raise DownloadError(
                f"The server sent an empty file for {target} ({url}). Nothing was "
                "kept. Run the download again; if it repeats, see dataprepare.md "
                "for manual download instructions."
            )
        if expected is not None and written != expected:
            raise DownloadError(
                f"The download of {target} ended after {written} of {expected} bytes. "
                f"The incomplete data was deleted. {_unreachable_advice(url)}"
            )
        os.replace(part, target)
    except BaseException:
        # Includes KeyboardInterrupt: Ctrl+C must not leave a .part behind.
        if part.exists():
            part.unlink()
        raise
    finally:
        response.close()
    print(f"[DONE] {target} ({written} bytes)")


def install_safe_download() -> None:
    """Make ``batteryml.preprocess.download_file`` this module's function.

    Must run before ``bin/batteryml.py`` executes its
    ``from batteryml.preprocess import ... download_file``; the CLI wrapper
    calls it right before ``runpy.run_path``. Upstream files are not touched.
    """
    import batteryml.preprocess as preprocess_package

    preprocess_package.download_file = download_file
