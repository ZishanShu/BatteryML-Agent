"""The dataset download never leaves a 0-byte or truncated file behind.

The reported failure (README replay, 2026-09-27): a blocked connection left
``data/raw/MATR/MATR_batch_20170512.mat`` at 0 bytes; the next ``download``
printed "already exists. Skip it."; preprocessing then died with h5py's
"file signature not found". ``batteryml_web/core/safe_download.py`` replaces
upstream's ``download_file`` through the CLI wrapper.

No network: every test passes a fake ``get``.

Run from the repository root::

    python -m unittest tests.test_safe_download -v
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import safe_download as sd  # noqa: E402

URL = "https://data.example.org/file/download"
BODY = b"0123456789" * 100  # 1000 bytes


class _Response:
    def __init__(
        self,
        chunks: Iterable[bytes] = (BODY,),
        *,
        status: int = 200,
        length: int | None = len(BODY),
        fail_after: int | None = None,
        failure: BaseException | None = None,
        encoding: str | None = None,
    ) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        if length is not None:
            self.headers["Content-Length"] = str(length)
        if encoding is not None:
            self.headers["Content-Encoding"] = encoding
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self._failure = failure
        self.read = False
        self.closed = False

    def iter_content(self, chunk_size: int = 4096):
        self.read = True
        for index, chunk in enumerate(self._chunks):
            if self._fail_after is not None and index == self._fail_after:
                raise self._failure  # type: ignore[misc]
            yield chunk

    def close(self) -> None:
        self.closed = True


class _Get:
    """Records calls; returns a response or raises."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="safe-download-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.target = self.tmp / "data" / "raw" / "MATR" / "MATR_batch_20170512.mat"
        self.part = self.target.with_name(self.target.name + ".part")

    def download(self, get, **kwargs) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            sd.download_file(URL, self.target, get=get, **kwargs)
        return out.getvalue()

    def assert_nothing_left(self) -> None:
        self.assertFalse(self.target.exists(), "a broken target file was left behind")
        self.assertFalse(self.part.exists(), "a .part file was left behind")


class TestFailuresLeaveNothing(_Case):
    def test_a_refused_connection_writes_no_file(self) -> None:
        """The exact replay case: 403 at the proxy, no bytes ever arrive."""
        get = _Get(requests.exceptions.ProxyError("Tunnel connection failed: 403 Forbidden"))
        with self.assertRaises(sd.DownloadError) as ctx:
            self.download(get)
        self.assert_nothing_left()
        message = str(ctx.exception)
        self.assertIn("data.example.org", message)
        self.assertIn(str(self.target), message)
        self.assertIn("run the same download command again", message)

    def test_an_http_error_writes_no_file(self) -> None:
        response = _Response(status=404)
        with self.assertRaises(sd.DownloadError) as ctx:
            self.download(_Get(response))
        self.assert_nothing_left()
        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertTrue(response.closed)

    def test_a_transfer_that_stops_early_is_deleted(self) -> None:
        response = _Response(chunks=[BODY[:400]], length=len(BODY))
        with self.assertRaises(sd.DownloadError) as ctx:
            self.download(_Get(response))
        self.assert_nothing_left()
        self.assertIn("400 of 1000 bytes", str(ctx.exception))

    def test_a_connection_dropped_mid_stream_is_deleted(self) -> None:
        response = _Response(
            chunks=[BODY[:500], BODY[500:]],
            fail_after=1,
            failure=requests.exceptions.ChunkedEncodingError("Connection broken"),
        )
        with self.assertRaises(sd.DownloadError) as ctx:
            self.download(_Get(response))
        self.assert_nothing_left()
        self.assertIn("stopped part-way", str(ctx.exception))

    def test_ctrl_c_mid_stream_leaves_nothing_and_still_stops(self) -> None:
        response = _Response(
            chunks=[BODY[:500], BODY[500:]], fail_after=1, failure=KeyboardInterrupt()
        )
        with self.assertRaises(KeyboardInterrupt):
            self.download(_Get(response))
        self.assert_nothing_left()
        self.assertTrue(response.closed)

    def test_an_empty_body_is_not_kept(self) -> None:
        with self.assertRaises(sd.DownloadError):
            self.download(_Get(_Response(chunks=[], length=None)))
        self.assert_nothing_left()

    def test_a_failed_redownload_keeps_the_previous_file_untouched(self) -> None:
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(BODY[:300])
        with self.assertRaises(sd.DownloadError):
            self.download(_Get(_Response(chunks=[BODY[:10]], length=len(BODY))))
        self.assertEqual(self.target.read_bytes(), BODY[:300])
        self.assertFalse(self.part.exists())


class TestSuccess(_Case):
    def test_a_complete_download_lands_under_the_final_name(self) -> None:
        output = self.download(_Get(_Response(chunks=[BODY[:600], BODY[600:]])))
        self.assertEqual(self.target.read_bytes(), BODY)
        self.assertFalse(self.part.exists())
        self.assertIn("[DONE]", output)

    def test_the_request_asks_for_an_unencoded_body_and_has_a_timeout(self) -> None:
        get = _Get(_Response())
        self.download(get)
        call = get.calls[0]
        self.assertTrue(call["stream"])
        self.assertEqual(call["headers"].get("Accept-Encoding"), "identity")
        self.assertEqual(call["timeout"], sd.REQUEST_TIMEOUT_S)

    def test_an_encoded_body_is_not_size_checked_against_the_header(self) -> None:
        """Content-Length counts compressed bytes; it cannot be compared."""
        self.download(_Get(_Response(length=17, encoding="gzip")))
        self.assertEqual(self.target.read_bytes(), BODY)

    def test_a_stale_part_file_is_removed_first(self) -> None:
        self.target.parent.mkdir(parents=True)
        self.part.write_bytes(b"stale")
        output = self.download(_Get(_Response()))
        self.assertIn("earlier interrupted download", output)
        self.assertEqual(self.target.read_bytes(), BODY)
        self.assertFalse(self.part.exists())


class TestExistingFiles(_Case):
    def setUp(self) -> None:
        super().setUp()
        self.target.parent.mkdir(parents=True)

    def test_a_zero_byte_file_is_downloaded_again(self) -> None:
        """Upstream skipped exactly this file."""
        self.target.write_bytes(b"")
        output = self.download(_Get(_Response()))
        self.assertIn("is empty", output)
        self.assertEqual(self.target.read_bytes(), BODY)

    def test_a_zero_byte_file_is_gone_even_when_the_retry_fails(self) -> None:
        """So §9 says 'Batch file not found', not h5py's signature error."""
        self.target.write_bytes(b"")
        get = _Get(requests.exceptions.ProxyError("403 Forbidden"))
        with self.assertRaises(sd.DownloadError):
            self.download(get)
        self.assert_nothing_left()

    def test_a_truncated_file_is_downloaded_again(self) -> None:
        self.target.write_bytes(BODY[:123])
        output = self.download(_Get(_Response()))
        self.assertIn("123 bytes but the server's copy has 1000", output)
        self.assertEqual(self.target.read_bytes(), BODY)

    def test_a_complete_file_is_skipped_without_reading_the_body(self) -> None:
        self.target.write_bytes(BODY)
        response = _Response()
        output = self.download(_Get(response))
        self.assertIn("already exists and is complete", output)
        self.assertFalse(response.read)
        self.assertTrue(response.closed)

    def test_a_declared_length_is_checked_without_any_request(self) -> None:
        self.target.write_bytes(BODY)
        get = _Get(AssertionError("no request may be made"))
        output = self.download(get, total_length=len(BODY))
        self.assertIn("already exists and is complete", output)
        self.assertEqual(get.calls, [])

    def test_a_declared_length_mismatch_downloads_again(self) -> None:
        self.target.write_bytes(BODY[:10])
        output = self.download(_Get(_Response()), total_length=len(BODY))
        self.assertIn("should have 1000", output)
        self.assertEqual(self.target.read_bytes(), BODY)

    def test_an_unverifiable_file_is_kept_and_says_so(self) -> None:
        self.target.write_bytes(BODY[:50])
        output = self.download(_Get(_Response(length=None)))
        self.assertIn("completeness cannot be checked", output)
        self.assertEqual(self.target.read_bytes(), BODY[:50])

    def test_an_unreachable_server_cannot_confirm_an_existing_file(self) -> None:
        self.target.write_bytes(BODY)
        get = _Get(requests.exceptions.ConnectionError("unreachable"))
        with self.assertRaises(sd.DownloadError) as ctx:
            self.download(get)
        self.assertIn("cannot be checked", str(ctx.exception))
        self.assertEqual(self.target.read_bytes(), BODY)

    def test_force_downloads_a_complete_file_again(self) -> None:
        self.target.write_bytes(b"x" * len(BODY))
        self.download(_Get(_Response()), force=True)
        self.assertEqual(self.target.read_bytes(), BODY)


class TestWiring(unittest.TestCase):
    """The replacement reaches bin/batteryml.py without editing upstream."""

    def test_the_signature_is_compatible_with_upstream(self) -> None:
        tree = ast.parse(
            (REPO_ROOT / "batteryml" / "preprocess" / "download.py").read_text(encoding="utf-8")
        )
        upstream = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "download_file"
        )
        upstream_params = [a.arg for a in upstream.args.args]
        ours = [
            p.name for p in inspect.signature(sd.download_file).parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        self.assertEqual(ours, upstream_params)

    def test_install_replaces_the_package_attribute(self) -> None:
        import batteryml.preprocess as package

        original = package.download_file
        self.addCleanup(setattr, package, "download_file", original)
        sd.install_safe_download()
        self.assertIs(package.download_file, sd.download_file)

    def test_the_wrapper_installs_it_before_handing_over(self) -> None:
        tree = ast.parse(
            (REPO_ROOT / "batteryml_web" / "core" / "run_batteryml_cli.py").read_text(
                encoding="utf-8"
            )
        )
        calls = [
            (node.lineno, ast.unparse(node.func))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        install = [line for line, name in calls if name == "install_safe_download"]
        handover = [line for line, name in calls if name == "runpy.run_path"]
        self.assertEqual(len(install), 1)
        self.assertEqual(len(handover), 1)
        self.assertLess(install[0], handover[0])

    def test_the_cli01_check_fails_when_the_wiring_is_removed(self) -> None:
        from scripts.verify import check_cli_contract as cli01

        problems, _ = cli01._download_wiring_problems(REPO_ROOT)
        self.assertEqual(problems, [])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (cli01.WRAPPER, cli01.SAFE_DOWNLOAD, cli01.UPSTREAM_DOWNLOAD):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REPO_ROOT / relative, root / relative)
            wrapper = root / cli01.WRAPPER
            wrapper.write_text(
                wrapper.read_text(encoding="utf-8").replace(
                    "    install_safe_download()\n", ""
                ),
                encoding="utf-8",
            )
            problems, _ = cli01._download_wiring_problems(root)
        self.assertEqual(len(problems), 1)
        self.assertIn("install_safe_download", problems[0])

    def test_upstream_is_unmodified_by_this_fix(self) -> None:
        """CLAUDE.md §4: the upstream function still has its own body."""
        source = (REPO_ROOT / "batteryml" / "preprocess" / "download.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("with open(filename, 'wb') as f:", source)
        self.assertNotIn("safe_download", source)


if __name__ == "__main__":
    unittest.main()
