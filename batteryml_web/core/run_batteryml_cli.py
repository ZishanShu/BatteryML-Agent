import runpy
import sys
from pathlib import Path

import requests


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Put repo root before bin/ so that "batteryml" resolves to the package,
    # not bin/batteryml.py.
    sys.path.insert(0, str(repo_root))

    # Register models defined outside upstream BatteryML before the CLI runs.
    # bin/batteryml.py imports only batteryml.*, so without this an extension
    # model's @MODELS.register() never fires and the run fails later inside
    # MODELS.build() with a bare "<X> is not registered to Models!".
    # Imported here rather than at module scope so that the repo root is on
    # sys.path first. No-op while batteryml_web/models declares no models.
    from batteryml_web.models import (
        ExtensionRegistrationError,
        register_extension_models,
    )

    try:
        register_extension_models()
    except ExtensionRegistrationError as exc:
        # The message already names the module and the corrective action, so a
        # traceback would only bury it. Any other exception is an unexpected
        # bug and is deliberately left to propagate with its traceback intact.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    # Replace upstream's download_file, which leaves a 0-byte or truncated
    # file behind on any failure and then skips it on the next run. Must
    # happen before run_path: bin/batteryml.py binds the name when it imports
    # batteryml.preprocess. Upstream files are not modified (CLAUDE.md §4);
    # see batteryml_web/core/safe_download.py.
    from batteryml_web.core.safe_download import DownloadError, install_safe_download

    install_safe_download()

    cli_path = repo_root / "bin" / "batteryml.py"

    # Preserve user-provided CLI arguments.
    sys.argv = [str(cli_path)] + sys.argv[1:]

    try:
        runpy.run_path(str(cli_path), run_name="__main__")
    except DownloadError as exc:
        # The message names the file, the URL and what to do; a traceback
        # through runpy would only bury it.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except requests.exceptions.RequestException as exc:
        # A network failure outside the safe downloader (download failures are
        # reported above as DownloadError). Name the host this request used —
        # never an assumed one — without credentials or query parameters.
        from batteryml_web.core.safe_download import request_host

        print(
            f"error: Unable to connect to {request_host(exc) or 'the remote server'}. "
            "Check Windows firewall/network policy or configure HTTPS_PROXY, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
