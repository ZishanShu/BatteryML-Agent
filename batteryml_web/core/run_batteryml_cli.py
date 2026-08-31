import runpy
import sys
from pathlib import Path


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

    cli_path = repo_root / "bin" / "batteryml.py"

    # Preserve user-provided CLI arguments.
    sys.argv = [str(cli_path)] + sys.argv[1:]

    runpy.run_path(str(cli_path), run_name="__main__")


if __name__ == "__main__":
    main()
