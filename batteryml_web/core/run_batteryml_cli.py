import runpy
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Put repo root before bin/ so that "batteryml" resolves to the package,
    # not bin/batteryml.py.
    sys.path.insert(0, str(repo_root))

    cli_path = repo_root / "bin" / "batteryml.py"

    # Preserve user-provided CLI arguments.
    sys.argv = [str(cli_path)] + sys.argv[1:]

    runpy.run_path(str(cli_path), run_name="__main__")


if __name__ == "__main__":
    main()
