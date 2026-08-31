"""Extension point for models that are not part of upstream BatteryML.

`CLAUDE.md` §4 forbids adding code to `batteryml/`. A model defined here is
registered into BatteryML's own ``MODELS`` registry through the documented
extension point (``@MODELS.register()``), so the pipeline treats it exactly
like a core model without `batteryml/` being touched.

How registration reaches the pipeline
-------------------------------------

``bin/batteryml.py`` imports only ``batteryml.*``, so it never imports this
package and an extension model's ``@MODELS.register()`` would never fire.
``batteryml_web/core/run_batteryml_cli.py`` — the wrapper every invocation goes
through per `CLAUDE.md` §5 — calls :func:`register_extension_models` before
handing control to the CLI. The Streamlit UI shells out to that same wrapper
(``batteryml_web/app.py:698``, ``:1176``), so both routes are covered.

Anything that bypasses the wrapper does **not** see these models: the
``batteryml`` console script installed by ``setup.py`` and
``run_all_rul_baseline.sh`` both execute ``bin/batteryml.py`` directly.

Adding a model
--------------

1. Create ``batteryml_web/models/<name>.py`` with a class decorated
   ``@MODELS.register()``, subclassing ``SklearnModel`` or ``NNModel``.
2. Add ``"<name>"`` to :data:`EXTENSION_MODULES` below.
3. Create ``configs/baselines/sklearn/<name>/`` (or ``nn_models/<name>/``)
   with one config per split preset.
4. Register the display name in ``batteryml_web/core/config_registry.py``.

Step 2 is not optional and not inferred: a module file that is present but
unlisted is never imported, so its registration never fires and the failure
would surface much later as a bare
``KeyError: <X> is not registered to Models!`` from ``MODELS.build()``.
:func:`register_extension_models` therefore refuses to run at all while an
unlisted module file exists, and ``REG-01`` reports the same condition.

Full walkthrough: ``docs/EXTENDING_MODELS.md``.

Keep this module import-light
-----------------------------

It must stay importable without torch, sklearn or streamlit installed: the
verification harness imports it to read :data:`EXTENSION_MODULES` and to detect
unlisted files, and it does so in environments where the pipeline cannot run.
Put every heavy import inside the individual model modules, never here.
"""

from __future__ import annotations

import importlib
from pathlib import Path

#: Extension modules to import, without the package prefix. Listed explicitly
#: rather than auto-discovered so that what gets registered is declared in one
#: readable place. Files whose name starts with "_" are helpers and are never
#: imported as models.
EXTENSION_MODULES: tuple[str, ...] = (
    "extra_trees",
    "huber",
    "knn",
    "lasso",
)

_PACKAGE_DIR = Path(__file__).resolve().parent


class ExtensionRegistrationError(RuntimeError):
    """An extension model could not be registered.

    Raised instead of letting the pipeline fail later with an unrelated
    ``KeyError`` from the registry lookup.
    """


def extension_module_files() -> list[Path]:
    """Every module file in this package that is meant to define a model.

    Excludes ``__init__.py`` and any ``_``-prefixed helper module.
    """
    return sorted(
        path
        for path in _PACKAGE_DIR.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def unlisted_modules() -> list[str]:
    """Module names present on disk but absent from :data:`EXTENSION_MODULES`."""
    listed = set(EXTENSION_MODULES)
    return sorted(path.stem for path in extension_module_files() if path.stem not in listed)


def missing_modules() -> list[str]:
    """Names in :data:`EXTENSION_MODULES` that have no module file on disk."""
    on_disk = {path.stem for path in extension_module_files()}
    return sorted(name for name in EXTENSION_MODULES if name not in on_disk)


def register_extension_models() -> list[str]:
    """Import every listed extension module so its ``@MODELS.register()`` fires.

    Returns:
        The fully qualified names of the modules imported, in declaration
        order. Empty when no extension model is declared, which is the
        shipped state.

    Raises:
        ExtensionRegistrationError: if a declared module is missing, if a
            module file exists but is not declared, or if importing a declared
            module fails. The message names the offending module and the
            corrective action; the original exception is chained.
    """
    missing = missing_modules()
    if missing:
        raise ExtensionRegistrationError(
            f"EXTENSION_MODULES lists {len(missing)} module(s) with no file in "
            f"{_PACKAGE_DIR}: {', '.join(missing)}. Create the file(s), or remove "
            f"the name(s) from EXTENSION_MODULES in {__name__}."
        )

    unlisted = unlisted_modules()
    if unlisted:
        raise ExtensionRegistrationError(
            f"{len(unlisted)} module file(s) in {_PACKAGE_DIR} are not listed in "
            f"EXTENSION_MODULES: {', '.join(unlisted)}. An unlisted module is never "
            f"imported, so its @MODELS.register() never fires and the run would fail "
            f"later with 'is not registered to Models!'. Add the name(s) to "
            f"EXTENSION_MODULES in {__name__}, or prefix the file with '_' if it is a "
            f"helper rather than a model."
        )

    imported: list[str] = []
    for module_name in EXTENSION_MODULES:
        qualified = f"{__name__}.{module_name}"
        try:
            importlib.import_module(qualified)
        except Exception as exc:  # noqa: BLE001 - re-raised with context, never hidden
            raise ExtensionRegistrationError(
                f"Could not import extension model module {qualified}: "
                f"{type(exc).__name__}: {exc}. Fix that module, or remove "
                f"'{module_name}' from EXTENSION_MODULES in {__name__}."
            ) from exc
        imported.append(qualified)
    return imported
