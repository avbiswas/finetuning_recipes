"""Compatibility helpers for TRL/Transformers version skew."""

from __future__ import annotations


def patch_trl_optional_dependency_checks() -> None:
    """Make TRL 0.24 optional dependency flags real booleans.

    TRL 0.24 imports `transformers.utils.import_utils._is_package_available`
    and expects a bool when `return_version=False`. Newer Transformers returns
    `(available, version)` in that case. Non-empty tuples are truthy, so TRL
    tries to import optional extras like mergekit and llm-blender even when
    they are not installed.
    """

    import trl.import_utils as import_utils

    for name, value in vars(import_utils).items():
        if name.endswith("_available") and isinstance(value, tuple):
            setattr(import_utils, name, value[0])
