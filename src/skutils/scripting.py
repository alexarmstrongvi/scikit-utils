"""Utilities for scripts."""
# Standard library
from collections.abc import Iterable
from copy import deepcopy
import logging
from pathlib import Path
import shutil
import time

# 1st party
from skutils import user_input

# Globals
log = logging.getLogger(__name__)


################################################################################
def merge_config_files(
    cfgs: Iterable[dict],
    default_cfg: dict | None = None,
) -> dict:
    """Merge configuration files with later ones overwriting earlier ones.

    Parameters
    ==========
    cfgs:
        Separate configurations to be merged
    default_cfg:
        Configuration with default values for all parameters

    Returns
    =======
    Single merged configuration file
    """
    if default_cfg is not None:
        merged_cfg = deepcopy(default_cfg)
        allow_new_keys = False
    else:
        merged_cfg = {}
        allow_new_keys = True

    for cfg_update in cfgs:
        update_config(
            original=merged_cfg,
            update=cfg_update,
            copy=False,
            allow_new_keys=allow_new_keys,
        )
    return merged_cfg


def update_config(
    original: dict,
    update: dict,
    copy: bool = True,
    allow_new_keys: bool = False,
    concat_lists: bool = False,
    overwrite_lists: bool = False,
) -> dict:
    """Update a configuration dictionary using a dictionary with updated
    values.

    The intended use case is combining common configuration formats (e.g. YAML,
    JSON, TOML) after being read into python dictionaries. Therefore, this
    function aims to handle the subset of python types these configuration
    formats support (e.g. int, float, str, dict, list). Other types (e.g. tuple,
    set, numpy array, bytes) are not handled in any special way and therefore
    will overwrite the original value similar to an int or float value.

    Parameters
    ==========
    original:
        Original configuration dictionary
    update:
        Configuration dictionary with values to update in the original
    copy:
        Apply updates to a copy of the original, returning a new dictionary
    allow_new_keys:
        Allow the update to contain keys not in the original
    concat_lists:
        Update lists by concatenating them, allowing duplicates
    overwrite_lists:
        Update lists by overwriting the original with the updated list

    Returns
    =======
    Updated configuration dictionary
    """

    if copy:
        # Use shallow copy to reduce memory usage
        # This requires care be taken when updating mutable values below
        original = original.copy()

    for key, val in update.items():
        if key not in original:
            if not allow_new_keys:
                raise KeyError(f"{key!r} not in original dictionary")
            original[key] = val
        elif isinstance(val, dict):
            if isinstance(original[key], dict):
                original[key] = update_config(original[key], val, copy, allow_new_keys)
            else:
                original[key] = update_config({}, val, copy, allow_new_keys=True)
        elif isinstance(val, list):
            original_list = original[key] or []
            if overwrite_lists:
                original[key] = val
            elif concat_lists:
                original[key] = original_list + val
            else:
                original[key] = update_list(original_list, val)
        else:
            original[key] = val

    return original


def update_list(original, update):
    """Append new elements, preserving order from both lists.

    This will not handle duplicates already in the original or update.
    It is assumed the user intends those duplicates.
    """
    merged_list = original.copy()
    for x in update:
        if x not in original:
            merged_list.append(x)
    return merged_list


def require_empty_dir(
    path: Path,
    parents: bool = False,
    overwrite: bool = False,
) -> None:
    # Make directory if it doesn't exist or is empty
    if not path.is_dir():
        path.mkdir(parents=parents)
        return
    if not any(path.iterdir()):
        return

    if not overwrite:
        # Check if user wants to delete contents of directory
        overwrite = user_input.request_permission(f"Delete contents of {path}?")

    files = list(path.rglob("*"))
    if overwrite:
        log.warning("Deleting all %d files from %s", len(files), path)
        time.sleep(2)  # Give the user a moment to realize if this was a mistake
        shutil.rmtree(path)
        path.mkdir(parents=parents)
    else:
        raise FileExistsError(
            f"{len(files)} files found (e.g. {files[0].name}): {path}"
        )
