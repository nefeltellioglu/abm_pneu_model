#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract tables and params from all disease_X_Y.hd5 files found under a given
output folder (recursively into any depth of sub-folders).

Usage
-----
    python extract_tables_from_hd5_file.py <folder_name>

    folder_name : sub-folder name under src/pneu_abm/output/
                  e.g. "base_results"

Output
------
For each ``disease_X_Y.hd5`` file found, the following CSV files are written
**alongside** the .hd5 file:

* ``params_X_Y.csv``
    One-row CSV with one column per /params attribute.  Byte-string values are
    decoded as UTF-8; pickle-encoded values (e.g. ``age_classes``, ``years``)
    are unpickled and then JSON-encoded; list/array values are JSON-encoded.

* ``{group}_{table}_X_Y.csv``
    One CSV per dataset found anywhere in the HDF5 hierarchy.  The table name
    is built from the HDF5 path by replacing ``/`` with ``_``.
    e.g. ``disease_byage/base_data`` → ``disease_byage_base_data_0_1.csv``
    Cells whose values are NumPy arrays (per-row array fields) are serialised
    as JSON strings so they survive the CSV round-trip and can be decoded by
    ``read_tables_from_extracted_files.py``.
"""

import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUTPUT_BASE = os.path.join(REPO_ROOT, "src/pneu_abm/output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_param_attr(value):
    """Decode a single /params attribute value to a JSON-serialisable Python type.

    * numpy scalars → Python int / float
    * bytes / numpy bytes_ → try pickle.loads first; on failure decode as UTF-8
    * numpy arrays → list (then JSON-encoded by the caller)
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, np.bytes_)):
        raw = bytes(value)
        try:
            decoded = pickle.loads(raw)
            # Pickle decoded: could be list, dict, int …
            if isinstance(decoded, (list, dict)):
                return decoded  # caller will JSON-encode if needed
            return decoded
        except Exception:
            return raw.decode("utf-8", errors="replace")
    return value


def _to_serializable(obj):
    """Recursively convert numpy / bytes objects to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return [_to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (bytes, np.bytes_)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def _cell_to_csv_value(value):
    """Convert a single dataset cell to a CSV-safe value.

    NumPy arrays (per-row array fields) are serialised as JSON strings so that
    ``read_tables_from_extracted_files.py`` can decode them back.
    Plain scalars are returned as their Python equivalents.
    """
    if isinstance(value, np.ndarray):
        return json.dumps(_to_serializable(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bytes_, bytes)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def _extract_params(f, hdf_path: str, stem: str) -> None:
    """Write /params attributes to params_{stem}.csv next to the .hd5 file."""
    row = {}
    for k, v in f["params"].attrs.items():
        decoded = _decode_param_attr(v)
        # JSON-encode lists/dicts so the CSV cell is a readable string
        if isinstance(decoded, (list, dict)):
            row[k] = json.dumps(decoded)
        else:
            row[k] = decoded

    df = pl.DataFrame([row])
    out_path = os.path.join(os.path.dirname(hdf_path), f"params_{stem}.csv")
    df.write_csv(out_path)
    print(f"    params  -> {out_path}")


def _extract_dataset(dataset: h5py.Dataset, path_in_hdf: str,
                     hdf_path: str, stem: str) -> None:
    """Write a structured HDF5 dataset to a CSV file.

    Parameters
    ----------
    dataset    : open h5py Dataset object
    path_in_hdf: HDF5-internal path, e.g. "disease_byage/base_data"
    hdf_path   : filesystem path of the .hd5 file
    stem       : year-range stem extracted from the filename, e.g. "0_1"
    """
    arr = dataset[()]

    # Only handle structured (record/table) arrays
    if arr.dtype.names is None:
        return

    data_dict: dict = {}
    for col in arr.dtype.names:
        col_data = arr[col]
        if col_data.ndim == 0:
            # Scalar column (shouldn't happen in a table but be defensive)
            data_dict[col] = [_cell_to_csv_value(col_data.item())]
        elif col_data.ndim == 1:
            first = col_data[0] if len(col_data) > 0 else None
            if isinstance(first, np.ndarray):
                # Each row cell is itself an array — JSON-encode each cell
                data_dict[col] = [json.dumps(_to_serializable(v)) for v in col_data]
            elif isinstance(first, (np.bytes_, bytes)):
                data_dict[col] = [
                    bytes(v).decode("utf-8", errors="replace") for v in col_data
                ]
            else:
                data_dict[col] = col_data.tolist()
        else:
            # 2-D column: JSON-encode each row with full recursive conversion
            data_dict[col] = [
                json.dumps(_to_serializable(row))
                for row in col_data
            ]

    # Table name: replace HDF5 path separators with underscores
    table_name = path_in_hdf.replace("/", "_")
    out_path = os.path.join(
        os.path.dirname(hdf_path), f"{table_name}_{stem}.csv"
    )
    df = pl.DataFrame(data_dict)
    df.write_csv(out_path)
    print(f"    {path_in_hdf:<40} -> {os.path.basename(out_path)}")


def extract_hd5(hdf_path: str) -> None:
    """Extract all tables and params from a single .hd5 file."""
    fname = os.path.basename(hdf_path)
    # stem = "0_1" from "disease_0_1.hd5"
    stem = fname.removeprefix("disease_").removesuffix(".hd5")
    print(f"\n  {hdf_path}")

    with h5py.File(hdf_path, "r") as f:
        # --- params attributes ---
        if "params" in f:
            _extract_params(f, hdf_path, stem)

        # --- all structured datasets anywhere in the hierarchy ---
        def _visitor(path_in_hdf: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and obj.dtype.names:
                _extract_dataset(obj, path_in_hdf, hdf_path, stem)

        f.visititems(_visitor)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(folder_name: str = None) -> None:
    if folder_name is None:
        parser = argparse.ArgumentParser(
            description=(
                "Extract tables and params from disease_X_Y.hd5 files to CSV."
            )
        )
        parser.add_argument(
            "folder_name",
            help='Sub-folder under src/pneu_abm/output/ (e.g. "base_results")',
        )
        args = parser.parse_args()
        folder_name = args.folder_name

    target_dir = os.path.join(OUTPUT_BASE, folder_name)
    if not os.path.isdir(target_dir):
        print(f"Error: directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    # Recursively collect all disease_*.hd5 files
    hd5_files = []
    for root, _dirs, files in os.walk(target_dir):
        for fname in sorted(files):
            if fname.startswith("disease_") and fname.endswith(".hd5"):
                hd5_files.append(os.path.join(root, fname))

    if not hd5_files:
        print(f"No disease_*.hd5 files found under {target_dir}")
        sys.exit(0)

    print(f"Found {len(hd5_files)} HDF5 file(s) under '{folder_name}'")
    for hdf_path in hd5_files:
        extract_hd5(hdf_path)

    print("\nDone.")


if __name__ == "__main__":
    main(folder_name="base_results")
