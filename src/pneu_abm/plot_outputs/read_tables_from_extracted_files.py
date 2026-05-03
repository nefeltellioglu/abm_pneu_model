#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read extracted CSV tables from all sub-folders and combine them into one
Polars DataFrame, with extra columns identifying each sub-folder level.

Usage
-----
    python read_tables_from_extracted_files.py <folder_name> <table_name> [--output PATH]

    folder_name : sub-folder under src/pneu_abm/output/ (e.g. "base_results")
    table_name  : base table name to collect, **without** the trailing _X_Y
                  e.g. "disease_byage_base_data"
                  matches files such as disease_byage_base_data_0_1.csv

    --output PATH  (optional) write the combined DataFrame to this CSV path.

How it works
------------
1.  Walk ``src/pneu_abm/output/<folder_name>/`` recursively.
2.  Collect every CSV file whose name starts with ``<table_name>_`` and ends
    with ``_<int>_<int>.csv``  (the year-range stem written by the extract
    script).
3.  For each file, read it and add one extra column per directory level
    between the output folder root and the file.
    e.g. for a file at
      ``…/base_results/params_vaccine_list=vaccine_configs/vaccine_list.csv/seed_0/disease_byage_base_data_0_1.csv``
    the added columns are:
      ``subfolder_1 = "params_vaccine_list=vaccine_configs"``
      ``subfolder_2 = "vaccine_list.csv"``
      ``subfolder_3 = "seed_0"``
4.  Columns whose values are JSON-encoded lists (written by the extract
    script) are decoded back to Python lists and stored as Polars List
    columns with an automatically inferred element type.
5.  All per-file DataFrames are concatenated diagonally into one big frame
    which is printed (and optionally saved).

Helper
------
You can also ``import`` this module and call ``read_extracted_tables()``
directly, which returns the combined Polars DataFrame.
"""

import argparse
import json
import os
import re
import sys

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUTPUT_BASE = os.path.join(REPO_ROOT, "src/pneu_abm/output")

# Regex for the year-range suffix: _<int>_<int>.csv
_STEM_RE = re.compile(r"_\d+_\d+\.csv$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_json_list(value) -> bool:
    """Return True when *value* is a string that decodes to a JSON list."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    try:
        parsed = json.loads(stripped)
        return isinstance(parsed, list)
    except (json.JSONDecodeError, ValueError):
        return False


def _infer_list_element_dtype(sample_list: list):
    """Return the Polars dtype that best represents elements of *sample_list*."""
    if not sample_list:
        return pl.Float64
    first = sample_list[0]
    if isinstance(first, bool):
        return pl.Boolean
    if isinstance(first, int):
        return pl.Int64
    if isinstance(first, float):
        return pl.Float64
    return pl.String


def _decode_list_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Detect JSON-list string columns and decode them to Polars List columns.

    The element dtype is inferred from the first non-null value in each column.
    If inference fails for any column it is left as a String column.
    """
    if df.height == 0:
        return df

    # Sample the first non-null row to detect list columns
    sample_row = df.head(1).to_dicts()[0]
    list_cols = [col for col, val in sample_row.items() if _looks_like_json_list(val)]

    for col in list_cols:
        # Find first non-null value to infer element type
        non_null = df.filter(pl.col(col).is_not_null()).head(1)
        if non_null.height == 0:
            continue
        try:
            sample_list = json.loads(non_null[col][0])
        except (json.JSONDecodeError, TypeError):
            continue

        elem_dtype = _infer_list_element_dtype(sample_list)
        try:
            df = df.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x: json.loads(x) if isinstance(x, str) else None,
                    return_dtype=pl.List(elem_dtype),
                )
                .alias(col)
            )
        except Exception:
            # Fall back: keep as JSON string if decoding fails
            pass

    return df


def _subfolder_columns(csv_path: str, folder_name: str) -> dict:
    """Return a dict of ``subfolder_N → name`` for each directory level between
    the named output folder and the CSV file (exclusive of both)."""
    base = os.path.join(OUTPUT_BASE, folder_name)
    rel = os.path.relpath(os.path.dirname(csv_path), base)
    parts = rel.split(os.sep) if rel != "." else []
    return {f"subfolder_{i + 1}": part for i, part in enumerate(parts)}


def _find_csv_files(target_dir: str, table_name: str) -> list:
    """Walk *target_dir* recursively and return all CSV files matching the
    pattern ``<table_name>_<int>_<int>.csv``."""
    prefix = table_name + "_"
    found = []
    for root, _dirs, files in os.walk(target_dir):
        for fname in sorted(files):
            if fname.startswith(prefix) and _STEM_RE.search(fname):
                found.append(os.path.join(root, fname))
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_extracted_tables(folder_name: str, table_name: str) -> pl.DataFrame:
    """Find, read, and combine all extracted CSV files for *table_name*.

    Parameters
    ----------
    folder_name:
        Sub-folder under ``src/pneu_abm/output/``  (e.g. ``"base_results"``).
    table_name:
        Base name of the table to collect, without trailing ``_X_Y``
        (e.g. ``"disease_byage_base_data"``).

    Returns
    -------
    pl.DataFrame
        All matching rows concatenated into one DataFrame, with extra
        ``subfolder_N`` columns identifying the directory path of each row.
    """
    target_dir = os.path.join(OUTPUT_BASE, folder_name)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Output directory not found: {target_dir}")

    csv_files = _find_csv_files(target_dir, table_name)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{table_name}_<X>_<Y>.csv' under {target_dir}"
        )

    print(f"Found {len(csv_files)} CSV file(s) for table '{table_name}'")

    frames = []
    for csv_path in csv_files:
        # Read all columns as strings first, then decode lists and numerics
        df = pl.read_csv(csv_path, infer_schema_length=0)

        # Decode JSON-list columns
        df = _decode_list_columns(df)

        # Cast remaining string columns to numeric where possible
        for col in df.columns:
            if df[col].dtype == pl.String:
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Int64).alias(col))
                    continue
                except Exception:
                    pass
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
                except Exception:
                    pass  # leave as String

        # Add subfolder identifier columns
        sub_cols = _subfolder_columns(csv_path, folder_name)
        for col_name, col_value in sub_cols.items():
            df = df.with_columns(pl.lit(col_value).alias(col_name))

        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _discover_table_names(target_dir: str) -> list:
    """Return all distinct base table names found recursively under *target_dir*.

    A base table name is the CSV filename with the trailing ``_<int>_<int>.csv``
    stem removed, and excluding ``params_`` files (which are not data tables).
    """
    names = set()
    for root, _dirs, files in os.walk(target_dir):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            if fname.startswith("params_"):
                continue
            m = _STEM_RE.search(fname)
            if m:
                base = fname[: m.start()]
                names.add(base)
    return sorted(names)


def _write_csv_safe(combined: pl.DataFrame, output: str) -> None:
    """Write *combined* to *output*, re-encoding List columns as JSON strings."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    csv_ready = combined
    for col in combined.columns:
        if isinstance(combined[col].dtype, pl.List):
            csv_ready = csv_ready.with_columns(
                pl.col(col)
                .map_elements(
                    lambda v: json.dumps(v.to_list()) if v is not None else None,
                    return_dtype=pl.String,
                )
                .alias(col)
            )
    csv_ready.write_csv(output)


def read_save_all_combined_tables(folder_name: str) -> dict:
    """Combine and save every extracted table found under *folder_name*.

    Discovers all distinct table base-names (e.g. ``disease_byage_base_data``,
    ``prevalence_base_data``, …) by scanning CSV files recursively under
    ``src/pneu_abm/output/<folder_name>/``, then calls
    :func:`read_extracted_tables` for each one and saves the combined DataFrame
    to ``src/pneu_abm/output/<folder_name>/<table_name>.csv``.

    Parameters
    ----------
    folder_name:
        Sub-folder under ``src/pneu_abm/output/``  (e.g. ``"base_results"``).

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping of table name → combined DataFrame for all tables that were
        successfully processed.
    """
    target_dir = os.path.join(OUTPUT_BASE, folder_name)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Output directory not found: {target_dir}")

    table_names = _discover_table_names(target_dir)
    if not table_names:
        print(f"No extracted CSV tables found under {target_dir}")
        return {}

    print(f"Found {len(table_names)} table(s) under '{folder_name}': {table_names}")

    results = {}
    save_dir = os.path.join(OUTPUT_BASE, folder_name)
    for table_name in table_names:
        try:
            combined = read_extracted_tables(folder_name, table_name)
        except FileNotFoundError as exc:
            print(f"  [skip] {table_name}: {exc}")
            continue

        output = os.path.join(save_dir, f"{table_name}.csv")
        _write_csv_safe(combined, output)
        print(f"  {table_name}: {combined.shape[0]} rows × {combined.shape[1]} cols  -> {output}")
        results[table_name] = combined

    print("\nDone.")
    return results


def main(folder_name: str = None, table_name: str = None, output: str = None) -> None:
    if folder_name is None or table_name is None:
        parser = argparse.ArgumentParser(
            description=(
                "Combine extracted CSV table files into one Polars DataFrame."
            )
        )
        parser.add_argument(
            "folder_name",
            help='Sub-folder under src/pneu_abm/output/ (e.g. "base_results")',
        )
        parser.add_argument(
            "table_name",
            help=(
                'Base table name without year suffix '
                '(e.g. "disease_byage_base_data")'
            ),
        )
        parser.add_argument(
            "--output",
            default=None,
            metavar="PATH",
            help="Optional path to save the combined DataFrame as a CSV.",
        )
        args = parser.parse_args()
        folder_name = args.folder_name
        table_name = args.table_name
        output = args.output

    try:
        combined = read_extracted_tables(folder_name, table_name)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nCombined DataFrame: {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(combined.head())

    # Default save path: output/<folder_name>/<table_name>.csv
    if output is None:
        save_dir = os.path.join(OUTPUT_BASE, folder_name)
        output = os.path.join(save_dir, f"{table_name}.csv")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    _write_csv_safe(combined, output)
    print(f"\nSaved combined DataFrame to: {output}")
    return combined


if __name__ == "__main__":
    combined_data = main(folder_name="base_results", table_name="disease_byage_base_data")
    print(combined_data.head())
    all_data = read_save_all_combined_tables(folder_name="base_results")
    print(list(all_data.keys()))
