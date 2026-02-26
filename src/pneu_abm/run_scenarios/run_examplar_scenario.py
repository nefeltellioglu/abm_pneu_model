#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:25 2023
@author: ntellioglu
"""
import os
import sys, time
import json

# Resolve the repository root (three levels up from this script).
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Ensure the package source tree is importable when running this file directly.
code_path = os.path.abspath(os.path.join(repo_path, "src"))
if code_path not in sys.path:
    sys.path.append(code_path)

# Make paths below deterministic regardless of where the script is invoked from.
os.chdir(repo_path)

import pneu_abm  # noqa: F401  # Imported to ensure package is discoverable / for side-effects if any.
from pneu_abm.data.scenario_configs.base_params import p  # Base scenario parameters (mutable dict)
from pneu_abm.utils.param_combo import ParamComboIt  # Helper to generate parameter combinations for sweeps
from pneu_abm.run_scenarios.plotting_helper_functions import (
    get_prev_data,
    plot_base_prev,
)  # Read prevalence output + plot helper
from pneu_abm.model.disease.disease_utils import looks_like_json_list  # Func to detect JSON-encoded lists


import numpy as np

# Avoid truncating large arrays when printing (useful for debugging).
np.set_printoptions(threshold=sys.maxsize)

import polars as pl

# Configure Polars table display for interactive inspection.
pl.Config.set_tbl_rows(150)
pl.Config.set_tbl_cols(100)

import parq  # Parallel runner used for multi-run batches (see commented usage below)
import matplotlib.pyplot as plt
import time


from pneu_abm.run_scenarios.disease_model import new_go_single

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Configure a single exemplar scenario run
    # -----------------------------------------------------------------------------
    # NOTE: `p` is a mutable dict imported from base params; edits here mutate it.
    # Output directories/prefix (relative to repo root)
    p["prefix"] = "src/pneu_abm/output/historical_run"
    p["pop_prefix"] = p["prefix"]  # where population checkpoints are saved/read
    p["epi_prefix"] = p["prefix"]  # where epidemiology inputs are kept
    p["overwrite"] = True  # overwrite existing outputs at the same prefix
    p["years"] = [0, 4]  # simulate from year 2002 + 0 up to (and incl.) 2002 + 4

    # Population checkpointing controls
    p["read_population"] = False  # start from initial population if False
    p["save_population"] = True  # persist end-of-run population if True

    # Population reading filename used only if read_population == True
    p["pop_reading_address"] = (
        "saved_checkpoints/non_indigenous_varying_trans_year_{p['years'][0]}"
    )

    # Population saving filename used only if save_population == True
    p["pop_saving_address"] = (
        f"saved_checkpoints/non_indigenous_varying_trans_year_{p['years'][1]}"
    )

    # Vaccine rollout details
    p["vaccine_list"] = "vaccine_configs/vaccine_list.csv"

    # Intended number of runs for batch mode; this exemplar script runs one by default.
    p["num_runs"] = 100

    # Keep a snapshot copy if you want to compare/restore later (currently unused).
    p1 = p.copy()

    # -----------------------------------------------------------------------------
    # Parameter sweep configuration
    # -----------------------------------------------------------------------------
    # `sweep_params` generates parameter combinations on top of the existing `p`.
    # Multiple parameters and multiple values per parameter can be supplied.
    sweep_params = [{"name": "save_population", "values": ["True"]}]

    # Generate parameter combinations (convert iterator to list so we can index/reuse).
    param_combos = list(ParamComboIt(p, sweep_params))

    # Materialize combos and attach a deterministic seed index.
    all_combos = []
    for i, x in enumerate(param_combos):
        x["seed_no"] = i
        all_combos.append(x)

    # `parq.run` expects an iterable of tuple-args, hence wrapping dicts in `(dict,)`.
    job_inputs = [(i,) for i in all_combos]

    # Select a single combination to run (the first one).
    sel_input = job_inputs[0][0]
    
    # Explicit RNG seeds for different stochastic components of the model.
    sel_input["pop_seed"] = 1234
    sel_input["transmission_seed"] = 123
    sel_input["disease_seed"] = 1234
    sel_input["vaccine_seed"] = 1234

    # Example subset for batch runs (first 8 combos).
    sel_inputs = [(i,) for i in all_combos[:8]]

    start = time.time()

    # Run a single simulation among combinations.
    new_go_single(sel_input)

    # OR: run multiple simulations in parallel (uncomment to use)
    # results = parq.run(new_go_single, sel_inputs, n_proc=4, results=False)

    end = time.time()
    print(f"Simulation time: {end - start:.4f} seconds")

    # -----------------------------------------------------------------------------
    # Read model output and prepare prevalence plots
    # -----------------------------------------------------------------------------
    # Read prevalence output at the selected prefix. `year=2002` sets the calendar baseline.
    df, strain_list = get_prev_data(
        os.path.join(sel_input["prefix"]),
        "src/pneu_abm/data",
        "dummy",
        year=2002,
    )

    # Expand the per-run list columns so each row corresponds to a single strain/serotype.
    df = (
        df.with_columns(
            pl.Series("strain_list", [strain_list] * df.height).alias("strain_names")
        )
        .explode("strain_names", "strain_list")
    )


    # Load vaccine config; some columns may contain JSON-encoded lists.
    vaccine_fname = os.path.join("src/pneu_abm/data", p["vaccine_list"])
    vacdf = pl.read_csv(vaccine_fname)

    # Infer which columns are JSON list strings using the first row as a schema sample.
    sample = vacdf.head(1).to_dicts()[0]
    LIST_FIELDS = [col for col, val in sample.items() if looks_like_json_list(val)]

    # Decode JSON list columns into proper Python lists.
    for col in LIST_FIELDS:
        s = vacdf[col].map_elements(lambda x: json.loads(x) if isinstance(x, str) and x else x)
        vacdf = vacdf.with_columns(pl.Series(col, s))

    # Convert to dict-of-dicts keyed by vaccine "name" for easier downstream usage.
    vaccines = {
        row["name"]: {k: v for k, v in row.items() if k != "name"}
        for row in vacdf.to_dicts()
    }
    # Load strain list / serotype metadata (used to map strains to vaccine groups).
    strain_fname = os.path.join(p["resource_prefix"], p["strain_list"])
    strains = pl.read_csv(
        strain_fname,
        comment_prefix="#",
        has_header=True,
    )

    # For each vaccine, derive the set of covered serotypes from the strain table.
    for vaccine, value in vaccines.items():
        vaccines[vaccine]["serotypes"] = pl.Series(
            "serotypes",
            values=sorted(
                set(
                    strains.filter(pl.col(vaccines[vaccine]["vaccine_given"]))[
                        "serotype"
                    ]
                )
            ),
        )
    
    # Annotate each strain with a "vaccine_type" for grouped plotting.
    df = (
        df.with_columns(pl.lit("Rest of the serotypes").alias("vaccine_type"))
        .with_columns(
            pl.when(
                pl.col("strain_names").is_in(
                    sorted(set(strains.filter(pl.col("23vPPV"))["serotype"]))
                )
            )
            .then(pl.lit("23vPPV"))
            .otherwise(pl.col("vaccine_type"))
            .alias("vaccine_type")
        )
        .with_columns(
            pl.when(
                pl.col("strain_names").is_in(
                    sorted(set(strains.filter(pl.col("13vPCV"))["serotype"]))
                )
            )
            .then(pl.lit("13vPCV"))
            .otherwise(pl.col("vaccine_type"))
            .alias("vaccine_type")
        )
        .with_columns(
            pl.when(
                pl.col("strain_names").is_in(
                    sorted(set(strains.filter(pl.col("7vPCV"))["serotype"]))
                )
            )
            .then(pl.lit("7vPCV"))
            .otherwise(pl.col("vaccine_type"))
            .alias("vaccine_type")
        )
    )

    # Aggregate prevalence by vaccine groups (and also keep per-strain aggregation for plots).
    df_agg = (
        df.group_by("vaccine_type", "t")
        .agg(
            prev_mean=pl.col("strain_list").mean(),
            prev_min=pl.col("strain_list").quantile(0.025),
            prev_max=pl.col("strain_list").quantile(0.975),
        )
        .rename({"vaccine_type": "strain_names"})
    )

    df_agg3 = df.group_by("strain_names", "t").agg(
        vaccine_type=pl.col("vaccine_type").first(),
        prev_mean=pl.col("strain_list").mean(),
        prev_min=pl.col("strain_list").quantile(0.025),
        prev_max=pl.col("strain_list").quantile(0.975),
    )

    # Plot prevalence trajectories
    figsize = (7, 3.5)  # inches (width, height)
    fig, axes = plt.subplots(figsize=figsize)

    axes = plot_base_prev(
        axes,
        os.path.join(os.getcwd(), "data"),
        df_agg3,
        strain_list,
        years=[2002, 2002 + p["years"][1]],
        ymax=1.5,
        pcv_strains=False,
        overall_prev=False,
        t_per_year=p["t_per_year"],
        vaccines=vaccines,
        strains=strains,
    )
    plt.show()

    # OR: run multiple simulations
    # results = parq.run(new_go_single, job_inputs, n_proc=32, results=False)
