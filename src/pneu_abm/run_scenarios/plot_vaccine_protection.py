import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

from pneu_abm.run_scenarios.disease_model import DiseaseModel, make_contact_matrix
from pneu_abm.model.disease.antibody_levels import waning_ratio



# helpers --------------------------------------------------------------------

def _make_disease_obj():
    """Return a DiseaseModel instance initialised with the current base params.

    This is a lightweight helper used by the plotting routines so they can
    access the age-specific protection parameters and sigmoid formulas defined
    on the disease object without having to run an entire simulation.
    """
    # build contact matrix using shared helper from disease_model
    cmatrix = make_contact_matrix(p)

    # temporary HDF5 file path (we never write anything other than the params)
    import tempfile
    fname = tempfile.NamedTemporaryFile(suffix=".h5", delete=False).name

    disease = DiseaseModel(p, cmatrix, fname, mode="w")
    # we only need the data tables; close the file to prevent warnings
    try:
        disease.h5file.close()
    except Exception:
        pass
    return disease


def _log_antibody_range():
    # sensible range for plotting
    return np.linspace(-10, 10, 400)


# plotting functions --------------------------------------------------------


def _get_age_params(age):
    # fetch rows from age-specific protection table for given age
    disease = _make_disease_obj()
    params = disease.age_specific_prot_parameters
    row = params.filter(pl.col("age") == age).to_dicts()[0]
    return row


def _get_vaccine_antibody(vaccine, serotype, dose=1):
    disease = _make_disease_obj()
    vacdf = disease.vaccine_antibody_df
    # exposed_strains column is textual; ensure comparison value is string
    serotype = str(serotype)
    filt = (
        (pl.col("vaccine_type") == vaccine)
        & (pl.col("no_of_doses") == dose)
        & (pl.col("exposed_strains") == serotype)
    )
    matches = vacdf.filter(filt)
    if matches.height == 0:
        raise ValueError(f"no antibody row for {vaccine} dose {dose} serotype {serotype}")
    row = matches.to_dicts()[0]
    return row


def plot_vaccine_protection_against_acq_given_vacc_serotype(
    vaccine: str,
    serotype: str,
    dose: int = 1,
    max_days: int = 365 * 5,
):
    """Show acquisition prob vs days since vaccination for given vaccine/strain.

    Age slider lets you explore age-specific parameters.  Waning is applied
    using the disease model's halflife values.
    """
    disease = _make_disease_obj()
    age_list = disease.age_specific_prot_parameters["age"].unique().to_list()
    ab_row = _get_vaccine_antibody(vaccine, serotype, dose)
    base_meanlog = ab_row["meanlog"]

    days = np.linspace(0, max_days, 400)

    def compute_curve(age_val):
        age_params = _get_age_params(age_val)
        shape = age_params["prob_acq_logantibody_shape"]
        shift = age_params["prob_acq_logantibody_shift"]
        scale = age_params["prob_acq_logantibody_scale"]
        # broadcast inputs so waning_ratio sees same-length sequences
        vaccine_arr = np.full_like(days, vaccine, dtype=object)
        final_time_arr = np.zeros_like(days)
        age_arr = np.full_like(days, age_val)
        waning = waning_ratio(
            vaccine_arr,
            days,
            final_time_arr,
            disease.waning_halflife_day_adult,
            disease.waning_halflife_day_child,
            age_arr,
        )
        antibodies = np.exp(base_meanlog) * waning
        log_ab = np.log(antibodies)
        result = 1 / (1 + scale * np.exp(shape * (log_ab - shift)))
        return result

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    y = compute_curve(age_list[0])
    line, = ax.plot(days, y)
    ax.set_xlabel("days since vaccination")
    ax.set_ylabel("prob_acq")
    ax.set_title(f"{vaccine} / {serotype} age={age_list[0]}")
    # keep the acquisition curve between 0 and 1
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax_age = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_age = Slider(ax_age, "age_idx", 0, len(age_list) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider_age.val)
        a = age_list[idx]
        line.set_ydata(compute_curve(a))
        ax.set_title(f"{vaccine} / {serotype} age={a}")
        fig.canvas.draw_idle()

    slider_age.on_changed(update)
    plt.show()


def plot_vaccine_protection_against_disease_given_vacc_serotype(
    vaccine: str,
    serotype: str,
    dose: int = 1,
    max_days: int = 365 * 5,
):
    """Same as above but using disease probability formula."""
    disease = _make_disease_obj()
    age_list = disease.age_specific_prot_parameters["age"].unique().to_list()
    ab_row = _get_vaccine_antibody(vaccine, serotype, dose)
    base_meanlog = ab_row["meanlog"]

    days = np.linspace(0, max_days, 400)

    def compute_curve(age_val):
        age_params = _get_age_params(age_val)
        add = age_params["prob_dis_logantibody_additive"]
        adj = age_params["prob_dis_logantibody_adjust"]
        shape = age_params["prob_dis_logantibody_shape"]
        scale = age_params["prob_dis_logantibody_scale"]
        age_param = age_params["prob_dis_logantibody_age"]
        vaccine_arr = np.full_like(days, vaccine, dtype=object)
        final_time_arr = np.zeros_like(days)
        age_arr = np.full_like(days, age_val)
        waning = waning_ratio(
            vaccine_arr,
            days,
            final_time_arr,
            disease.waning_halflife_day_adult,
            disease.waning_halflife_day_child,
            age_arr,
        )
        antibodies = np.exp(base_meanlog) * waning
        log_ab = np.log(antibodies)
        denom = np.maximum(
            1,
            1 + age_param * np.exp(shape * (log_ab - add + adj * scale))
        )
        # Numerical guard: disease probability should be non-negative.
        return np.clip(scale / denom, 0.0, None)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    y = compute_curve(age_list[0])
    line, = ax.plot(days, y)
    ax.set_xlabel("days since vaccination")
    ax.set_ylabel("prob_disease")
    ax.set_title(f"{vaccine} / {serotype} age={age_list[0]}")
    max_val = max(np.max(compute_curve(a)) for a in age_list)
    # keep autoscaling; curves are clipped to be non-negative in compute_curve
    ax.grid(True)
    ax_age = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_age = Slider(ax_age, "age_idx", 0, len(age_list) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider_age.val)
        a = age_list[idx]
        line.set_ydata(compute_curve(a))
        ax.set_title(f"{vaccine} / {serotype} age={a}")
        fig.canvas.draw_idle()

    slider_age.on_changed(update)
    plt.show()

def plot_vaccine_protection_against_acq():
    """Interactive plot of acquisition probability as a function of log-antibodies.

    The age-specific parameters are pulled from the base parameter set via a
    temporary ``DiseaseModel`` instance.  A single slider allows the user to
    cycle through the available age groups; the curve updates accordingly.
    """
    disease = _make_disease_obj()
    params = disease.age_specific_prot_parameters
    ages = params["age"].unique().to_list()

    x = _log_antibody_range()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # initial plot using first age value
    def compute_curve(age_value):
        row = params.filter(pl.col("age") == age_value).to_dicts()[0]
        shape = row["prob_acq_logantibody_shape"]
        shift = row["prob_acq_logantibody_shift"]
        scale = row["prob_acq_logantibody_scale"]
        result = 1 / (1 + scale * np.exp(shape * (x - shift)))
        return result

    y = compute_curve(ages[0])
    line, = ax.plot(x, y)
    ax.set_xlabel("log_antibodies")
    ax.set_ylabel("prob_acq")
    ax.set_title(f"Age = {ages[0]}")
    ax.set_ylim(0, 1)
    ax.grid(True)

    ax_age = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_age = Slider(ax_age, "age_idx", 0, len(ages) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider_age.val)
        age_val = ages[idx]
        line.set_ydata(compute_curve(age_val))
        ax.set_title(f"Age = {age_val}")
        fig.canvas.draw_idle()

    slider_age.on_changed(update)
    plt.show()


def plot_vaccine_protection_against_disease():
    """Interactive plot of disease probability as a function of log-antibodies.

    Similar interface to ``plot_vaccine_protection_against_acq``, except that the
    more complex disease sigmoid formula is used.  Slider controls age group.
    """
    disease = _make_disease_obj()
    params = disease.age_specific_prot_parameters
    ages = params["age"].unique().to_list()

    x = _log_antibody_range()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    def compute_curve(age_value):
        row = params.filter(pl.col("age") == age_value).to_dicts()[0]
        add = row["prob_dis_logantibody_additive"]
        adj = row["prob_dis_logantibody_adjust"]
        shape = row["prob_dis_logantibody_shape"]
        scale = row["prob_dis_logantibody_scale"]
        age_param = row["prob_dis_logantibody_age"]
        # formula translated from Disease.calc_prob_dis_givenlog_antibodies
        result = (
            scale
            / (
                1
                + age_param
                * np.exp(shape * (x - add + adj * scale))
            )
        )
        return result

    y = compute_curve(ages[0])
    line, = ax.plot(x, y)
    ax.set_xlabel("log_antibodies")
    ax.set_ylabel("prob_disease")
    ax.set_title(f"Age = {ages[0]}")
    # determine max value across all age curves to scale y-axis
    max_val = max(np.max(compute_curve(age_val)) for age_val in ages)
    ax.set_ylim(0, max_val * 1.05)
    ax.grid(True)

    ax_age = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_age = Slider(ax_age, "age_idx", 0, len(ages) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider_age.val)
        age_val = ages[idx]
        line.set_ydata(compute_curve(age_val))
        ax.set_title(f"Age = {age_val}")
        fig.canvas.draw_idle()

    slider_age.on_changed(update)
    plt.show()
    
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Configure a single exemplar scenario run
    # -----------------------------------------------------------------------------
    # NOTE: `p` is a mutable dict imported from base params; edits here mutate it.
    # Output directories/prefix (relative to repo root)
    p["prefix"] = "src/pneu_abm/output/historical_run1"
    p["pop_prefix"] = p["prefix"]  # where population checkpoints are saved/read
    p["epi_prefix"] = p["prefix"]  # where epidemiology inputs are kept
    p["overwrite"] = True  # overwrite existing outputs at the same prefix
    p["years"] = [0, 18]  # simulate from year 2002 + 0 up to (and incl.) 2002 + 4

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

    # -------------------------------------------------------------
    # show interactive vaccine protection curves before sim
    # -------------------------------------------------------------
    #plot_vaccine_protection_against_acq()
    #plot_vaccine_protection_against_disease()

    # hit the sigmoids for PPSV23 protecting against strain “17F” (first dose)
    plot_vaccine_protection_against_acq_given_vacc_serotype(
        vaccine="pcv7_30",    # vaccine name from vaccine_antibody_df
        serotype="17F",      # serotype code in the same table
        dose=1,              # optional, defaults to 1
        max_days=365 * 15,    # e.g. show three years of waning
    )

    plot_vaccine_protection_against_disease_given_vacc_serotype(
        vaccine="pcv7_30",
        serotype="17F",
        dose=1,
        max_days=365 * 15,
    )
