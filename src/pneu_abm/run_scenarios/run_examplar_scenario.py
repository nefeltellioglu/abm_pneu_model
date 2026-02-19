#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:25 2023
@author: ntellioglu
"""
import os
import sys,time
import json

#print(os.getcwd())
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
code_path = os.path.join(repo_path, "src")
code_path = os.path.abspath(os.path.join(repo_path, "src"))

if code_path not in sys.path:
    sys.path.append(code_path)

os.chdir(os.path.join(repo_path))
#print(sys.path)
import pneu_abm
from pneu_abm.data.scenario_configs.base_params import p
from pneu_abm.utils.param_combo import ParamComboIt
from pneu_abm.run_scenarios.plotting_helper_functions import get_prev_data, plot_base_prev
from pneu_abm.model.disease.disease_utils import looks_like_json_list


import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import polars as pl
pl.Config.set_tbl_rows(150)
pl.Config.set_tbl_cols(100)
#from pathos.multiprocessing import ProcessingPool as Pool
import parq
import matplotlib.pyplot as plt
import time


from pneu_abm.run_scenarios.disease_model import new_go_single

if __name__ == "__main__":

    # setup base parameters
    p['prefix'] = 'src/pneu_abm/output/historical_run'
    p['pop_prefix'] = p['prefix']
    p['epi_prefix'] = p['prefix']
    p['overwrite'] = True
    p['years'] = [0,4]
    p['read_population'] = False
    p['save_population'] = True
    #population reading filename used only if read_population == True
    p["pop_reading_address"] = "saved_checkpoints/non_indigenous_varying_trans_year_{p['years'][0]}"
    #population saving filename used only if save_population == True
    p["pop_saving_address"] = f"saved_checkpoints/non_indigenous_varying_trans_year_{p['years'][1]}"
    #vaccine rollout details
    p['vaccine_list'] =  "vaccine_configs/vaccine_list.csv"
    p['num_runs'] = 100
    p1 = p.copy()
    # sweep parameters used to generate parameter combinations on top of existing p parameter set
    # multiple parameters and multiple parameeter values can be given here to generate parameter combinations
    sweep_params = [{'name': 'save_population', 'values': ["True"]},]

    # generate parameter combinations (converting iterator to list)
    param_combos = list(ParamComboIt(p, sweep_params))

    # just for info, print out all prefixes (these will be used as output directories)
    all_combos = []
    for i,x in enumerate(param_combos):
        x["seed_no"] = i
        all_combos.append(x)
    job_inputs = [(i,) for i in all_combos]
    sel_input = job_inputs[0][0]
    sel_input["pop_seed"] = 1234
    sel_input["transmission_seed"] = 123
    sel_input["disease_seed"] = 1234
    sel_input["vaccine_seed"] = 1234
    sel_inputs = [(i,) for i in all_combos[:8]]
    
    start = time.time()
    #run a single simulation among combinations
    new_go_single(sel_input)
    #results = parq.run(new_go_single, sel_inputs, n_proc=4, results=False)
    
    end = time.time()

    print(f"Simulation time: {end - start:.4f} seconds")

    ###read output
    df, strain_list = get_prev_data( 
        os.path.join(sel_input["prefix"]), 
                'src/pneu_abm/data', "dummy", year = 2002)
    ###### plot output
    #prev plots
    df = (df.with_columns(
        pl.Series("strain_list", [strain_list] * \
               df.height).alias("strain_names"))
    .explode("strain_names", "strain_list"))


    vaccine_fname = os.path.join('src/pneu_abm/data',
                                 p['vaccine_list'])
    vacdf = pl.read_csv(vaccine_fname)
    # Infer list fields from the first row
    sample = vacdf.head(1).to_dicts()[0]
    
    LIST_FIELDS = [col for col, val in sample.items() if 
                   looks_like_json_list(val)]
    # Decode JSON list columns
    for col in LIST_FIELDS:
        s = vacdf[col].map_elements(
            lambda x: json.loads(x) if isinstance(x, str) and x else x
        )
        vacdf = vacdf.with_columns(pl.Series(col, s))
    # Convert to dict-of-dicts keyed by "name"
    vaccines = {
        row["name"]: {k: v for k, v in row.items() if k != "name"}
        for row in vacdf.to_dicts()
    }
    strain_fname = os.path.join(p['resource_prefix'],
                                            p['strain_list'])
    strains = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    
    for vaccine, value in vaccines.items():
        vaccines[vaccine]["serotypes"] = \
            pl.Series("serotypes", values = 
                      sorted(set(strains
                .filter(pl.col(vaccines[vaccine]["vaccine_given"])
                        )['serotype'])))
    
    df = (df.with_columns(pl.lit("Rest of the serotypes")
                          .alias("vaccine_type"))
        .with_columns(
        pl.when(pl.col("strain_names")
                .is_in(sorted(set(strains.filter(pl.col("23vPPV"))['serotype']))))
        .then(pl.lit("23vPPV")).otherwise(pl.col("vaccine_type"))
        .alias("vaccine_type")
        ).with_columns(
            pl.when(pl.col("strain_names")
                    .is_in(sorted(set(strains.filter(pl.col("13vPCV"))['serotype']))))
            .then(pl.lit("13vPCV")).otherwise(pl.col("vaccine_type"))
            .alias("vaccine_type")
            ).with_columns(
                pl.when(pl.col("strain_names")
                        .is_in(sorted(set(strains.filter(pl.col("7vPCV"))['serotype']))))
                .then(pl.lit("7vPCV")).otherwise(pl.col("vaccine_type"))
                .alias("vaccine_type")
                ))

    df_agg = df.group_by("vaccine_type","t").agg(
        prev_mean = pl.col("strain_list").mean(),
        prev_min = pl.col("strain_list").quantile(0.025),
        prev_max = pl.col("strain_list").quantile(0.975),).rename(
            {"vaccine_type":"strain_names"})

    df_agg3 = df.group_by("strain_names","t").agg(
        vaccine_type = pl.col("vaccine_type").first(),
        prev_mean = pl.col("strain_list").mean(),
        prev_min = pl.col("strain_list").quantile(0.025),
        prev_max = pl.col("strain_list").quantile(0.975),)
    figsize=(7,3.5)#no in x axis, no in yaxis
    #%matplotlib inline
    fig, axes = plt.subplots(figsize=figsize)
    axes =  plot_base_prev(axes,
                           os.path.join(os.getcwd(), 'data'), 
                           df_agg3, strain_list, 
                           years = [2002, 2002 + p['years'][1]],ymax = 1.5,
                         pcv_strains = False,overall_prev = False, 
                         t_per_year= p["t_per_year"],
                         vaccines = vaccines, strains = strains)  
    plt.show()
    #OR run multiple simulations
    #results = parq.run(new_go_single, job_inputs, n_proc=32, results=False)
