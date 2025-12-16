#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:25 2023
@author: ntellioglu
"""

if __name__ == "__main__":
    import sys,os, time
    import json
    import model

    #print(os.getcwd())
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    code_path = os.path.join(repo_path, "code/src")
    code_path = os.path.abspath(os.path.join(repo_path, "code/src"))

    if code_path not in sys.path:
        sys.path.append(code_path)

    os.chdir(os.path.join(repo_path))
    #print(sys.path)

    from run_scenarios.base_params import p
    from utils.param_combo import ParamComboIt
    from run_scenarios.plotting_helper_functions import get_prev_data, plot_base_prev



    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    import polars as pl
    pl.Config.set_tbl_rows(150)
    pl.Config.set_tbl_cols(100)
    #from pathos.multiprocessing import ProcessingPool as Pool
    import parq
    import matplotlib.pyplot as plt

    #def test_worker(i):
    #    print(f"{i}")


    #parq.run(test_worker, [(1,), (2,), (3,)], n_proc=2)

    from run_scenarios.varying_disease_model import new_go_single
    # setup base parameters
    p['prefix'] = 'output/historical_run'
    p['pop_prefix'] = p['prefix']
    p['epi_prefix'] = p['prefix']
    p['overwrite'] = True
    p['years'] = [0,2]
    p['read_population'] = False
    p['save_population'] = True
    #population reading filename used only if read_population == True
    p["pop_group"] = "non_indigenous_varying_trans_v116_year_0"
    #population saving filename used only if save_population == True
    p["pop_saving_address"] = "non_indigenous_varying_trans_v116_year_18_startfrom2002"
    #vaccine rollout details
    p['vaccine_list'] =  "vaccine_configs/vaccine_list.dat"
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
        #print(x['prefix'], x['seed'])
        x["seed_no"] = i
        all_combos.append(x)
    job_inputs = [(i,) for i in all_combos]
    #run a single simulation among combinations
    new_go_single(job_inputs[0][0])

    ###read output
    df, strain_list = get_prev_data( 
        os.path.join(job_inputs[0][0]["prefix"]), 
                'data', "dummy", year = 2002)
    ###### plot output
    #prev plots
    df = (df.with_columns(
        pl.Series("strain_list", [strain_list] * \
               df.height ).alias("strain_names"))
    .explode("strain_names", "strain_list"))


    vaccine_fname = os.path.join('data', 'vaccine_configs/vaccine_list.dat')
    f = open(vaccine_fname)
    vaccines = json.load(f)

    df = df.with_columns(pl.lit("Rest of the serotypes").alias("vaccine_type")).with_columns(
        pl.when(pl.col("strain_names")
                .is_in(vaccines["ppsv23_1"]["serotypes"]))
        .then(pl.lit("23vPPV")).otherwise(pl.col("vaccine_type"))
        .alias("vaccine_type")
        ).with_columns(
            pl.when(pl.col("strain_names")
                    .is_in(vaccines["pcv13_21"]["serotypes"]))
            .then(pl.lit("13vPCV")).otherwise(pl.col("vaccine_type"))
            .alias("vaccine_type")
            ).with_columns(
                pl.when(pl.col("strain_names")
                        .is_in(vaccines["pcv7"]["serotypes"]))
                .then(pl.lit("7vPCV")).otherwise(pl.col("vaccine_type"))
                .alias("vaccine_type")
                )

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
    axes =  plot_base_prev(axes,'data', df_agg3, strain_list, 
                           years = [2002, 2004],ymax = 1.5,
                         pcv_strains = False,overall_prev = False)  
    plt.show()
    #OR run multiple simulations
    #results = parq.run(new_go_single, job_inputs, n_proc=32, results=False)
