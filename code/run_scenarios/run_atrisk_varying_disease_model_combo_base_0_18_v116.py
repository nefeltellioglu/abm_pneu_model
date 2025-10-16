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
    repo_path = "../.."
    code_path = os.path.join(repo_path, "code")
    code_path = os.path.abspath(os.path.join(repo_path, "code"))

    if code_path not in sys.path:
        sys.path.append(code_path)

    os.chdir(os.path.join(repo_path))
    #print(sys.path)

    from run_scenarios.base_params import p
    from utils.param_combo import ParamComboIt




    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    import polars as pl
    pl.Config.set_tbl_rows(150)
    pl.Config.set_tbl_cols(100)
    #from pathos.multiprocessing import ProcessingPool as Pool
    import parq

    #def test_worker(i):
    #    print(f"{i}")


    #parq.run(test_worker, [(1,), (2,), (3,)], n_proc=2)

    from run_scenarios.at_risk_varying_disease_model import new_go_single
    # setup base parameters
    p['prefix'] = 'output/historical_run'
    p['pop_prefix'] = p['prefix']
    p['epi_prefix'] = p['prefix']
    p['overwrite'] = True
    p['years'] = [0,18]
    p['read_population'] = False
    p['save_population'] = False
    p["pop_group"] = "non_indigenous_varying_trans_v116_year_0"
    p["pop_saving_address"] = "non_indigenous_varying_trans_v116_year_18_startfrom2002"
    p['vaccine_list'] =  "vaccine_configs/vaccine_list.dat"
    
    p1 = p.copy()
    # (basic usage) run simulation
    # sweep parameters
    sweep_params = [
        {'name': 'save_population', 'values': ["False"]},
        ]
    # number of different seeds to use for each parameter combination
    
    # generate parameter combinations (converting iterator to list)
    param_combos = list(ParamComboIt(p, sweep_params))

    # just for info, print out all prefixes (these will be used as output directories)
    all_combos = []
    for i,x in enumerate(param_combos):
        print(x['prefix'], x['seed'])
        x["seed_no"] = i
        #x['save_population'] = False
        if i == 0:
            x['save_population'] = False
        else:
            x['save_population'] = False
        all_combos.append(x)

    
    job_inputs = [(i,) for i in all_combos]
    
    #run a single simulation among combinations
    new_go_single(job_inputs[0][0])
    
    #OR run multiple simulations
    #results = parq.run(new_go_single, job_inputs, n_proc=32, results=False)
    
    
    
