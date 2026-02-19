#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:25 2023

@author: ntellioglu
"""
import sys,os

#print(os.getcwd())
repo_path = "../.."
code_path = os.path.join(repo_path, "code")
code_path = os.path.abspath(os.path.join(repo_path, "code"))

if code_path not in sys.path:
    sys.path.append(code_path)

os.chdir(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
print(os.getcwd())
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import polars as pl
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)
import parq


import pneu_abm
from pneu_abm.data.scenario_configs.base_params import p
from pneu_abm.utils.param_combo import ParamComboIt
from pneu_abm.run_scenarios.disease_model import new_go_single


if __name__ == "__main__":   

    # setup base parameters
    p['prefix'] = 'src/pneu_abm/output/base_results'
    p['pop_prefix'] = p['prefix']
    p['epi_prefix'] = p['prefix']
    p['overwrite'] = True
    p['years'] = [0,1]
    p['read_population'] = False
    p['save_population'] = True
    p['num_runs'] = 2
    
    p1 = p.copy()
     
    sweep_params = [
        {'name': 'vaccine_list', 'values': ["vaccine_configs/vaccine_list.csv",
                                            "vaccine_configs/examplar_all_vaccine_list.csv"]},]
    
    # generate parameter combinations (converting iterator to list)
    param_combos = list(ParamComboIt(p, sweep_params))

    # just for info, print out all prefixes (these will be used as output directories)
    all_combos = []
    for i,x in enumerate(param_combos):
        print(x['prefix'])
        x["seed_no"] = i
        all_combos.append(x)

    
    job_inputs = [(i,) for i in all_combos]
    results = parq.run(new_go_single, job_inputs, n_proc=4, results=False)
    
    
    
