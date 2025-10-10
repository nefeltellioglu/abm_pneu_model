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
    import numpy as np
    import polars as pl
    import parq
    
    np.set_printoptions(threshold=sys.maxsize)
    pl.Config.set_tbl_rows(150)
    pl.Config.set_tbl_cols(100)
    
    #print(os.getcwd())
    repo_path = "../.."
    code_path = os.path.join(repo_path, "code")
    code_path = os.path.abspath(os.path.join(repo_path, "code"))

    if code_path not in sys.path:
        sys.path.append(code_path)

    os.chdir(os.path.join(repo_path))
    
    from run_scenarios.at_risk_varying_disease_model import new_go_single
    from run_scenarios.base_params import p
    from utils.param_combo import ParamComboIt

    
    p['prefix'] = 'output/example'
    p['pop_prefix'] = p['prefix']
    p['epi_prefix'] = p['prefix']
    p['overwrite'] = True
    p['years'] = [18,30]
    p['read_population'] = True
    p['save_population'] = False
    p['num_runs'] = 100
    
    p["pop_group"] = "non_indigenous_varying_trans_v116_year_18_updated_202403242"
    p["pop_saving_address"] = "non_indigenous_varying_trans_atrisk_v116_year_30"             
            
    p1 = p.copy()
    # (basic usage) run simulation
    
    atrisk_v_names = [#"atrisk_v116",
                     #"atrisk_pcv13_ppv23_ppv23",
                     #"atrisk_pcv20",
                     "atrisk_novacc",
                     #"atrisk_pcv20_ppv23_ppv23",
                     #"atrisk_pcv15_ppv23_ppv23",
                     ]
    
    at_risk_vaccine_target_groups = [[1]]
    uptake_percentages = [50]
    adult_vacc_age = 65

    all_v_combos = []
    for uptake_perc in uptake_percentages:
        for atrisk_v in atrisk_v_names:
            if atrisk_v == "atrisk_v116":
                v_combos = [[
            "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_v116_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
            "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_pcv20_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
            
             ]]
            elif atrisk_v == "atrisk_pcv13_ppv23_ppv23":
                v_combos = [[
            "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_pcv13_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
            
             ]]
            elif atrisk_v == "atrisk_pcv15_ppv23_ppv23":
                v_combos = [[
            "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_pcv15_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
            "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_v116_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
            
             ]]
            elif atrisk_v == "atrisk_pcv20_ppv23_ppv23":
                 v_combos = [[
             "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_pcv20_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
             "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_v116_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
             
              ]]
            elif atrisk_v == "atrisk_pcv20":
                 v_combos = [[
             "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_v116_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
             
              ]]
            elif atrisk_v == "atrisk_novacc":
                 v_combos = [[
             "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_pcv13_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
             "vaccine_configs/at_risk_adult_vacc_age_%s_v116_%s_vacc_uptake/vaccine_list_child_pcv20_adult_v116_%s.dat"%(adult_vacc_age,uptake_perc, atrisk_v),
             
              ]]
            
            all_v_combos = all_v_combos + v_combos
    flat_v_combos = [item for sublist in all_v_combos for item in sublist] 
    
    sweep_params = [
        {'name': 'vaccine_list', 'values': flat_v_combos},
        {'name': 'at_risk_vaccine_target_group', 
         'values': at_risk_vaccine_target_groups},
        ]
    
    # generate parameter combinations (converting iterator to list)
    param_combos = list(ParamComboIt(p, sweep_params))

    # just for info, print out all prefixes (these will be used as output directories)
    all_combos = []
    for i,x in enumerate(param_combos):
        print(x['prefix'], x['seed'])
        x["seed_no"] = i
        all_combos.append(x)
    

    job_inputs = [(i,) for i in all_combos]
    
    #run a single simulation among combinations
    new_go_single(job_inputs[0][0])
    
    #OR run multiple simulations
    #results = parq.run(new_go_single, job_inputs, n_proc=32, results=False)
    
    
    
