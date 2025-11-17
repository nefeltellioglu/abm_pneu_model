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

    from varying_transmission.base_params import p
    from param_combo import ParamComboIt




    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    import polars as pl
    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(100)
    #from pathos.multiprocessing import ProcessingPool as Pool
    import parq

    #def test_worker(i):
    #    print(f"{i}")


    #parq.run(test_worker, [(1,), (2,), (3,)], n_proc=2)

    from varying_transmission.varying_disease_model import new_go_single
    # setup base parameters
    p['prefix'] = 'output/varying_trans_base_updated_ppv23'
    p['prefix'] = 'output/varying_trans_base_recheck'
    p['prefix'] = 'output/varying_trans_base_no_vaccine'
    p['prefix'] = 'output/varying_trans_base_manu'
    p['pop_prefix'] = p['prefix']
    p['epi_prefix'] = p['prefix']
    p['overwrite'] = True
    p['years'] = [0,18]
    p['read_population'] = False
    p['save_population'] = False
    p['num_runs'] = 100
    p['vaccine_list'] = "vaccine_list.dat"
    p["transmission_coefficient"] = 0.0545
    
    """p["prob_dis_logantibody_scale"] =  [i * 1.1 if x >= 9
                                  else (i) for x,i in enumerate(
                               p["prob_dis_logantibody_scale"])] """
    #max prob of developing disease
    #p["prob_dis_logantibody_scale"] = [9.9049944e-05, 
    #     3.7083228e-05, 3.675504e-05, 6.6781155e-05, 1.4900139449999997e-05, 
    #                                1.514037e-06, 1.1855445e-06, 
    #                                6.447074117647058e-07, 
    #                                6.491453181818181e-06, 5.12299432e-05]
    p1 = p.copy()
    # (basic usage) run simulation
    # sweep parameters
    """
    child_v_names = ["child_pcv13","child_pcv15","child_pcv20"]
    adult_v_names = [#"adult_novacc", 
                     #"adult_ppsv23", 
                     #"adult_pcv13", 
                     "adult_pcv15", "adult_pcv20"]
    v_combos = [[
        "vaccine_list_child_pcv13_%s.dat"%v,
        "vaccine_list_child_pcv15_%s.dat"%v,
        "vaccine_list_child_pcv20_%s.dat"%v] for v in adult_v_names]
    flat_v_combos = [item for sublist in v_combos for item in sublist] 
    """
    
    
    
    values = [#{"pcv7": 1.009,"pcv13": 1.0063, "ppv23": 0.996, "nonppv23": 0.996},
              #{"pcv7": 1.009,"pcv13": 1.0063, "ppv23": 0.9958, "nonppv23": 0.9958},
              #{"pcv7": 1.009,"pcv13": 1.0063, "ppv23": 0.9955, "nonppv23": 0.9955},
              {"pcv7": 1.009,"pcv13": 1.0063, "ppv23": 0.995, "nonppv23": 0.995},
                         ]
    #"pcv7": 1.009,"pcv13": 1.0063 set ppv above 1 & nonppv23 less than  0.994 maybe 0.993 or 0.9935
    
    
    sweep_params = [
        {'name': 'transmission_coefficient_multipliers', 'values': values},
        #{'name': 'transmission_coefficient', 'values': [0.0545]},
        {'name': 'prob_acq_logantibody_scale', 'values': [5]},
        {'name': 'vaccine_list', 'values': [#"vaccine_list.dat",
                                            "no_vaccine_list.dat"
                                            ]},
        
        ]
    # number of different seeds to use for each parameter combination
    
    # generate parameter combinations (converting iterator to list)
    param_combos = list(ParamComboIt(p, sweep_params))

    # just for info, print out all prefixes (these will be used as output directories)
    all_combos = []
    for i,x in enumerate(param_combos):
        print(x['prefix'], x['seed'])
        x["seed_no"] = i
        if i == 0:
            x['save_population'] = False
        else:
            x['save_population'] = False
        all_combos.append(x)

    # combo index is passed in as (only) argument
    #combo_num = int(sys.argv[1])
    #cur_params = all_combos[0]

    #new_go_single(cur_params)
    #seeds=[item['seed'] for item in all_combos]   
    #for param in param_combos:
    #    go_single(param, DiseaseModel, 
    #              KnownContactMatrix(givenmatrix = knowncmatrix), 
    #                          param['seed'], verbose=True)
    job_inputs = [(i,) for i in all_combos]
    # Run these 10 jobs using 4 processes.

    #(Pool(7).map(go_single, all_combos, repeat(DiseaseModel),
    #                repeat(KnownContactMatrix(givenmatrix = knowncmatrix)),
    #                                    seeds))
    #for i in range(2):
    new_go_single(job_inputs[0][0])
    #results = parq.run(new_go_single, job_inputs, n_proc=32, results=False)
    
    
    
