#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:29:12 2023

@author: ntellioglu
"""

p = {
    # directories
    'resource_prefix': 'data/',
    'prefix': 'output/',

    # demographic model parameters
    'age_distribution': 'population/age_dist_2002.dat',
    'initial_death_rates': 'population/death_rates_2002.dat',
    'later_death_rates': None,#'death_rates_2008.dat',
    'year_to_change_death_rates' : 6,
    'birth_rates': 'population/birth_rates.dat', 
    'mig_rates': 'population/migration_rates.dat',
    'age_distribution_mig': 'population/age_dist_migration_2018.dat',
    
    
    #disease parameters
    'strain_list': 'disease/strain_list.dat',
    'vaccine_list': 'vaccine_configs/vaccine_list.dat',
    'initial_cases': 'disease/initialize_infections_2002_updated_v4.dat',
    'duration_of_infection': 30, #days, not used
    'age_specific_duration_of_infections': [72,28,18,17], #days
    

    'transmission_coefficient': 0.0545, #
    'reduction_in_susceptibility_coinfections': 0.8, #
    'max_no_coinfections': 2,
    'expernal_exposure_prob': 0.0006,
    'external_exposure_check_per_year': 4,
    'no_daily_external_strains':2,
    'halt': True,
    'noise_in_strain_distribution': 0.0,
    "prob_acq_logantibody_scale": 7,
    "prob_acq_logantibody_shift": 4.1,
    "prob_acq_logantibody_shape": 0.5,
    
    
    #waning halflife days
    "waning_halflife_day_adult": 600,
    "waning_halflife_day_child": 125,
    
    #serotype specific multipliers
    'transmission_coefficient_multipliers':{"pcv7": 1.009,"pcv13": 1.0063, 
                                        "ppv23": 0.995, "nonppv23": 0.993}, 
    
    'disease_outcome_multipliers': {"v116_only": 7,
                                    "nvt": 0.6 ,
                                    "ppv23_nonpcv13": 3},
    
    
    #disease outcome parameters
    "prob_dis_logantibody_additive": -1.7,
    "prob_dis_logantibody_adjust": 1250,
    "prob_dis_logantibody_shift": 0,
    "prob_dis_logantibody_shape": 1.5, 
    
    "prob_dis_logantibody_age":  [1.6961140331386761,
                                     2.2651749734408235,
                                     2.285400859310723,
                                     1.5722998501598242,
                                     1.6207902000541345,
                                     6.935101321830312,
                                     8.856689900716507,
                                     3.8321076975547315,
                                     1.4704649757299755,
                                     0.2967014806293988,
                                     0.0767014806293988,
                                     0.0017014806293988], 
    
   
    "prob_dis_logantibody_scale" :[0.00016979990399999998,
                                     6.11873262e-05,
                                     6.381083333333333e-05,
                                     7.47948936e-05,
                                     5.302256981538461e-05,
                                     2.523395e-05,
                                     1.7387986000000002e-05,
                                     5.620722950980392e-05,
                                     0.0001160599560502283,
                                     0.0005979725967741935,
                                     0.001784545179087875,
                                     0.0037070518294051637],
    
                                                
    "ipd_fraction_by_age_group": [0.3142857142857143,
                                    0.34146341463414637,
                                    0.3684210526315789,
                                    0.27903225806451615,
                                    0.2373912131697599,
                                    0.17261029411764706,
                                    0.12693631669535285,
                                    0.13626534598214285,
                                    0.11677370706466626,
                                    0.073016477,
                                    0.050870022,
                                    0.041065923],
    
    "prob_dis_at_risk_tier1_multipliers" : [10.52,10.52,5.6,5.6,5.6],
    #[10.52,10.52,7.1,7.1,7.1],
    "prob_dis_at_risk_tier2_multipliers" : [1,1,1.2,1.2,1.2],
    "prob_dis_at_risk_no_risk_multipliers" : \
                                         [0.8461,0.8461,0.31,0.31,0.31],
                                         
    
    "at_risk_age_groups": [[5,14],[5,17],[18,34],[35,49],[50,69]],
    "at_risk_age_groups_str": ["5–14","15–17","18–34","35–49","50–69"],
    "at_risk_vaccine_target_group": [1,2],
    
    'pop_size': 1_000_000,
    'update_demog': True,
    #'demo_burn': 1,
    'num_runs': 100,
    'random_seed': False,
    'seed': 1234,
    'seed_no': 0,
    't_per_year': 52,
    'logging': False,
    'years': [10,18],
    'record_interval': 1,
    'save_population': True,
    "read_population": True,
    "pop_group": "non_indigenous_at_risk_varying_trans",
    "pop_saving_address": "non_indigenous_at_risk_varying_trans",
}
