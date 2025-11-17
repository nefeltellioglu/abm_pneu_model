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
    'age_distribution': 'age_dist_2002.dat',
    'initial_death_rates': 'death_rates_2002.dat',
    'later_death_rates': None,#'death_rates_2008.dat',
    'year_to_change_death_rates' : 6,
    'birth_rates': 'birth_rates.dat', 
    'mig_rates': 'migration_rates.dat',
    'age_distribution_mig': 'age_dist_migration_2018.dat',
    
    
    #disease parameters
    'strain_list': 'strain_list.dat',
    'vaccine_list': 'vaccine_list.dat',
     #'vaccine_list_child_pcv15_adult_pcv13.dat',
    'initial_cases': 'initialize_infections.dat',
    'duration_of_infection': 30, #days, not used
    'age_specific_duration_of_infections': [72,28,18,17], #days
    

    'transmission_coefficient': 0.0551, #
    'reduction_in_susceptibility_coinfections': 0.8, #
    'max_no_coinfections': 2,
    'expernal_exposure_prob': 0.0006,
    'external_exposure_check_per_year': 4,
    'no_daily_external_strains':2,
    'halt': True,
    'noise_in_strain_distribution': 0.0,
    "prob_acq_logantibody_scale": 0.05,#0.001,
    "prob_acq_logantibody_shift": -2.2,#-5.5, #fixed
    "prob_acq_logantibody_shape": 1,#5, #fixed
    #waning halflife days
    "waning_halflife_day_adult": 600,
    "waning_halflife_day_child": 125,
    
    'transmission_coefficient_multipliers': {"pcv7": 1.009,
                                              "pcv13": 1.006, 
                                              "ppv23": 0.997, 
                                              "nonppv23": 0.995},
    
    
    #disease outcome parameters
    "prob_dis_logantibody_age_10": [1 / (i * 1000) for i in [0.000589583,
                                0.000441467,
                                0.00043756,
                                0.000636011,
                                0.000616983,
                                0.000144194,
                                0.000112909,
                                0.000260953,
                                0.000680057,
                              0.003370391]], #-> this will be the age component
    "prob_dis_logantibody_shift": 0, #fixed
    "prob_dis_logantibody_shape": 1.5, #fixed
    
    
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
                                     0.0015013064377048236], 
    #-> this will be the age component
    
    "prob_dis_logantibody_scale_old2" : [0.00016979990399999998,
                                     6.11873262e-05,
                                     6.381083333333333e-05,
                                     7.47948936e-05,
                                     6.0464333999999996e-05,
                                     2.523395e-05,
                                     1.4226534e-05,
                                     4.5398146911764703e-05,
                                     0.00013927194726027397,
                                     0.0006976346962365591,
                                     0.001657077666295884,
                                     0.0034513930825496347],
    
    "prob_dis_logantibody_scale" :[0.00016979990399999998,
                                     6.11873262e-05,
                                     6.381083333333333e-05,
                                     0.00010471285104,
                                     6.0464333999999996e-05,
                                     1.8042274250000003e-05,
                                     1.3040989499999999e-05,
                                     3.7240667388556985e-05,
                                     0.00011489935648972603,
                                     0.0005637452090800478,
                                     0.0013104440728360306,
                                     0.00282386706754061],
    
                                                
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
 
    
    
    'pop_size': 1_000_000,
    'update_demog': True,
    #'demo_burn': 1,
    'random_seed': False,
    'seed': 1234,
    'seed_no': 0,
    't_per_year': 52,
    'logging': False,
    'years': [10,18],
    'record_interval': 1,
    'save_population': True,
    "read_population": True,
    "pop_group": "non_indigenous",
    "pop_saving_address": "non_indigenous_varying_trans_v116",
}
