#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:29:12 2023

@author: ntellioglu
"""

p = {
    # directories
    'resource_prefix': 'src/pneu_abm/data/',
    'prefix': 'src/pneu_abm/output/',

    # demographic model parameters
    'age_distribution': 'population/age_dist_2002.dat',
    'initial_death_rates': 'population/death_rates_2002.dat',
    'birth_rates': 'population/birth_rates.dat', 
    'mig_rates': 'population/migration_rates.dat',
    'age_distribution_mig': 'population/age_dist_migration_2018.dat',
    'cmatrix': 'population/all_contact_matrix_Australia_prem_2017.csv',
    #'cmatrix': 'population/cmatrix_mistry_Australia.csv',
    #disease parameters
    'strain_list': 'disease/strain_list.csv',
    'vaccine_list': 'vaccine_configs/vaccine_list.csv',
    'initial_cases': 'disease/initialize_infections.csv',
    'age_serotype_specific_disease_multipliers': \
        'disease/age_serotype_specific_disease_multipliers_all_same.csv',
    'age_specific_duration_of_infections': \
        'disease/age_specific_duration_of_infections.csv',
    'age_specific_prot_parameters': \
        'disease/age_specific_prot_parameters.csv',

    'transmission_coefficient': 0.058, #per week #0.00825, #per day #### 
    'reduction_in_susceptibility_coinfections': 0.8, #
    'max_no_coinfections': 2,
    'external_exposure_prob': 0.0006,#per week
    'external_exposure_check_per_year': 4, #per year
    'no_daily_external_strains':2,
    'halt': True,
    'noise_in_strain_distribution': 0.0,
    'infected_population_fraction': 0.12,
    'randomly_select_strains': False,
    'randomly_no_initial_strains': 20,
    
    #waning halflife days
    "waning_halflife_day_adult": 600,
    "waning_halflife_day_child": 125,
    
    
    'clinical_model_per_simulation': 100,
    
    'pop_size': 1_000_000,
    'update_demog': True,
    #'demo_burn': 1,
    'random_seed': False,
    'pop_seed': 1234,
    'transmission_seed': 1234,
    'disease_seed': 1234,
    'vaccine_seed': 1234,
    'seed_no': 0,
    't_per_year': 52, #365,
    'logging': False,
    'years': [0,18],
    'num_runs': 100,
    'record_interval': 1,
    'save_population': True,
    "read_population": True,
    "pop_reading_address": "initial_population",
    "pop_saving_address": "saved_checkpoints/saved_population",
}

import math

def weekly_rate_to_daily_prob(lambda_week):
    """
    Convert a weekly transmission probability to a daily probability.
    """
    lambda_day = lambda_week / 7
    return 1 - math.exp(-lambda_day)

if p['t_per_year'] == 365:
    p['transmission_coefficient'] = weekly_rate_to_daily_prob(
        p['transmission_coefficient'])

if 'cmatrix_mistry_Australia' in p["cmatrix"]:
    p['age_classes'] = list(range(0, 85, 1))                      
elif 'all_contact_matrix_Australia_prem_2017' in p["cmatrix"]:
    p['age_classes'] = list(range(0, 80, 5))
else:
    assert False, "please provide age classes of the contact matrix here"