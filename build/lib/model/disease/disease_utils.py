#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:00:35 2023

@author: ntellioglu
"""

import os
import sys
from math import exp, log
from collections import defaultdict

import numpy as np
import polars as pl
import itertools
from ..population.utils import gen_ages

####
def add_at_risk_column(I, disease, rng, ismigrated = False):
    #if ismigrated:
    #    pass
    #else:
        #pass
        
    #quantile is used rather than random column
    I = I.with_columns(
        (pl.when(pl.col("quantile") < \
                disease.at_risk_percentages["at_risk_tier_1"][I["age"]])
        .then(1).otherwise(
            pl.when(pl.col("quantile") < \
             disease.at_risk_percentages["at_risk_tier_2"][I["age"]])
            .then(2).otherwise(0)).alias("at_risk")),
         #(pl.lit(rng.rand(I.height))).alias("random")
         )
    #checks
    #disease.at_risk_percentages
    #selected = I.filter(pl.col("age").is_between(18, 34))
    #selected.filter(pl.col("at_risk") == 1).height/selected.height
    #selected.filter(pl.col("at_risk") > 0).height/selected.height
        
    return I



def gen_disease_states_for_mig_population(I, rng):
    #NOT USED 
    #TODO: generate disease and vaccine values for the migrated population
    disease_values, vaccine_values, immunity_values = None, None, None
    
    return disease_values, vaccine_values, immunity_values


def add_disease_vacc_columns(I, disease, new_inds, rng, ismigrated = False):
    #NOT USED 
    """
    Generates disese and vacc columns to a given individuals dataframe

    :param new_inds: table of individuals
    :type new_inds: pl dataframe
    :param rng: The random number generator to use.
    :type rng: :class:`random.Random`
    
    :returns updated new_inds dataframe
    """
    
    if ismigrated:
        
        disease_values, vaccine_values, immunity_values =\
                            gen_disease_states_for_mig_population(I, rng)
        new_inds = (
               new_inds.with_columns(
                create_disease_and_vaccine_states(len(new_inds),
                                                  list(disease.strains),
                                                  list(disease.vaccines.keys()),
                                                  disease_values, 
                                                  vaccine_values,
                                                  immunity_values),
                )
            )
    else: 
        
        new_inds = (
               new_inds.with_columns(
                create_disease_and_vaccine_states(len(new_inds),
                                                  list(disease.strains),
                                                  list(disease.vaccines.keys())
                                                  ),
                )
            )
    return new_inds

def create_disease_and_vaccine_states(pop_size,no_strains, 
                                            vaccine_states, 
                                            disease_values = None,
                                            vaccine_values = None,
                                            immunity_values = None):
    #NOT USED 
    """
    creates a pl dataframe with disease and vaccine states
    
    :param pop_size: The population size
    :type pop_size: int
    :param disease_states: name of the strain columns
    :type disease_states: list of strings
    :param vaccine_states: name of the vaccine columns
    :type vaccine_states: list of strings
    :param disease_values: list of strain infection data
    :type disease_values:   list, [int, [infected strains], 
                                   [end time of infection], 
                                   [start time of past infections]]
        Int: number of infections
         list of infected strains: either [] or ["st1", "st2" ,etc.] 
        List of end time of infections: [], or [14, 35,etc]
        List of start time of past infections: [], or [14, 35,etc]
        
        -> end time = days rather than simulation tick 
        -> start time = days rather than simulation tick 
        
        Initialized as [0, [], [], []]



    :param vaccine_values: list of vaccine data
    :type vaccine_values: list of objects, 
        List of [no vaccines received, type of vaccine, 
        [times of vaccines], [initial antibody levels after each dose]]
    : ex vaccine_values: 
        
        -no vaccines received: integer,
        - type of vaccine: string "pcv7"
        - times of vaccination doses received: list of days and 
        -initial (max) antibody level after each dose
        [[1,5,8], [0.7, 0.8, 0.9]] -> vaccinations received at days 1, 5, 8 
        and antibody levels are 0.7,0.8,0.9 in days 1, 5, 8.
        
        Initialized as [0, None, [], []]


    :param rng: The random number generator to use.
    :type rng: :class:`random.Random`
    """
    
    if not disease_values:
        single_ind = {"strain_list": [], 
                    "endTimes": [], 
                    "no_past_infections": 0}
        
        disease_values = [single_ind for _ in range(pop_size)]
        no_of_strains = [0] * pop_size
    
    if not immunity_values:
        immunity_values = [0] * pop_size
        
    if not vaccine_values:
        single_ind = {"no_of_doses": 0, 
                      "vaccine_type": None, 
                    "vaccine_times": [], 
                    "antibody_level_after_doses": []}
        vaccine_values = [single_ind for _ in range(pop_size)] 
    
    return pl.DataFrame({
        'vaccines': pl.Series(name='vaccines', values= vaccine_values),
        'antibody_level': pl.Series(name='antibody_level', values=immunity_values),
        'no_of_strains': pl.Series(name='no_of_strains', values= no_of_strains),
        'infections': pl.Series(name='infections', values=disease_values),
        
         } )

def gen_age_structured_pop(disease, t_per_year, pop, pop_size, age_probs, 
                           isUpload, rng):
    """
    Generate a pl df representing a population
     of individuals with given age structure.

    :param pop: The population
    :type pop: Population
    :param pop_size: The number of individuals to generate.
    :type pop_size: int
    :param age_probs: A table mapping probabilities to age.
    :type age_probs: list
    :type cutoffs: tuple
    :param rng: The random number generator to use.
    :type rng: :class:`random.Random`
    """
    ages, age_days = gen_ages(t_per_year, pop_size, age_probs, rng)
    
    pop.I, pop.next_id = pop.create_initial_population(disease, pop_size, 
                                            ages, age_days, rng,False,
                                            isUpload)
   
    
   
    return pop
