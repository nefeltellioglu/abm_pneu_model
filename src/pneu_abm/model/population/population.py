#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:44:30 2023

@author: ntellioglu
"""

import polars as pl
import numpy as np
from collections import defaultdict

from .utils import gen_ages, gen_deaths

class Population(object):
    """
    The base class for a population containing pl.dataFrame where
    each row represent an individual.
    
    
    :class:`.population.Population` is the simplest population class, providing
    sufficient functionality for age-structured populations and arbitrary
    contact groups.  It provides the basic functionality necessary for
    implementing demographic processes.


    """

    def __init__(self):
        
        # A pl dataframe of individuals
        self.I = pl.DataFrame()

        # A counter storing the next available ID for individuals.
        self.next_id = 0
        self.yearly_ages = 5
        

    #add individuals with given details
    def introduce_individuals(self, no_inds, ages, age_days, 
                              days_at_death):
        
        """
        A:param no_inds: Number of individuals to be introduced.
        :type no_inds: int
        :param ages: list of ages of individuals
        :type ages: list of integers
        :param age_days: list of age days of individuals
        :type age_days: list of integers
        
        :returns: I, individuals pl dataframe and next id.

        """
        new_inds = pl.DataFrame({
                    'id': pl.Series(name='id', values=list(range(self.next_id, 
                                                    self.next_id+ no_inds)),
                                    dtype=pl.Int64),
                    'age': pl.Series(name ='age', values= ages,dtype=pl.Int64),
                    'age_group': pl.Series(name='age_group', 
                        values = [age // self.yearly_ages for age in ages],
                                    dtype=pl.Int64),
                    'age_days': pl.Series(name='age_days', values= age_days,
                                          dtype=pl.Int64),
                    'days_at_death': pl.Series(name='age_days', 
                                               values= days_at_death,
                                          dtype=pl.Int64),
                    })
        
        self.I = pl.concat([self.I, new_inds], rechunk=True)
        self.next_id += no_inds
        
        return self.I, self.next_id

        
    #birth, migrations
    def introduce_births_and_migrations(self, t, t_per_year, no_of_births, 
                                        no_of_migrations, age_dist_mig):
        """
        A:param t: The current time step.
        :type t: int
        :param next_id: next available id
        :param I: population df
        :param no_of_births: total number of new births
        :param no_of_migrations: total number of new migrations
        
        :returns: None.

        """
        
        #new births
        ages = [0] * no_of_births
        age_days = [t * t_per_year % 365] * no_of_births
        days_at_death = gen_deaths(t_per_year, no_of_births, 
                                          ages, age_days, self.pop_rng)
        self.I, self.next_id = self.introduce_individuals(no_of_births, 
                                                ages, age_days, days_at_death)
        
        #migrated population
        mig_ages, mig_age_days = gen_ages(t_per_year, no_of_migrations, 
                                          age_dist_mig, self.pop_rng)
        days_at_death = gen_deaths(t_per_year, no_of_migrations, 
                                          mig_ages, mig_age_days, self.pop_rng)
        self.I, self.next_id = self.introduce_individuals(no_of_migrations, 
                                                mig_ages, mig_age_days,
                                                days_at_death)
        
        
        
 