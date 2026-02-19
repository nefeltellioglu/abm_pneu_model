#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:46:28 2023

@author: ntellioglu
"""

import os
import tables as tb
from random import Random
from math import exp
from collections import defaultdict
import numpy as np
import polars as pl

from .utils import load_age_rates, load_probs,load_prob_list,\
                    gen_age_structured_pop, will_live
                    
from .population import Population

def _adjust_prob(rate, t_per_year):
    """
    convert from an annual rate to a per-time-period probabiliy
    where time-period = 1/t_per_year

    :param rate: annual rate to convert.
    :type rate: double
    :param t_per_year: number of time periods per year.
    :type t_per_year: int
    """
    tp = 1.0 / t_per_year
    return 1 - pow(1 - rate, tp)

class Simulation(object):
    """
    Basic demographic simulation object.

    Handles updating of births, deaths, aging, immigration

    :param p: dictionary of simulation parameters.
    :type p: dict
    :param create_pop: If `True` (default), create a random population; 
        otherwise, this will need to be done later.
    :type create_pop: bool

    """
    def __init__(self, p, create_pop=True, h5file=None):
        self.p = p  # local copy of parameters
        self.obs_on = True
        self.observers = {}
        
        self.growth_residue = 0.0 #-> can be useful in small populations
        
        # store this in a data/rate dictionary?
        self.death_rates = []
        self.birth_rates = []
        self.mig_rates = []
        
        self._setup_params()

        self.h5file = h5file
        if self.h5file is None:
            self.h5file = tb.open_file(os.path.join(p['prefix'], 
                                                    'population.hd5'), 'w')
            self._store_params(p)

        self.P = None
        if create_pop:
            self.create_population()
        
        
        
    def _setup_params(self):
        """
        Reset various simulation parameters.

        """
        self._load_demographic_data()
        
        
    def _store_params(self, p):
        """
        If not already present, store simulation parameters in output file and mark file as incomplete.
        """
        if '/params' not in self.h5file:
            self.h5file.create_group('/', 'params', 'Parameters')
        for k, v in list(p.items()):
            # rhdf5 has problems reading boolean values, therefore write as 0/1
            if type(v) is bool:
                self.h5file.set_node_attr('/params', k, 1 if v else 0)
            else:
                self.h5file.set_node_attr('/params', k, v)
        self.h5file.set_node_attr('/params', 'complete', 0)
        self.h5file.flush()
        
    def create_population(self):
        """
        Create a population according to specified age distributions.
        """
        self._setup_params()
        self.P = Population(self.p['logging'])
        self.P.pop_rng = np.random.RandomState(self.p['pop_seed'])
        self.P.death_rates = np.array([v[0] for v in self.death_rates.values()])
        self.P = gen_age_structured_pop(self.p['t_per_year'], self.P, 
                             self.p['pop_size'], self.age_dist, 
                             self.P.death_rates, self.P.pop_rng)
        
        
    
    def add_observers(self, *observers):
        """Add a new observer to the observers list."""
        for observer in observers:
            self.observers[observer.label] = observer
    
    @staticmethod
    def _parse_age_rates(filename, factor, final):
        """
        Parse an age-year-rate table to produce a dictionary, keyed by age, 
        with each entry being a list of annual rates (by factor:time_per_year).
        It returns probability of not dying at a given time step for every age category
        
        Setting final to 'True' appends an age 100 rate of >1 (e.g., to 
        ensure everyone dies!
        """
        
        dat = load_age_rates(filename)
        rates = {}
        for line in dat:
            #UPDATED: used adjust prob structure
            #previously: rates[line[0]] = [x * factor for x in line[1:]]
            rates[line[0]] = [pow(1 - x, factor) for x in line[1:]]
            
        if final:
            rates[100] = [0 for _ in dat[0][1:]]  # added to try and fix appearance of individuals >100y
            #rates[101] = [0 for _ in dat[0][1:]]  # everybody dies...
        return rates
    
    @staticmethod
    def _parse_death_rates(filename, final):
        """
        Parse an age-year-rate table to produce a dictionary, keyed by age, 
        with each entry being a list of annual rates (by factor:time_per_year).
        It returns probability of not dying at a given time step for every age category
        
        Setting final to 'True' appends an age 100 rate of >1 (e.g., to 
        ensure everyone dies!
        """
        
        dat = load_age_rates(filename)
        rates = {}
        for line in dat:
            #UPDATED: used adjust prob structure
            #previously: rates[line[0]] = [x * factor for x in line[1:]]
            rates[line[0]] = [x for x in line[1:]]
            
        if final:
            rates[100] = [1 for _ in dat[0][1:]]  # added to try and fix appearance of individuals >100y
            rates[101] = [1 for _ in dat[0][1:]]  # everybody dies...
        return rates
    
    def _load_demographic_data(self):
        """
        Load data on age-specific demographic processes (mortality)
        and adjust event probabilities according to time-step.

        All paths, parameter values, etc. are contained in the parameter dictionary
        passed in when the :class:`Simulation` object is created.
        """

        # load age distribution
        self.age_dist = load_probs(os.path.join(self.p['resource_prefix'],
                                                self.p['age_distribution']))
        
        # load age distributionfor migrated pop
        self.age_dist_mig = load_probs(os.path.join(self.p['resource_prefix'],
                                                self.p['age_distribution_mig']))

        annual_factor = 1.0 / self.p['t_per_year']

        # load and scale MORTALITY rates (not yearly, t_per_yearly)
        initial_death_rates = self._parse_death_rates(os.path.join(
                self.p['resource_prefix'],
                self.p['initial_death_rates']), True)
        
        self.death_rates = initial_death_rates
        
        
        
        #create birth rate list
        yearly_birth_rates = load_prob_list(os.path.join(
            self.p['resource_prefix'],self.p['birth_rates'])
            )
        
        yearly_adj_birth_rates = [_adjust_prob(x, self.p['t_per_year'])
                                           for x in yearly_birth_rates]
        
        self.birth_rates = np.repeat(yearly_adj_birth_rates,
                                     self.p['t_per_year'])
        no_iterations = self.p['t_per_year'] *\
                            (self.p['years'][1] -self.p['years'][0] + 1)
        
        if len(self.birth_rates) - 1 < no_iterations:
            self.birth_rates = list(self.birth_rates) +\
            [self.birth_rates[-1]]* (no_iterations - len(self.birth_rates) + 1)
        
        #create mig rate list
        yearly_mig_rates = load_prob_list(
            os.path.join(self.p['resource_prefix'],
                        self.p['mig_rates']))
        
        yearly_adj_mig_rates = [_adjust_prob(x, self.p['t_per_year'])
                                           for x in yearly_mig_rates]
        
        self.mig_rates = np.repeat(yearly_adj_mig_rates,
                                     self.p['t_per_year'])
        no_iterations = self.p['t_per_year'] *\
                    (self.p['years'][1] -self.p['years'][0] + 1)
        
        if len(self.mig_rates) - 1 < no_iterations:
            self.mig_rates = list(self.mig_rates) +\
            [self.mig_rates[-1]]* (no_iterations - len(self.mig_rates) + 1)
         
    
    def update_all_demo(self, t):
        """
        Carry out a single update of all demographic aspects population.

        :param t: the current time step.
        :type t: int
        :returns: a tuple containing lists of births, deaths

        """
        
        deaths = 0
        
        
        #period: number of dates that each ind aged since the last tick
        period = 365 // self.p['t_per_year']
        death_rates = pl.Series("death_rates", [np.array([i[0] for i in self.death_rates])])
        #random_numbers = self.P.pop_rng.random(self.P.I.height)
        
        self.P.I = (
               self.P.I
               .with_columns(
                   
                  (pl.col('age').add((pl.col('age_days') + period) // 365).alias('age')),
                  ((pl.col('age_days') + period) % 365).alias('age_days'),
                  ((pl.col('age') + (pl.col('age_days') + period)// 365) // self.P.yearly_ages)
                  .alias('age_group'),
                  #((pl.col("age") * 365 + pl.col("age_days"))
                  # .le(pl.col('days_at_death'))
                  # .alias("alive"))
                   
                  )#.filter(pl.col("alive")).drop("alive")
               .filter(((pl.col("age") * 365 + pl.col("age_days"))
                .le(pl.col('days_at_death')))
               ))
       
        #BIRTHS + population growth
        ####################
        new_individuals = (len(self.P.I)) * self.birth_rates[t]
        # get whole part of new individuals
        new_now = int(new_individuals)
        # add fractional part to residue accumulation
        self.growth_residue += (new_individuals - new_now)
        # grab any new 'whole' individuals
        new_residue = int(self.growth_residue)
        new_now += new_residue
        self.growth_residue -= new_residue

        # immigration
        ##############
        imm_tgt = int(len(self.P.I) * self.mig_rates[t])
        #add that many newly born and migrated individuals
        
        self.P.introduce_births_and_migrations(t, self.p['t_per_year'], 
                                               new_now, imm_tgt, 
                                               self.age_dist_mig)
        
        return new_now, deaths, imm_tgt
    
    
    def update_observers(self, t, **kwargs):
        """
        Store observed data (if observers are switched on).
        t: must be given in days
        """
        if self.obs_on:
            for observer in self.observers.values():
                observer.update(t, **kwargs)
                

    def done(self, complete=False):
        """
        Close output file, marking as complete if specified.
        """
        if complete:
            self.h5file.set_node_attr('/params', 'complete', 1)
        self.h5file.close()
        