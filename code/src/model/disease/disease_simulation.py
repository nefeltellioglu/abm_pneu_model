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
import time
import matplotlib.pyplot as plt
import csv

from ..population.utils import will_live

from .disease_utils import gen_age_structured_pop
from ..population.disease_population import DisPopulation
from ..population.simulation import Simulation, _adjust_prob


class DisSimulation(Simulation):
    """
    Simulation class that simulates disease and demographical dynamics 
    at every step.

    The demographical updates include births, deaths, aging, immigration.
    It makes the demographical and disease related updates in 
    DisPopulation class object.
    
    In _main_loop(), it updated demography (if True) and disease transmission
    and save the population (if True) when the simulation ends. 
    
    Currently population is saved in three different csv files due to the 
    nested struct columns of list of strains (_strain_list.csv) and 
    corresponding time until clearance (_endList.csv) and the rest of the 
    columns. This saving section can be later updated by using write_parquet
    function of polars library which can write nested columns into the same
    file. The only downside would be that the saved parquet files cannot be 
    opened by excel for an easy review of the dataset.
    

    :param p: dictionary of simulation parameters.
    :type p: dict
    :param create_pop: If `True` (default), create a random population; 
        otherwise, this will need to be done later.
    :type create_pop: bool

    """
    def __init__(self, p, disease, rng):
        super(DisSimulation, self).__init__(p, 
                            h5file=disease.h5file, create_pop=True)
        self.disease = disease
        self.rng = rng
        #self.nprng = np.random.RandomState(self.rng.randint(0, 99999999))
        
        create_pop = True
        if create_pop:
            self.create_population()  
        
            
    
        
         
    ###############
    #initialization
    def create_population(self):
        """
        Create a population according to specified age distributions.
        """
        self._setup_params()
        self.P = DisPopulation(self.p['logging'])
        self.P = gen_age_structured_pop(self.disease, self.p['t_per_year'], 
                                self.P, self.p['pop_size'], self.age_dist, 
                                self.p['read_population'],self.rng)
        self.setup()
    

    def _init_contact_matrix(self, t=0):
        # initialise contact matrix
        self.disease.cmatrix.init_age_classes(self.P)
        self.disease.cmatrix.init_contact_matrix()
        
    def setup(self, start_year=0, verbose=True, seed_inds=None):
        """
        Setup population for start of simulation.
        
        If no epidemic burn in is required, run (or load) demographic burn in  
        and seed infection.  Otherwise, run (or load) epidemic burn in.
        """
        self._load_demographic_data()
        self._init_contact_matrix()
        # seed infection, switch observers off, burn in and save population
        self.disease.seed_infection(start_year * self.p['t_per_year'], self.P,
                                    self.p, self.rng)
    
    
    def plot_dur_infections(self):
        """
        only make sense in the initialization as it does only print the end
        time of the infections. 
        """
        durations = self.P.I["endTimes"].list.get(0).fill_null(np.nan)
        ages = list(self.P.I.select(["age"]))
        plt.scatter(ages, durations)
        plt.show()
        
    ################
    #updating
    def _main_loop(self, year_begin, years, verbose=False):
        """
        Run simulation.
        """
        t_begin = int(year_begin * self.p['t_per_year'])
        t_end = int((year_begin + years) * self.p['t_per_year'])

        if verbose:
            self.start_time = time.time()
            #self.print_column_labels()
            self.print_pop_numbers(t_begin)

        self.disease.update_observers(t_begin, disease=self.disease, 
                                      pop=self.P, cases=[],introduction=False,
                                      day = int(year_begin * 364),
                                      new_I=[], rng=self.rng)
        self.plot_dur_infections()
        for t in range(t_begin + 1, t_end + 1):
            
            day = t * 364 // self.p['t_per_year']
            # update demography (if required)
            if self.p['update_demog']:  # and t%52==0:
                births, deaths, imms = self.update_all_demo(t, day)  
                
            #print(self.P.I.select(["age", "age_days", "age_group"]))
           # print(self.P.I.select(["age", "infections"]))
            #print(self.P.I.sort("id").head)
            
            # update disease
            if self.disease.update(t, day, self.P, self.p["t_per_year"],
                                   self.rng):
                if verbose:
                    self.print_pop_numbers(t)
                #break  # update returns true if halting upon fade out

            #if verbose:
            #    self.print_pop_numbers(t)
            
        if verbose:
            self.print_column_labels()
            self.end_time = time.time()
            print("time:", self.end_time - self.start_time)
            
        if self.p["save_population"]:
           # output_fname = os.path.join(self.p['resource_prefix'], 
           #                                 "disease_pop_data", 
           #                                 "seed_%s"%self.p["seed_no"])
            output_fname = os.path.join(self.p['resource_prefix'], 
                                            "%s"%self.p["pop_saving_address"])
            self.P.I = self.P.I.unnest("vaccines")
            self.P.I = self.P.I.with_columns(
                pl.concat_list("id","strain_list" ).alias("n_strain_list"),
                pl.concat_list("id","endTimes" ).alias("n_endTimes"))
            
            with open('%s_strain_list.csv'%output_fname, 
                      'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "st11", "st2"])
                writer.writerows((self.P.I["n_strain_list"]).to_list())
            with open('%s_endTimes.csv'%output_fname, 
                      'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "endt1", "endt2"])
                writer.writerows(self.P.I["n_endTimes"].to_list())
                
    
            
            (self.P.I.drop("strain_list", "endTimes", 
                           "n_strain_list", "n_endTimes")
            .write_csv(os.path.join("%s.csv"%output_fname
                             )))
            
            
            
    def run(self, verbose):
        """
        A version of run (above) that checks for the existence of a
        checkpoint file before running, and loads that by preference.
        Otherwise, run is called normally, and a checkpoint file is 
        saved (in directory specified by prefix).
        
        'observers' is a Boolean value specifying whether observers
        should be switched on or off for this period.
        """
        start_year = self.p['years'][0]
        #self.p['burn_in'] + self.p['epi_burn_in']
        years = self.p['years'][1] - self.p['years'][0] 
        
        observers = True #(cur_years == year_list[-1])
        
        self._init_contact_matrix(start_year * self.p['t_per_year'])
        # switch observers off, burn in and save population
        self.disease.obs_on = observers
        self._main_loop(start_year, years, verbose)
        
            
            
    def update_all_demo(self, t, day):
        """
        Carry out a single update of all demographic aspects population.

        :param t: the current time step.
        :type t: int: simulation tick rather than day
        :returns: a tuple containing lists of births, deaths

        """
        
        deaths = 0
        
        
        #period: number of dates that each ind aged since the last tick
        period = 364 // self.p['t_per_year']
        
        cur_year = day // 364
        
        death_rates = pl.Series("death_rates", 
                    [np.array([i[0] for i in 
                               self.death_rates[cur_year].values()])])
        
        
        self.P.I = (
               self.P.I
               .with_columns(
                   
               (pl.col('age').add((pl.col('age_days') + period) // 364)
               .alias('age')),
               
               ((pl.col('age_days') + period) % 364).alias('age_days'),
               
               (
                   pl.when(pl.col('age_group')  >= 15)
                    .then(15)
                    .otherwise((pl.col('age') + \
                            (pl.col('age_days') + period) // 364) // 5)
                        .alias('age_group')
                ),
        
              ((pl.col('random') < death_rates.list.take(self.P.I["age"])[0])
               .alias("alive")),
              
              (pl.lit(self.rng.rand(self.P.I.height))).alias("random"),
               ).filter(pl.col("alive")).drop("alive")
               )
            
        
                
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
        
        self.P.introduce_births_and_migrations(self.disease, day, 
                                self.p['t_per_year'], new_now, imm_tgt, 
                                               self.age_dist_mig,self.rng)
        
        return new_now, deaths, imm_tgt
    
    # OUTPUT AND HELPER FUNCTIONS ###########################################
    #TODO: print_column_labels needs to be updated
    
    def print_column_labels(self):
        #TODO: needs to be updated
        #print(self.disease.state_labels())
        #print(''.join(['%7s' % x
        #               for x in
        #         (['t', 't(y)', 't(d)'] + self.disease.state_labels())]))
        print(''.join(['%7s' % x
                       for x in
                       (['t', 't(y)', 't(d)'] )]))
    
   
    def print_pop_numbers(self, t):
        #TODO: print_pop_numbers needs to be updated
        #print(''.join(['%7d' % x
        #               for x in
        #               ([t, t // self.p['t_per_year'],
        #         t % self.p['t_per_year'] * (364.0 / self.p['t_per_year'])]
        #                + self.disease.state_counts())]))
        print(''.join(['%7d' % x
                       for x in
                       ([t, t // self.p['t_per_year'],
                 t % self.p['t_per_year'] * (364.0 / self.p['t_per_year'])]
                        )]))
    
    
    