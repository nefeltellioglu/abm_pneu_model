#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:44:30 2023

@author: ntellioglu
"""

import polars as pl
import numpy as np
from .utils import gen_ages, update_age_group, gen_deaths
from .population import Population
from ..disease.disease_utils import add_disease_vacc_columns

class DisPopulation(Population):
    """
    The population class for a population containing pl.dataFrame where
    each row represent an individual.
    
    
    :class:`.population.Population` adds disease and vaccination rows 
    to individuals pl.dataFrame.
    
    :param disease: takes disease class as an input

    """
    def __init__(self,  logging=True):
        super(DisPopulation, self).__init__()
        
    
    def create_initial_population(self, disease, no_inds, 
                              ages, age_days, days_at_death, pop_rng,
                              ismigrated = False, isUpload = False):
        if isUpload:
            pop_fname = 'src/pneu_abm/data/%s.parquet'%disease.pop_reading_address
            self.I = pl.read_parquet(pop_fname)
            self.next_id = self.I["id"].max() + 1
            
        else:
            self.I = pl.DataFrame([
                    pl.Series("id", list(range(self.next_id, 
                                      self.next_id+ no_inds)), dtype=pl.Int64),
                    pl.Series("age", ages, dtype=pl.Int64),
                    pl.Series("age_days", age_days, dtype=pl.Int64),
                    pl.Series("days_at_death", days_at_death, dtype=pl.Int64),
                    #pl.Series("age_group", ages, dtype=pl.Int64),
                ]).with_columns( 
                    update_age_group(disease.cmatrix.yearly_ages, 
                                     disease.cmatrix.max_age),
                    quantile = pl.lit(self.pop_rng.random(no_inds)),
                    vaccines = pl.struct(no_of_doses = pl.lit(0), 
                                         on_time = pl.lit(0).cast(pl.Int64),
                                vaccine_type = pl.lit("").cast(pl.Utf8),
                                final_vaccine_time = \
                                        pl.Series('final_vaccine_time', 
                                [None] * no_inds, dtype = pl.Int32),
                                #antibody_level_after_final_dose = pl.lit(0.0),
                        ),
                    no_of_strains =  pl.lit(0),
                    strain_list = pl.Series('strain_list', 
                                    [[]] * no_inds, dtype=pl.List(pl.Utf8)),
                    endTimes = pl.Series('endTimes', 
                                    [[]] * no_inds, dtype=pl.List(pl.Float64)),
                    no_past_infections = pl.lit(0),
                    )
            self.next_id += no_inds
        self.disease_pop = pl.DataFrame([])
        self.vaccinated_disease_pop = pl.DataFrame([])
        self.vaccinated_acq_pop = pl.DataFrame([])
        return self.I, self.next_id
    
    def introduce_individuals(self, day, disease, no_inds, 
                              ages, age_days,days_at_death,
                              ismigrated = False):
        
        """
        #Adds individuals with given details.
        
        A:param no_inds: Number of individuals to be introduced.
        :type no_inds: int
        :param ages: list of ages of individuals
        :type ages: list of integers
        :param age_days: list of age days of individuals
        :type age_days: list of integers
        :param disease_states: list of list of disese states
        :type age_days: list of list
        
        :param rng: random number generator
        
        :returns: I, individuals pl dataframe and next id.

        """
        
        if not ismigrated:
            chosen = pl.DataFrame([
                    pl.Series("id", list(range(self.next_id, 
                                      self.next_id+ no_inds)), dtype=pl.Int64),
                    pl.Series("age", ages, dtype=pl.Int64),
                    pl.Series("age_days", age_days, dtype=pl.Int64),
                    pl.Series("days_at_death", days_at_death, dtype=pl.Int64),
                    #pl.Series("age_group", ages, dtype=pl.Int64),
                ]).with_columns( 
                    update_age_group(disease.cmatrix.yearly_ages, 
                                     disease.cmatrix.max_age),
                    quantile = pl.lit(self.pop_rng.random(no_inds)),
                    vaccines = pl.struct(no_of_doses = pl.lit(0), 
                                         on_time = pl.lit(0).cast(pl.Int64),
                                vaccine_type = pl.lit("").cast(pl.Utf8),
                                final_vaccine_time = \
                                        pl.Series('final_vaccine_time', 
                                [None] * no_inds, dtype = pl.Int32)),
                    no_of_strains =  pl.lit(0),
                    strain_list = pl.Series('strain_list', 
                                    [[]] * no_inds, dtype=pl.List(pl.Utf8)),
                    endTimes = pl.Series('endTimes', 
                                    [[]] * no_inds, dtype=pl.List(pl.Float64)),
                    no_past_infections = pl.lit(0),
                    )
            
        else:
            #if migrated: age_days are used directly from the sampled inds
            #TODO: find another way to select individuals with given ages
            #given age list, ie [30,45,3,5,30], select individuals with
            #same ages
            """
            Important: mig_age_days is not used here. age_days is kept 
            the same among the chosen individuals who will have the same
            characteristics as the migrated population. This was a 
            simplification to keep the vaccine schedule 
            especially for the paediatric population to make sense 
            ie: if chosen age:0, age_days: 350 with 3 doses of 7vPCV
            do not update the age_days as a random value (ie 7) to
            avoid unrealistic vaccine rollouts in the migrated population
            """
          
            migrants = (pl.DataFrame({'age': ages}).group_by("age")
                        .agg(pl.count().alias('migrant_count')).sort("age"))
            #migrants.join(self.I, on='age', how='left').group_by("age")
            seed = self.pop_rng.integers(0, 2**32 - 1)
            chosen = (migrants.join(self.I.select('age', 'id'), 
                                   on='age', 
                                   how='left').group_by('age')
                      .agg(pl.all().shuffle(seed = seed)
                           .head(pl.col('migrant_count').first()))
                      .explode(pl.exclude('age')).drop('migrant_count'))
                             
            chosen = chosen.join(self.I, on=['age', 'id']).with_columns(
                        (pl.Series("id", list(range(self.next_id, 
                                 self.next_id + no_inds)), dtype=pl.Int64)
                        .alias("id")),
                        pl.Series("quantile", self.pop_rng.random(no_inds)),
                        #pl.Series("age_days", age_days, dtype=pl.Int64),
                        #pl.Series("days_at_death", days_at_death,
                        #          dtype=pl.Int64),
                        )

        self.I = pl.concat([self.I, chosen], rechunk=True,\
                               how = 'diagonal')
        self.next_id += no_inds
        
        return self.I, self.next_id

    #birth, migrations
    def introduce_births_and_migrations(self, disease, day, t_per_year, 
                                        no_of_births, no_of_migrations, 
                                        age_dist_mig):
        """
        A:param day: The current day.
        :type day: int
        :param next_id: next available id
        :param I: population df
        :param no_of_births: total number of new births
        :param no_of_migrations: total number of new migrations
        
        :returns: None.

        """
        #new births
        if no_of_births:
            ages = [0] * no_of_births
            age_days = [0] * no_of_births
            days_at_death = gen_deaths(t_per_year, no_of_births, 
                                              ages, age_days, 
                                              self.death_rates, self.pop_rng)
            
            self.I, self.next_id = self.introduce_individuals(day, disease,
                                   no_of_births, ages, age_days, days_at_death)
        
        #migrated population
        if no_of_migrations:
            mig_ages, mig_age_days = gen_ages(t_per_year, no_of_migrations, 
                                              age_dist_mig, self.pop_rng)
            days_at_death = gen_deaths(t_per_year, no_of_migrations, 
                                        mig_ages, mig_age_days, 
                                        self.death_rates, self.pop_rng)
            self.I, self.next_id = self.introduce_individuals(day, disease,
                                                     no_of_migrations, 
                                                    mig_ages, mig_age_days,
                                                    days_at_death,
                                                    ismigrated = True)
        
        
        
      
    

    

