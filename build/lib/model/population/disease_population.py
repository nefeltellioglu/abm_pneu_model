#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:44:30 2023

@author: ntellioglu
"""

import polars as pl
import numpy as np
from .utils import gen_ages
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
                              ages, age_days, rng, 
                              ismigrated = False, isUpload = False):
        if isUpload:
            pop = 'data/disease_pop_data/%s.csv'%disease.pop_group
            endTimes = \
                'data/disease_pop_data/%s_endTimes.csv'%disease.pop_group
            stran_list = \
                'data/disease_pop_data/%s_strain_list.csv'%disease.pop_group
            
            pop = pl.read_csv(pop)
            endTimes = pl.read_csv(endTimes)
            strain_list = pl.read_csv(stran_list)
            endTimes = endTimes.with_columns(
                pl.col("id").cast(pl.Int64),
                pl.concat_list(pl.col("endt1").cast(pl.Float64),
                               pl.col("endt2").cast(pl.Float64))
                .alias("endTimes")).with_columns(
                    (pl.col("endTimes").list.eval(pl.element()
                            .filter(pl.element().is_not_null())))
                    ).drop("endt1", "endt2")
            strain_list = strain_list.with_columns(
                pl.col("id").cast(pl.Int64),
                
                pl.concat_list("st11", "st2")
                .alias("strain_list")).with_columns(
                    (pl.col("strain_list").list.eval(pl.element()
                            .filter(pl.element().is_not_null())))
                    ).drop("st11", "st2")
            pop = pop.with_columns(
                pl.struct(pl.col("no_of_doses").cast(pl.Int32),
                          pl.col("on_time"),
                         pl.col("vaccine_type"), 
                         pl.col("final_vaccine_time").cast(pl.Int32),
                      ).alias("vaccines"),
                pl.col("no_of_strains").cast(pl.Int32),
                pl.col("no_past_infections").cast(pl.Int32),
                #endTimes["endTimes"],
                #strain_list["strain_list"]
                ).drop("no_of_doses",
                         "on_time","vaccine_type", "final_vaccine_time")
            pop = pop.join(strain_list, on="id", how="left")
            
            self.I = pop.join(endTimes, on="id", how="left")
            #remove vacc adults
            vacc_adults = (
                self.I.filter((pl.col("age")
                    .is_between(65,101)
                    
                    ) & (
                      pl.col("vaccines").struct.field("no_of_doses") > 0 
                        ))
                   .select(["id", "vaccines"])
            )
            if vacc_adults.height:             
                """vacc_adults = vacc_adults.with_columns(
                    vaccines = pl.struct(no_of_doses = pl.lit(0), 
                                     on_time = pl.lit(0).cast(pl.Int64),
                            vaccine_type = pl.lit("").cast(pl.Utf8),
                            final_vaccine_time = \
                                    pl.Series('final_vaccine_time', 
                        [None] * vacc_adults.height, dtype = pl.Int32)))"""
                no_vaccine =  pl.struct(no_of_doses = pl.lit(0), 
                                 on_time = pl.lit(0).cast(pl.Int64),
                        vaccine_type = pl.lit("").cast(pl.Utf8),
                        final_vaccine_time = \
                                pl.Series('final_vaccine_time', 
                        [None] * 1, dtype = pl.Int32))   
                self.I = self.I.with_columns(
                    pl.when(pl.col("id").is_in(vacc_adults["id"]))
                    .then(no_vaccine)
                    .otherwise(pl.col("vaccines")).alias("vaccines")
                    )
                
                
            self.next_id = pop["id"].max() + 1
            
        else:
            self.I = pl.DataFrame([
                    pl.Series("id", list(range(self.next_id, 
                                      self.next_id+ no_inds)), dtype=pl.Int64),
                    pl.Series("age", ages, dtype=pl.Int64),
                    pl.Series("age_days", age_days, dtype=pl.Int64),
                    #pl.Series("age_group", ages, dtype=pl.Int64),
                ]).with_columns( 
                    age_group = (pl.when(pl.col("age")>79)
                                 .then(15)
                                 .otherwise(pl.col("age") // 5)),
                    quantile = pl.lit(rng.rand(no_inds)),
                    random = pl.lit(rng.rand(no_inds)),
                    exp_random =(pl.lit(rng.exponential(
                        disease.duration_of_infection, no_inds))),
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
                              ages, age_days, rng, 
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
                    #pl.Series("age_group", ages, dtype=pl.Int64),
                ]).with_columns( 
                    age_group = (pl.when(pl.col("age")>79)
                                 .then(15)
                                 .otherwise(pl.col("age") // 5)),
                    quantile = pl.lit(rng.rand(no_inds)),
                    random = pl.lit(rng.rand(no_inds)),
                    exp_random =(pl.lit(rng.exponential(
                        disease.duration_of_infection, no_inds))),
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
            """ages = pl.DataFrame(pl.Series("age", ages))
            migrants = pl.DataFrame({'age': ages}).group_by("age")
            .agg(pl.count().alias('migrant_count')).sort("age")
            #migrants.join(self.I, on='age', how='left').group_by("age")
            chosen = migrants.join(self.I.select('age', 'id'), 
                                   on='age', how='left').group_by('age')
            .agg(pl.all().shuffle().head(pl.col('migrant_count').first()))
            .explode(pl.exclude('age')).drop('migrant_count')
            chosen = chosen.join(self.I, on=['age', 'id'])
            # TODO: choose new IDs and add to the end of self.I
            """
          
            migrants = (pl.DataFrame({'age': ages}).group_by("age")
                        .agg(pl.count().alias('migrant_count')).sort("age"))
            #migrants.join(self.I, on='age', how='left').group_by("age")
            chosen = (migrants.join(self.I.select('age', 'id'), 
                                   on='age', 
                                   how='left').group_by('age')
                      .agg(pl.all().shuffle()
                           .head(pl.col('migrant_count').first()))
                      .explode(pl.exclude('age')).drop('migrant_count'))
            
            chosen = chosen.join(self.I, on=['age', 'id']).with_columns(
                        (pl.Series("id", list(range(self.next_id, 
                                 self.next_id+ no_inds)), dtype=pl.Int64)
                        .alias("id")),
                        pl.Series("random", rng.rand(no_inds)),
                        pl.Series("quantile", rng.rand(no_inds)),
                        pl.Series("exp_random", rng.exponential(
                           disease.duration_of_infection, no_inds))
                        )
              
           
           
            
                
            
            """
            new_infecteds = (
                chosen.filter(#(pl.col("no_of_strains") > 0) &\
                (pl.col("random") <= self.random_strain_fraction_mig_pop))
                .select(["id", "strain_list", "random", "no_of_strains",
                         "endTimes"])
            )
            new_infecteds = (new_infecteds
                              .with_columns(
                                  pl.Series("strain_list", 
                        [disease.strains.sample(1)] * new_infecteds.height),
                                pl.Series("random", 
                                        rng.rand(new_infecteds.height)),
                         (pl.col("endTimes").list.concat( day +
                                    self.period * (pl.col("exp_random") *\
                      (1 - (pl.col("age")/101) * 0.8)/self.period).round())),
                                ))
            
                           
            chosen = chosen.update(new_infecteds, on="id", how="left")
            """
            
            
            #self.I = self.I.join(chosen, on="id", how="outer")
        """if len(self.I["id"].unique()) !=  self.I.height:
            self.I.filter(pl.count("id").over("id")> 1)
            print("ids are not unique!")
        
        a = pl.concat([self.I, chosen], rechunk=True,\
                               how = 'diagonal')
        
        if len(a["id"].unique()) !=  a.height:
            self.I.filter(pl.count("id").over("id")> 1)
            print("ids are not unique!")
        else:
            self.I = a"""
        
        self.I = pl.concat([self.I, chosen], rechunk=True,\
                               how = 'diagonal')
        self.next_id += no_inds
        
        return self.I, self.next_id

    #birth, migrations
    def introduce_births_and_migrations(self, disease, day, t_per_year, 
                                        no_of_births, no_of_migrations, 
                                        age_dist_mig, rng):
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
            self.I, self.next_id = self.introduce_individuals(day, disease,
                                                        no_of_births, 
                                                    ages, age_days, 
                                                    
                                                     rng)
        
        #migrated population
        if no_of_migrations:
            mig_ages, mig_age_days = gen_ages(t_per_year, no_of_migrations, 
                                              age_dist_mig, rng)
            
            self.I, self.next_id = self.introduce_individuals(day, disease,
                                                     no_of_migrations, 
                                                    mig_ages, mig_age_days,
                                                    rng,
                                                    ismigrated = True)
        
        
        
      
    

    

