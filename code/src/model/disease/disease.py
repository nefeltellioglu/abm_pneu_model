#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:40:02 2023

@author: ntellioglu
"""


#import libraries
import numpy as np
#import tables as tb
from random import Random
from scipy.stats import lognorm,logistic
import tables as tb
import os
import json
import polars as pl
from copy import deepcopy
from .antibody_levels import create_vaccine_antibody_df, waning_ratio
class Disease(object):
    """
    Disease class that updates disease state of a population.
    
    An agent-based model that simulated multi-strain pathogen transmission.
    
    :param 
    
    """
    def __init__(self, p, cmatrix,rng, fname, mode):
        #########################
        #population parameters
        #########################
        
        # cmatrix: contact matrix 
        self.cmatrix = cmatrix
        
        # halt: if True, terminate simulation when no individuals are 
        #in a disease state
        self.halt = p['halt']

        #self.nprng = np.random.RandomState(rng.randint(0, 99999999))
        self.period = 364 // p['t_per_year']
        ####################
        #disease parameters
        ####################
        # External exposure rate: per person, 
        #per time step rate of disease introduction
        #self.external_exposure_rate = p['external_exposure_rate']
        
        # vaccines: set of vaccines to be applied during simulation 
        self.vaccines = {}

        # strains
        self.strains = []

        # disease states: set of states that denote presence of disease 
        #(e.g., latent, infectious)
        #self.disease_states = disease_states

        # infectious states: labels of states that contribute to force of 
        # infection 
        #self.infectious_states = infectious_states

        # by age: dictionary, keyed by state label, of number of individuals 
        #in that state per age class
        # for calculating community force of infection in an efficient fashion
        self.by_age = {}




        ###############
        #storage: h5file: HDF file for storing output of simulation
        self.h5file = tb.open_file(fname, mode)
        if mode in 'w':
            self.store_params(p)
            
        # obs_on: if True, record output on simulation
        self.obs_on = True

        # observers: data collection objects
        self.observers = {}
        
        self._load_disease_data(p)
    
    def _load_disease_data(self,p):
        """
        Load disease data 
        """

        # load disease data
        
        strain_fname = os.path.join(p['resource_prefix'],
                                                p['strain_list'])
        
        vaccine_fname = os.path.join(p['resource_prefix'],
                                                p['vaccine_list'])
        
        
        self.transmission_coef = p['transmission_coefficient']
        self.foi = pl.DataFrame()
        self.strain_distribution =  pl.DataFrame()
        self.duration_of_infection = p['duration_of_infection']
        self.reduction_in_susceptibility = \
                                p['reduction_in_susceptibility_coinfections']
        self.max_no_coinfections = p["max_no_coinfections"]
        self.expernal_exposure_prob = p["expernal_exposure_prob"]
        self.no_daily_external_strains = p["no_daily_external_strains"]
        self.noise_in_strain_distribution = p["noise_in_strain_distribution"]
        self.prob_acq_logantibody_shape = p["prob_acq_logantibody_shape"]
        self.prob_acq_logantibody_shift = p["prob_acq_logantibody_shift"]
        self.prob_acq_logantibody_scale = p["prob_acq_logantibody_scale"]
        
        self.prob_dis_logantibody_shape = p["prob_dis_logantibody_shape"]
        self.prob_dis_logantibody_shift = p["prob_dis_logantibody_shift"]
        #self.prob_dis_logantibody_scale = p["prob_dis_logantibody_scale"]
        self.prob_dis_logantibody_scale = pl.Series(
            "prob_dis_logantibody_scale", p["prob_dis_logantibody_scale"])
        self.prob_dis_logantibody_age = \
            pl.Series("prob_dis_logantibody_age", 
                      p["prob_dis_logantibody_age"])
        
        self.ipd_fraction_by_age_group = pl.Series(
                "ipd_fraction_by_age_group",p["ipd_fraction_by_age_group"])
        
        self.waning_halflife_day_adult = p["waning_halflife_day_adult"]
        self.waning_halflife_day_child = p["waning_halflife_day_child"]
        self.age_specific_duration_of_infections = \
                                p['age_specific_duration_of_infections']
                                
        self.age_specific_duration_of_infections = pl.DataFrame([
            pl.Series("age",list(range(0,102))),
            pl.Series("mean_dur_of_infection",
                      
                [self.age_specific_duration_of_infections[0]] * 3 +\
                [self.age_specific_duration_of_infections[1]] * 3 +\
                [self.age_specific_duration_of_infections[2]] * 13 +\
                [self.age_specific_duration_of_infections[3]] * 83 )
            ])
        self.isPopUploaded = p["read_population"]
        self.pop_group = p["pop_group"]
        
        #adjust rollout days
        period = 364 // p["t_per_year"]
            
        self.external_exposure_check_period = \
                int(p["t_per_year"] // p['external_exposure_check_per_year'])
            
        self.age_0_conversion = pl.DataFrame([
            pl.Series("age", [0] * 364),
            pl.Series("age_days", list(range(364))),
            pl.Series("age_coef", [0] * 30 + [1] * 60 + [2] * 60 + \
                                [3] * (180  +34)), #+ [-1] * 34 ),
            
            ])
        """self.age1_conversion = pl.DataFrame([
            pl.Series("age", list(range(1,102))),
             pl.Series("age_coef", [4] * 4 + [5] * 10 + [6] * 10 + \
                                   [7] * 25 + [8] * 15 + [9] * 37 
                       ), #+ [-1] * 34 ),
             ])    """
        self.age1_conversion = pl.DataFrame([
            pl.Series("age", list(range(1,102))),
             pl.Series("age_coef", [4] * 4 + [5] * 10 + [6] * 10 + \
                                  [7] * 25 + [8] * 15 + [9] * 10 + [10] * 5 +\
                                       [11] * 22
                       ), #+ [-1] * 34 ),
             ])    
        
        for l in open(strain_fname, 'r'):
            if l[0] == '#': continue
            line = l.strip().split(' ')
            self.strains.append(line[0])
        
        self.strains = pl.Series("strain", values = self.strains)
        f = open(vaccine_fname)
        self.vaccines = json.load(f)
        
        for vaccine, value in self.vaccines.items():
            self.vaccines[vaccine]["daily_schedule"] = \
                [period * (day // period) for day in value["daily_schedule"]]
        
            self.vaccines[vaccine]["on_time_coverage_frac"] = \
                self.vaccines[vaccine]["on_time_coverage_frac"] + \
                    [self.vaccines[vaccine]["on_time_coverage_frac"][-1]] * \
                        int(self.vaccines[vaccine]["years"][1] - \
                            self.vaccines[vaccine]["years"][0] - \
                     len(self.vaccines[vaccine]["on_time_coverage_frac"]) + 1)
            self.vaccines[vaccine]["late_coverage_frac"] = \
                self.vaccines[vaccine]["late_coverage_frac"] + \
                    [self.vaccines[vaccine]["late_coverage_frac"][-1]] * \
                        int(self.vaccines[vaccine]["years"][1] - \
                            self.vaccines[vaccine]["years"][0] - \
                        len(self.vaccines[vaccine]["late_coverage_frac"]) + 1)
            
            #self.vaccines[vaccine]["antibody_multipliers_next_doses"] = \
            #    pl.Series(
            #        self.vaccines[vaccine]["antibody_multipliers_next_doses"])
            self.vaccines[vaccine]["daily_schedule"] = \
                pl.Series(self.vaccines[vaccine]["daily_schedule"])
            self.vaccines[vaccine]["on_time_coverage_frac"] = \
                pl.Series(self.vaccines[vaccine]["on_time_coverage_frac"])
            self.vaccines[vaccine]["late_coverage_frac"] = \
                pl.Series(self.vaccines[vaccine]["late_coverage_frac"])
            self.vaccines[vaccine]["serotypes"] = \
                pl.Series(self.vaccines[vaccine]["serotypes"])
            
        f.close()
        self.vaccine_antibody_df = \
                create_vaccine_antibody_df(self.vaccines, self.strains)
        
    def store_params(self, p):
        """
        If not already present, store simulation parameters in output 
        file and mark file as incomplete.
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
    
    def is_complete(self):
        """
        Check if current output file has been marked as complete.
        """
        return self.h5file.get_node_attr('/params', 'complete')

    def done(self, complete=False):
        """
        Close output file, marking as complete if specified.
        """
        if complete:
            self.h5file.set_node_attr('/params', 'complete', 1)
        self.h5file.close()

    def close(self):
        self.h5file.close()
        
    ### Initialisation and clean-up # # # # # # # # # # # #
    
    #no current epi_burn_in structure 
    
    def add_states(self, *states):
        """Add a new disease state."""
        pass

    

    def add_observers(self, *observers):
        """Add a new observer to the observers list."""
        for observer in observers:
            self.observers[observer.label] = observer
    
    
            
        
    
    def generate_infection_duration_initialization(self, age,
                                                 rng):
        #NOT USED
        """
        Given a duration of infection value, returns a randomly
        selected value for the initialization. It uses uniform distribution 
        with high value sampled from exponential distribution
        
        mean_duration_infection: should be in days rather than the simulation
        tick.
        """
        
        full_duration = \
            self.generate_infection_duration(age, rng) 
       
        
        return rng.randint(0, int(full_duration / self.period)) * self.period

    

    
    def seed_infection(self, day, P, p, rng):
        """Seed initial infection (set everyone else to susceptible)."""
        
        if self.isPopUploaded == False:
            
        
            initial_cases_fname = os.path.join(p['resource_prefix'],
                                                    p['initial_cases'])
            
                
            f = open(initial_cases_fname)
            initial_infections = json.load(f)
            initial_infections["strain_distribution"] = \
                    np.array(initial_infections["strain_distribution"])/ \
                        sum(initial_infections["strain_distribution"])
            f.close()
            #if randomly assign == True, select strains with equal distribution
            if initial_infections["randomly_select_strains"]:
                sampled_strain_indexes = rng.sample(range(len(self.strains)),\
                            initial_infections["randomly_no_initial_strains"])
                frac_values = 1 / initial_infections[
                                            "randomly_no_initial_strains"]
                initial_infections["strain_distribution"] = [frac_values\
                                        if i in sampled_strain_indexes else 0\
                                        for i in range(len(self.strains))]
            
            #initial_infections["strain_distribution"][0] += (1 - \
            #                sum(initial_infections["strain_distribution"]))
            #print(P.I.head)
            
            P.I = (
                    P.I.with_columns(
                exposed_strains = self.strains[rng.choice(len(self.strains),
                                p = initial_infections["strain_distribution"],
                                              size = len(P.I))]
                    )).sort("random", "no_of_strains", 
                            "age_group", descending=False)
            
            
            #TODO: initial duration time distribution can be U(Exp(meanDuration))
            #rather than Exp(meanDuration)
            child_02_yearolds = P.I.filter(pl.col("age") <= 2)
            older_ages = P.I.filter(pl.col("age") > 2)
            
            child_02_yearolds = (child_02_yearolds.with_columns(
                 (pl.col("random") <= 
                  ( 2 * (initial_infections["infected_population_fraction"]))
                 ).alias("will_infected"),
             
                (pl.lit(rng.rand(child_02_yearolds.height)).alias("random")),
                 (pl.lit(rng.exponential(self.duration_of_infection,
                          child_02_yearolds.height)).alias("exp_random")),
                      
                    ))
            older_ages = (older_ages.with_columns(
                       (pl.col("random") <= 
                          (initial_infections["infected_population_fraction"])
                       ).alias("will_infected"),
                   
                       (pl.lit(rng.rand(older_ages.height)).alias("random")),
                    (pl.lit(rng.exponential(self.duration_of_infection,
                                     older_ages.height)).alias("exp_random")),
                    ))
            P.I = pl.concat([child_02_yearolds,older_ages], how= "diagonal")
            infected = (P.I.filter(pl.col("will_infected"))
                .select(["id","age", "strain_list", "endTimes",
                         "no_past_infections", "exp_random","random",
                         "exposed_strains"])
                        )   
    
                
            infected = self.generate_duration_of_infection(day, infected, rng)
            
            
            infected = (infected.with_columns(
                        (pl.col("strain_list").list
                         .concat(pl.col("exposed_strains"))),
                        (pl.col("endTimes").list.concat(
                                self.period * (#pl.col("random") * \
                                   pl.col("exp_random") /self.period).round()
                            )),
                        (pl.col("no_past_infections") + 1),
                        (((pl.col("strain_list").list.lengths()) + 1)
                          .cast(pl.Int32).alias("no_of_strains")),
                            
                       pl.lit(rng.rand(infected.height)).alias("random")
                        ))
                
            P.I = (P.I.update(infected, on="id", how="left")
                   .drop(['will_infected', 'exposed_strains']))
        
        
    def calc_age_group_fois(self):
        #inf_fraction is sorted already
        """
        takes a sorted inf_fraction with length 15.
        It assumes that the inf_fraction is sorted based on age groups.
        
        TODO: make sure that len(inf_fraction) == 15 
              for pop with a small pop size
              
        returns a list of len == 16:
                sum(contacts * infected fraction) in each age group for 
                age groups
        TODO: the value must be a probability between 0-1.
        
        It's ok to not use polars structure as foi has length of 15
        """
        
        foi =  [self.transmission_coef *\
                sum([x * y for x, y in zip(self.foi["inf_fraction"],row)])\
                for row in self.cmatrix.C]
            
        return pl.Series(name = "prob_infection",
            values = [(1 - np.exp(-cur_foi)) for cur_foi in foi])
        
    def calc_foi(self, day, P, cmatrix, rng): 
         """
         Calculates probability of a fully susceptible individual being
         infected through community contact matrix.
         
         Also calculates the strain distribution
         """
         #calculate fraction of total infections in each age group
         #fraction of total infections: 
         #total infected strains / N* self.max_no_coinfections
         #to set the fraction to be between 0-1
         
        
         #assumes that 80+ is already aggregated together  
         #aggregated based on age groups
         #calculate total no of infections and individuals in each group
         #add an infection fraction column
         #sort based on age groups   
         self.foi = (P.I
                     .groupby("age_group").agg(
               pl.col("no_of_strains").sum(),
               pl.count().alias("total_inds") #number of people in each group
              )
              ).with_columns(
              inf_fraction = pl.col("no_of_strains")/(pl.col("total_inds") *\
                                                      self.max_no_coinfections)
              ).sort("age_group")
                  
         missing_ages = [ i for i in list(range(16)) if i not in 
                         self.foi["age_group"]]
         missing_foi_df = pl.DataFrame({
             "age_group" : missing_ages,
             "no_of_strains": [0]* len(missing_ages),
             "total_inds": [0]* len(missing_ages),
             "inf_fraction": [0]* len(missing_ages),

             })
         
         self.foi = pl.concat([self.foi, missing_foi_df], rechunk=True,\
                        how = 'vertical_relaxed').sort("age_group")
         
         #calculate and add foi column
         self.foi = (self.foi.with_columns(
                        (self.calc_age_group_fois()).alias("prob_infection")
                        )).with_columns(
                            
                           ((pl.col("prob_infection")* pl.col("total_inds"))
                            .round(0).cast(pl.Int32))
                           .alias("no_of_exposed"))
         
         
         #if there is at least one infected person
         #calculate strain distribution
         if sum(self.foi["prob_infection"]):
             self.strain_distribution =  ((P.I.select(
                pl.col("strain_list").alias("strain"))
                 .explode("strain")
                ).groupby(['strain']).agg(pl.count())
                 .filter(pl.col("strain") != "null"))
             
             
             #remove null count & create distribution
             self.strain_distribution = (self.strain_distribution
                            .with_columns(
                                (self.strain_distribution['count']/\
                         (self.strain_distribution['count'].sum()))
                            .alias('fraction')).drop("count")
                )
            
            
             """
             #two alternative way to introduce noise
             self.strain_distribution = (self.strain_distribution
                           .with_columns(
                               (pl.col("fraction") * (1 -
                        rng.random(size=self.strain_distribution.height) * \
                            self.noise_in_strain_distribution))
                ).with_columns(
                    pl.col("fraction") * 1 / pl.col("fraction").sum())
                ) 
             
             self.strain_distribution = (self.strain_distribution
                          .with_columns(
                              (pl.col("fraction") * (1 - \
                            rng.permutation((np.arange(0.001,1, 
                                    1/self.strain_distribution.height))) * \
                                   self.noise_in_strain_distribution))
               ).with_columns(
                   pl.col("fraction") * 1 / pl.col("fraction").sum())
               )  
             """
             
             self.strain_distribution = (self.strain_distribution
                           .with_columns(
                               (pl.col("fraction") * (1 -
                        rng.random(size=self.strain_distribution.height) * \
                            self.noise_in_strain_distribution))
                ).with_columns(
                    pl.col("fraction") * 1 / pl.col("fraction").sum())
                ) 
            
             
                 
             
             return True#self.foi["prob_infection"], self.strain_distribution
         else:
             return False #no inf individuals in the community
            
            
    
    #############  Updating # # # # # # # # # # # #
    
    
    def update(self, t, day, P, t_per_year, rng):
        """
        Update the disease state of the population.
        """
        self.check_vaccines(t, day, P, t_per_year, rng)
        #calculate foi
        is_population_infected = self.calc_foi(day, P, self.cmatrix, rng)
        if is_population_infected:
            #at least one inf individual in the community
            cases = self.check_exposure(t, day, P, rng)
        introductions = self.external_exposure(t, day, P, rng)
        new_I = []
        if is_population_infected:
            new_I = self.update_ind_states(t, day, P)
        self.update_observers(t, disease=self, pop=P,
                              day=day,
                              rng=rng)
        return True

    def check_vaccines(self, t, day, P, t_per_year, rng):
        """
        Reset vaccine counts and check for successful vaccinations.
        
        for v in self.vaccines.values():
            v.count = 0
            #v.vaccinate_pop(t, P, self.states, rng)
            if v.check_vaccination_period(t):
                v.vaccinate_pop(t, P, self.states, rng) #vaccination is 
                applied at this time step
                if v.immediately_effective: #vaccine is immediately effective
                    #May42021 -added below line to update state of 
                    individuals-without updating the state counters-.
                    self.update_ind_states_after_vaccination(t, P)
        """
        cur_year = day // 364 
        cur_day_in_year = day % 364
        period = day / t
        """#remove adult vaccinations after year 19
        if cur_year == 18 and cur_day_in_year == 0:
            vacc_adults = (
                P.I.filter((pl.col("age")
                    .is_between(65,101)
                    
                    ) & (
                      pl.col("vaccines").struct.field("no_of_doses") > 1  
                        ))
                   .select(["id", "vaccines"])
            )
            if vacc_adults.height:             
                vacc_adults = vacc_adults.with_columns(
                    vaccines = pl.struct(no_of_doses = pl.lit(0), 
                                     on_time = pl.lit(0).cast(pl.Int64),
                            vaccine_type = pl.lit("").cast(pl.Utf8),
                            final_vaccine_time = \
                                    pl.Series('final_vaccine_time', 
                            [None] * vacc_adults.height, dtype = pl.Int32)))
                    
                P.I = P.I.update(vacc_adults, on= "id", how="left")"""
        
        for vaccine, value in self.vaccines.items():
            #vaccine, value = list(self.vaccines.items())[0]
            #vacc_rollout_year = 0
            #if day > 364 * 2:
            #    print("here")
            if value["years"][0] <= cur_year <= value["years"][1]:
                vacc_rollout_year = cur_year - value["years"][0]
                on_time_coverage = value["on_time_coverage_frac"][\
                                vacc_rollout_year]
                late_coverage = value["late_coverage_frac"][\
                                vacc_rollout_year]
                #time to vaccinate individuals 
                if vaccine.endswith(("_1", "_2", "_adult")):
                    vaccine = vaccine.strip("_1").strip("_2")
                    vacc_age_group = (
                        P.I.filter((pl.col("age")
                            .is_between(value["vaccination_age_range"][0],
                                        value["vaccination_age_range"][1])
                            
                            ) )
                           .select(["id","age", "age_days", 
                                    "vaccines", "random"])
                    )
                    vacc_target_group = (
                        vacc_age_group.filter(
                            (pl.col("vaccines").struct.field("no_of_doses")
                                 < len(value["daily_schedule"]))
                                         ))
                    """if not vacc_target_group.height:
                        on_time_coverage = on_time_coverage * (
                            vacc_age_group.height / vacc_target_group.height)
                        late_coverage = late_coverage * (
                            vacc_age_group.height / vacc_target_group.height)
                        """
                    #on time first dose
                    on_time_first_vacc = (vacc_target_group.filter(
                    (pl.col("age_days") == value["daily_schedule"][0]) &
                          (pl.col("random") <= on_time_coverage)
                         ))
                    late_first_vacc = (vacc_target_group.filter(#on time 
                      (pl.col("age_days") == value["daily_schedule"][0]) &
                          ((on_time_coverage < pl.col("random")) & \
                           (pl.col("random") <= \
                           on_time_coverage + (late_coverage))
                         )))
                                        
                    cont_vacc_target_group = (vacc_age_group.filter(
                       (pl.col("age_days") != value["daily_schedule"][0]) |
                           (pl.col("random") > (on_time_coverage + \
                                               (late_coverage)))
                         ))
                    on_time_first_vacc = (on_time_first_vacc.with_columns(
                            (pl.lit(rng.rand(on_time_first_vacc.height))
                                 .alias("random")),
                        pl.struct(
                    (pl.col("vaccines").struct.field("no_of_doses") + 1),
            (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                            pl.lit(day).alias("final_vaccine_time"),
                            ).alias("vaccines")
                            ))
                        
                elif vaccine.endswith("catchup"):
                    #find pcv7 vaccinated indvs
                    prev_vacc = value["previous_vacc"]
                    vacc_target_group = (
                        P.I.filter((pl.col("age")
                            .is_between(value["vaccination_age_range"][0],
                                        value["vaccination_age_range"][1])
                            
                            ) & (pl.col("vaccines")
                                 .struct.field("no_of_doses") == \
                                  len(\
                        self.vaccines[prev_vacc]["daily_schedule"]) 
                                    ) & \
                                 (pl.col("vaccines")
                                  .struct.field("vaccine_type") == prev_vacc) 
                                         )
                           .select(["id","age", "age_days", 
                                    "vaccines", "random"])
                    )
                    #on time first dose
                    on_time_first_vacc = (vacc_target_group.filter(#on time 
                         (pl.col("age_days") == value["daily_schedule"][0]) &
                         #(pl.col("age") == value["vaccination_age_range"][0]) &
                          (pl.col("random") <= on_time_coverage)
                         ))
                    late_first_vacc = (vacc_target_group.filter(#on time 
                         (pl.col("age_days") == value["daily_schedule"][0]) &
                         #(pl.col("age") == value["vaccination_age_range"][0]) &
                          ((on_time_coverage < pl.col("random")) & \
                           (pl.col("random") <= \
                           on_time_coverage + (late_coverage))
                         )))
                                        
                    cont_vacc_target_group = (vacc_target_group.filter(#on time 
                         (pl.col("age_days") != value["daily_schedule"][0]) |
                         #(pl.col("age") != value["vaccination_age_range"][0]) |
                          (pl.col("random") > (on_time_coverage + \
                                               (late_coverage)))
                         ))
                    on_time_first_vacc = (on_time_first_vacc.with_columns(
                            (pl.lit(rng.rand(on_time_first_vacc.height))
                                 .alias("random")),
                        pl.struct(
                        (pl.col("vaccines").struct.field("no_of_doses") + 1),
                (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                            #(pl.col("vaccines").struct.field("on_time") + 1),
                            pl.lit(day).alias("final_vaccine_time"),
                            
                            ).alias("vaccines")
                            
                            ))
                else:
                    vacc_target_group = (
                        P.I.filter((pl.col("age")
                            .is_between(value["vaccination_age_range"][0],
                                        value["vaccination_age_range"][1])
                            
                            ) & (pl.col("vaccines").struct.field("no_of_doses")
                                 < len(value["daily_schedule"]))
                                 )
                        .select(["id","age", "age_days", "vaccines", "random"])
                        
                    )
               
                             
                                                              
                    #on time first dose
                    on_time_first_vacc = (vacc_target_group.filter(#on time 
                        (pl.col("age_days") == value["daily_schedule"][0]) &
                        (pl.col("age") == value["vaccination_age_range"][0]) &
                         (pl.col("random") <= on_time_coverage)
                        ))
                    late_first_vacc = (vacc_target_group.filter(#on time 
                        (pl.col("age_days") == value["daily_schedule"][0]) &
                        (pl.col("age") == value["vaccination_age_range"][0]) &
                         ((on_time_coverage < pl.col("random")) & \
                          (pl.col("random") <= \
                          on_time_coverage + (late_coverage))
                        )))
                                       
                    cont_vacc_target_group = (vacc_target_group.filter(#on time 
                        (pl.col("age_days") != value["daily_schedule"][0]) |
                        (pl.col("age") != value["vaccination_age_range"][0]) |
                         (pl.col("random") > (on_time_coverage + \
                                              (late_coverage)))
                        ))
                
                    on_time_first_vacc = (on_time_first_vacc.with_columns(
                        (pl.lit(rng.rand(on_time_first_vacc.height))
                             .alias("random")),
                        pl.struct(
                        (pl.col("vaccines").struct.field("no_of_doses") + 1),
                            (pl.lit(1).cast(pl.Int64).alias("on_time")),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                            #(pl.col("vaccines").struct.field("on_time") + 1),
                            pl.lit(day).alias("final_vaccine_time"),
                            
                            ).alias("vaccines")
                        
                        ))
                late_first_vacc = (late_first_vacc.with_columns(
                        (pl.lit(rng.rand(late_first_vacc.height))
                             .alias("random")),
                        pl.struct(
                        (pl.col("vaccines").struct.field("no_of_doses")),
                        
                        #pl.Series("on_time",values = -1 * (
                        #    day + \
                        #((rng.randint((value["late_vacc_age"] * t_per_year - \
                        #    value["daily_schedule"][-1] / period), 
                        # size = late_first_vacc.height) * period))),
                        #        dtype = pl.Int64),
                        pl.Series("on_time",values = -1 * (
                            day + \
                        (((rng.exponential(16, 
                         size = late_first_vacc.height)) // 1) * period) ),
                                dtype = pl.Int64),
                        #(pl.col("vaccines").struct.field("vaccine_type")),
                        (pl.lit(vaccine)).alias("vaccine_type"),
                       (pl.col("vaccines").struct
                         .field("final_vaccine_time")),
                        ).alias("vaccines")
                        
                        ))
                
                
                #on time following doses
                on_time_following_vacc = (cont_vacc_target_group.filter(
                    (pl.col("vaccines").struct.field("on_time") == 1) &\
                (pl.col("vaccines").struct.field("vaccine_type") == vaccine) &\
                  (pl.col("age").is_in(value["vaccination_age_range"][0])) &\
                     (pl.col("age_days").is_in(value["daily_schedule"][1:]))))
                
                cont_vacc_target_group = (cont_vacc_target_group.filter(
                          (pl.col("vaccines").struct.field("on_time") != 1)  |\
                (pl.col("vaccines").struct.field("vaccine_type") != vaccine) |\
           (pl.col("age").is_in(value["vaccination_age_range"][0]).is_not()) |\
                     (pl.col("age_days").is_in(value["daily_schedule"][1:])
                              .is_not())))
                
                on_time_following_vacc = (on_time_following_vacc.with_columns(
                        (pl.lit(rng.rand(on_time_following_vacc.height))
                             .alias("random")),
                        pl.struct(
                          (pl.col("vaccines").struct.field("no_of_doses") + 1),
                            (pl.lit(1).cast(pl.Int64).alias("on_time")),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                            #(pl.col("vaccines").struct.field("on_time") + 1),
                            pl.lit(day).alias("final_vaccine_time"),
                            ).alias("vaccines")
                        ))
                #update random column
                cont_vacc_target_group = cont_vacc_target_group.with_columns(
                     (pl.lit(rng.rand(cont_vacc_target_group.height))
                          .alias("random")),
                     )
                
                late_vacc = (cont_vacc_target_group.filter(
                    (
                     #catch up vaccination that tick
                     (pl.col("vaccines").struct.field("on_time") == -day) &\
                 (pl.col("vaccines").struct.field("vaccine_type")
                  .is_in([vaccine#, value["previous_vacc"]
                          ])) 
                     )
                    ))
                
                cont_vacc_target_group = (cont_vacc_target_group.filter(
                    (
                     #catch up vaccination that tick
                     #(pl.col("random") > \
                     #             late_coverage / t_per_year) |\
                    #(pl.col("id").is_in(on_time_first_vacc["id"]))
                     (pl.col("vaccines").struct.field("on_time") != -day) |
                   (pl.col("vaccines").struct.field("vaccine_type")
                    .is_in([vaccine#, value["previous_vacc"]
                            ]).is_not())    
                     )
                        ))
                current_daily_schedules =  value["daily_schedule"]
                current_daily_schedules = (
                                    current_daily_schedules.extend_constant(
                                        value["daily_schedule"][-1], n=1))
                if vaccine.endswith("catchup"):
                    late_vacc = (late_vacc.with_columns(
                        (pl.lit(rng.rand(late_vacc.height))
                             .alias("random")),
                        
                        pl.struct(
                        ((pl.col("vaccines").struct.field("no_of_doses") + 1)
                        .alias("no_of_doses")),
                         (-1 * (
                             day + \
                                 current_daily_schedules.take(
                     late_vacc["vaccines"].struct.field("no_of_doses") -
                     len(self.vaccines[prev_vacc]["daily_schedule"]) + 1)
                           - current_daily_schedules.take(
                     late_vacc["vaccines"].struct.field("no_of_doses") -
                         len(self.vaccines[prev_vacc]["daily_schedule"]))
                                         
                            )).cast(pl.Int64).alias("on_time"),   
                            #(pl.col("vaccines").struct.field("on_time")),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                             pl.lit(day).alias("final_vaccine_time"),
                            ).alias("vaccines")
                        
                        ))
                    """
                        #on time part alternatives
                        
                        #(pl.col("vaccines").struct.field("on_time")),
                        
                        (-1 * (
                            day + \
                                current_daily_schedules.take(
                    late_vacc["vaccines"].struct.field("no_of_doses") -
                    len(self.vaccines[prev_vacc]["daily_schedule"]) + 1)
                          - current_daily_schedules.take(
                    late_vacc["vaccines"].struct.field("no_of_doses") -
                        len(self.vaccines[prev_vacc]["daily_schedule"]))
                                        
                           )).cast(pl.Int64).alias("on_time"),
                        """
                else:
                    late_vacc = (late_vacc.with_columns(
                        (pl.lit(rng.rand(late_vacc.height))
                             .alias("random")),
                        
                        pl.struct(
                        ((pl.col("vaccines").struct.field("no_of_doses") + 1)
                        .alias("no_of_doses")),
                            (-1 * (
                                day + \
                                    current_daily_schedules.take(
                        late_vacc["vaccines"].struct.field("no_of_doses") + 1)
                              - current_daily_schedules.take(
                            late_vacc["vaccines"].struct.field("no_of_doses"))
                               )).cast(pl.Int64).alias("on_time"),
                            #(pl.col("vaccines").struct.field("on_time")),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                             pl.lit(day).alias("final_vaccine_time"),
                             ).alias("vaccines")
                        
                        ))
                #print(late_vacc)  
                
                #update random column
                cont_vacc_target_group = cont_vacc_target_group.with_columns(
                     (pl.lit(rng.rand(cont_vacc_target_group.height))
                      .alias("random")),
                     )
                
                #print(cont_vacc_target_group["vaccines"].struct.schema)
                updated_target_group = (pl.concat([on_time_first_vacc, 
                                                late_first_vacc,
                                            on_time_following_vacc,
                                            late_vacc,
                                            cont_vacc_target_group])
                                        .unnest("vaccines"))
                P.I = P.I.unnest("vaccines")
                P.I = P.I.update(updated_target_group, on="id", how="left") 
                P.I = (P.I.with_columns(pl.struct(pl.col(["no_of_doses",
                                 "on_time", "vaccine_type", 
                                 "final_vaccine_time" ]))
                                        .alias("vaccines"))
                           .drop(["no_of_doses",
                                            "on_time", "vaccine_type", 
                                            "final_vaccine_time"]))
                """
                updated_target_group = pl.concat([on_time_first_vacc, 
                                                late_first_vacc,
                                            on_time_following_vacc,
                                            late_vacc,
                                            cont_vacc_target_group])
                
                updated_target_group.filter(
                pl.col("vaccines").struct.field("on_time") < 0)["vaccines"]"""
                #P.I = P.I.update(updated_target_group, on="id", how="left") 
                #P.I.filter(pl.col("id") ==rid)["on_time"]
                #print(P.I["vaccines"].struct.schema)
                
                """ (P.I.filter(pl.col("vaccines")
                        .struct.field("on_time") < 0)["vaccines"])
                
                (P.I.update(updated_target_group, on="id", how="left")
                .filter(pl.col("vaccines")
                        .struct.field("on_time") < 0)["vaccines"])"""
        
 
        
    def check_exposure(self, t, day, P, rng):
        """
        TODO: function description
        
        """
        pop_size = P.I.height
        initial_pop = P.I
        prob_infection = pl.Series("prob_infection", 
                                   [self.foi["prob_infection"]])
            
        #assign strains to individuals to which are going to be exposed
        P.I = (
                    P.I.with_columns(
                    exposed_strains = self.strain_distribution["strain"][
                        rng.choice(len(self.strain_distribution["strain"]),
                                    p = self.strain_distribution["fraction"],
                                              size = len(P.I))]
             )).sort("random", "no_of_strains", "age_group", descending=False)
        
        
                
        
        #identify individuals that are going to be infected by exposed strains
        P.I = (P.I.with_columns(
                   (  pl.col("random") <= (\
                (1 * (pl.col("no_of_strains") < self.max_no_coinfections)) *\
                    (prob_infection.list.take(P.I["age_group"])[0] *\
                     (1 -  (self.reduction_in_susceptibility) * \
                          (pl.col("no_of_strains"))) * \
                            (1 - \
            pl.col("exposed_strains").is_in(pl.col("strain_list")))     
                                 )
                   )).alias("will_infected"),
               
                   (pl.lit(rng.rand(P.I.height)).alias("random")),
                   ))
        
        
        
        recovered = (
            P.I.filter((pl.col("endTimes").list.eval(pl.element()
                            .filter(pl.element() <= day))).list.lengths() > 0)
            .select(["id", "strain_list", "endTimes", "no_of_strains"])
        )
        
       
        
        recovered = (
                    recovered.with_columns(
                    (pl.col("endTimes").list.eval(pl.element()
                             .filter(pl.element() > day))),
                    (pl.col("strain_list").list.eval(pl.element()).list
                     .take(pl.col("endTimes").list
                .eval((pl.element() > day).arg_true()))),
                   ).with_columns(
                       (((pl.col("strain_list").list.lengths()))
                         .cast(pl.Int32).alias("no_of_strains"))))
                       #.filter(pl.col("endTimesIndexes").list.lengths() > 0)
                    
        P.I = P.I.update(recovered, on="id", how="left")
        
        
        
        
        
                                
                                
        #add infected strains to "strain_list" and add endTime for them       
        infected = (
            P.I.filter(pl.col("will_infected"))
            .select(["id","age","age_days", "strain_list", "endTimes","random",
                     "no_past_infections", "exp_random","vaccines", "quantile",
                     "exposed_strains"])
        )
        
        possibly_infected = infected.filter(
                        pl.col("vaccines").struct.field("no_of_doses") > 0)
        will_infected = infected.filter(
                        pl.col("vaccines").struct.field("no_of_doses") == 0)
        
        if possibly_infected.height:
            
            possibly_infected = possibly_infected.join(
                self.vaccine_antibody_df, 
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                pl.col("vaccines").struct.field("no_of_doses"),
                pl.col("exposed_strains")],
                right_on = ["vaccine_type","no_of_doses",
                            "exposed_strains"],
                how="left").drop("vaccine_type","no_of_doses") 
            possibly_infected.filter(pl.col("id").is_duplicated())

            possibly_infected.filter(pl.count().over("id") > 1)
            possibly_infected.filter(pl.count("id").over(pl.col("id")) > 1)

            log_antibodies = pl.DataFrame([
                pl.Series("meanlog" , possibly_infected["meanlog"]),
                pl.Series("log_antibodies" ,lognorm.ppf(possibly_infected["quantile"], 
                                         possibly_infected["sdlog"], 
                        loc=possibly_infected["meanlog"],
                        scale=1,
                        )),
                pl.Series("waning_ratio" ,
                waning_ratio(
                possibly_infected["vaccines"].struct.field("vaccine_type"), 
                day, 
              possibly_infected["vaccines"].struct.field("final_vaccine_time"),
              self.waning_halflife_day_adult, self.waning_halflife_day_child
              ))
                ])
            
            log_antibodies = log_antibodies.with_columns(
                pl.when(pl.col("meanlog") > -9)
                .then( (pl.col("log_antibodies") + 9) * \
                      (pl.col("waning_ratio")) - 9)
                .otherwise(pl.col("meanlog")).alias("waning_log_antibodies")
                )["waning_log_antibodies"]
            
            
            #np.mean(lognorm.rvs([1],loc=0, 
            #scale=np.exp(1), size=1000000, random_state=None))
            
            
            prob_of_transmission = ( 1 / \
                            (1 + self.prob_acq_logantibody_scale * \
                            np.exp( self.prob_acq_logantibody_shape * \
                          (log_antibodies - self.prob_acq_logantibody_shift))))  
            #prob_of_transmission = (prob_of_transmission *
            #                            (1 / prob_of_transmission.max()))
            
            P.vaccinated_acq_pop = (possibly_infected.with_columns(
               prob_of_transmission.alias("rel_prob_transmission"),
               log_antibodies,
               (pl.col("random") <= prob_of_transmission)
               .alias("will_infected")
               ).filter(pl.col("meanlog")> -9)
                ).unnest("vaccines").select("id", "age","quantile",
                                        "will_infected",
                         "final_vaccine_time", "vaccine_type", "no_of_doses",
                         "rel_prob_transmission", "meanlog", "sdlog",
                         "waning_log_antibodies", "exposed_strains")
           
           
            will_infected_vacc = (possibly_infected.filter(
                ((pl.col("random") <= (prob_of_transmission)
                  ).alias("will_infected")),
                ))
                
            will_infected_vacc = will_infected_vacc.with_columns(
                (pl.lit(rng.rand(will_infected_vacc.height))
                 .alias("random"))).drop("meanlog", "sdlog")
                    
            not_infected = (possibly_infected.filter(
                ~pl.col("id").is_in(will_infected_vacc["id"])))
             
            not_infected = not_infected.with_columns(
                            (pl.lit(rng.rand(not_infected.height))
                             .alias("random")))
            
            will_infected = pl.concat([will_infected, will_infected_vacc],\
                                      rechunk=True, how = 'diagonal')
          
        
        
        will_infected = self.generate_duration_of_infection(
                        day, will_infected, rng)
        
        #print(will_infected.filter(pl.col("age") == 0)["exp_random"].mean()) 
        #print(will_infected.filter(pl.col("age") > 18)["exp_random"].mean()) 
        
        will_infected = (will_infected.with_columns(
                    (pl.col("strain_list").list
                     .concat(pl.col("exposed_strains"))),
                    (pl.col("endTimes").list.concat( day +
                               self.period * \
                                (pl.col("exp_random") /self.period).round())),
                    (pl.col("no_past_infections") + 1),
                    (((pl.col("strain_list").list.lengths()) + 1)
                      .cast(pl.Int32).alias("no_of_strains")),
                    ))
        
        if pop_size != P.I.height:
            print("pop size has changed")
            
        will_infected_size = will_infected.height
        #will_infected = self.check_disease(t, day, P, will_infected, rng)
        
        if will_infected_size != will_infected.height:
            print("will_infected size has changed after checking disease")
        if possibly_infected.height and not_infected.height:
            
            updated_infections = pl.concat([will_infected, not_infected],\
                                  rechunk=True, how = 'diagonal')
           
        else:
            updated_infections = will_infected
        
        
        P.I = P.I.update(updated_infections, on="id", how="left")
        #check disease
        infected_pop = P.I.filter(pl.col("no_of_strains") >= 1)
        infected_pop = self.check_disease(t, day, P, infected_pop, rng)
        P.I = P.I.update(infected_pop, on="id", how="left")
        
        if pop_size != P.I.height:
            print("pop size has changed after check the disease")
        
        #et = time.time()
        #print(et - st)
        
        
        
        return []

    def check_disease(self,t,day, P, infected, rng):
        
        #cur_disease_pop = infected.sample(3).with_columns(
        #    pl.Series("disease", ["ipd", "cap", "ipd"])
        #    )
        initial_inf = infected
        #infected = initial_inf
        initial_len = infected.height
        initial_no_cols = len(infected.columns)
        infected = infected.join(self.vaccine_antibody_df, 
            left_on = [pl.col("vaccines").struct.field("vaccine_type"),
            pl.col("vaccines").struct.field("no_of_doses"),
            pl.col("exposed_strains")],
            right_on = ["vaccine_type","no_of_doses", 
                        "exposed_strains"],
            how="left").drop("vaccine_type","no_of_doses")
        
        
        
        
        """
            self.vaccine_antibody_df["age"].unique()
            self.vaccine_antibody_df.with_columns(
                pl.arange(pl.col("age").list.get(0),
                          pl.col("age").list.get(1)).list()
                .over(pl.col("age")).alias("age_list"))
            
            
            pl.arange(self.vaccine_antibody_df["age"].list.get(0),
                      self.vaccine_antibody_df["age"].list.get(1))
            .list().over(self.vaccine_antibody_df["age"])
            
        """
       
        
            
        #infected = infected.with_columns((pl.when(pl.col("age") > 64)
        #                                .then(13 + 4)
        #                .otherwise(4 + pl.col("age") // 5)).alias("age_coef"))
        infected = infected.join(self.age_0_conversion, on= ["age", "age_days"],
                                   how="left")
        """infected = infected.join(self.age_0_conversion,
                        on= ["age", "age_days"],how="left").with_columns(
                                (pl.when(pl.col("age_coef").is_null())
                                .then(pl.col("age")+4)
                                .otherwise("age_coef")).alias("age_coef")
                                )"""
                                   
        
        infected = (infected.update(self.age1_conversion, 
                on= ["age"], how="left"))
        
        infected = infected.filter(pl.col("age_coef") >= 0)
        
        #group them based on vaccine received
        infected_novacc = infected.filter(
                    pl.col("vaccines").struct.field("no_of_doses") == 0)
        
        infected_vacc = infected.filter(
                    pl.col("vaccines").struct.field("no_of_doses") > 0)
        
        
        
        
        log_antibodies_vacc = pl.DataFrame([
            pl.Series("meanlog" , infected_vacc["meanlog"]),
            pl.Series("log_antibodies" ,lognorm.ppf(infected_vacc["quantile"], 
                                     infected_vacc["sdlog"], 
                    loc=infected_vacc["meanlog"],
                    scale=1,
                    )),
                pl.Series("waning_ratio" ,
                waning_ratio(
                infected_vacc["vaccines"].struct.field("vaccine_type"), 
                day, 
              infected_vacc["vaccines"].struct.field("final_vaccine_time"),
              self.waning_halflife_day_adult, self.waning_halflife_day_child
              ))
                ])
        
        log_antibodies_vacc = log_antibodies_vacc.with_columns(
            pl.when(pl.col("meanlog") > -9)
            .then( (pl.col("log_antibodies") + 9) * \
                  (pl.col("waning_ratio")) - 9)
            .otherwise(pl.col("meanlog")).alias("waning_log_antibodies")
            )["waning_log_antibodies"]
        
        
        #np.mean(lognorm.rvs([1],loc=0, 
        #scale=np.exp(1), size=1000000, random_state=None))
        
        
        """
        
        log_antibodies_vacc = lognorm.ppf(infected_vacc["quantile"] * (
            waning_ratio(
            infected_vacc["vaccines"].struct.field("vaccine_type"), 
            day, 
          infected_vacc["vaccines"].struct.field("final_vaccine_time"),
          self.waning_halflife_day_adult,self.waning_halflife_day_child)
                ),  infected_vacc["sdlog"], 
                    loc=infected_vacc["meanlog"],
                    scale=1,
                    )"""
        log_antibodies_novacc = np.array([-9] * len(infected_novacc))
        
        
        
        prob_of_disease_vacc = ( 
           self.prob_dis_logantibody_scale[infected_vacc["age_coef"]] / ( 1 + \
                self.prob_dis_logantibody_age[infected_vacc["age_coef"]] * \
                    np.exp( self.prob_dis_logantibody_shape * \
                   (log_antibodies_vacc - 
                    #self.prob_dis_logantibody_shift
                    -1.7 + 1250 *\
                   self.prob_dis_logantibody_scale[infected_vacc["age_coef"]]
                   
                    ))))
        
        """prob_of_disease_vacc = ( 1 / ( 1 + \
                self.prob_dis_logantibody_age[infected_vacc["age_coef"]] * \
                    np.exp( self.prob_dis_logantibody_shape * \
                   (log_antibodies_vacc - self.prob_dis_logantibody_shift))))"""
        
        #if len(prob_of_disease_vacc):
        #    prob_of_disease_vacc = (prob_of_disease_vacc *
        #                            (self.prob_dis_logantibody_scale / \
        #                                     prob_of_disease_vacc.max()))
        
        prob_of_disease_novacc = ( 
         self.prob_dis_logantibody_scale[infected_novacc["age_coef"]] / ( 1 + \
                self.prob_dis_logantibody_age[infected_novacc["age_coef"]] * \
                    np.exp( self.prob_dis_logantibody_shape * \
                   (log_antibodies_novacc - 
                    #self.prob_dis_logantibody_shift
                    -1.7 + 1250 *\
                   self.prob_dis_logantibody_scale[infected_novacc["age_coef"]]
                   ))))
        """prob_of_disease_novacc = ( 1 / ( 1 + \
                self.prob_dis_logantibody_age[infected_novacc["age_coef"]] * \
                    np.exp( self.prob_dis_logantibody_shape * \
                   (log_antibodies_novacc - self.prob_dis_logantibody_shift))))"""
        
        #if len(prob_of_disease_novacc):
        #    prob_of_disease_novacc = (prob_of_disease_novacc *
        #                            (self.prob_dis_logantibody_scale / \
        #                                     prob_of_disease_novacc.max()))
        
        
           
        cur_disease_pop_vacc = infected_vacc.filter(
            (pl.col("random") <= (prob_of_disease_vacc)
              ))
        #P.vaccinated_disease_po
        P.vaccinated_disease_pop = (infected_vacc.with_columns(
            prob_of_disease_vacc,
            log_antibodies_vacc,
            (pl.col("random") <= prob_of_disease_vacc
              ).alias("will_develop_disease")
            )
        .filter(
            (pl.col("meanlog")> -9))
        ).unnest("vaccines").select("id", "age","age_coef","quantile",
                      "final_vaccine_time", "vaccine_type", "no_of_doses",
                      "prob_dis_logantibody_scale", "meanlog", "sdlog",
                      "waning_log_antibodies", "exposed_strains",
                      "will_develop_disease")
        
        
        cur_disease_pop_novacc = infected_novacc.filter(
            (pl.col("random") <= (prob_of_disease_novacc)
              ))
        
        cur_disease_pop = pl.concat([cur_disease_pop_vacc,
                                     cur_disease_pop_novacc ],
                                    how = 'diagonal')
        
        cur_disease_pop = (cur_disease_pop.with_columns(
                        (pl.lit(rng.rand(cur_disease_pop.height))
                         .alias("random")),
                        ).with_columns(
                 pl.when(pl.col("random") <= (self.ipd_fraction_by_age_group[
                     cur_disease_pop["age_coef"]])
                          )
                 .then("ipd")
                 .otherwise("cap").alias("disease") 
                             ))
        P.vaccinated_disease_pop = (P.vaccinated_disease_pop
                                .join(cur_disease_pop.select("id", "disease"),
                                      on="id", how= "left"))
        
        
        
        if day % (364) == 0:
            #set the counter zero
            P.disease_pop = cur_disease_pop
            
        else:
            P.disease_pop = pl.concat([P.disease_pop, cur_disease_pop],
                                      how="diagonal")
        infected = infected.with_columns(
                        (pl.lit(rng.rand(infected.height))
                         .alias("random"))).drop("age_coef", "meanlog","sdlog")
        
        
        if (infected.height != initial_len) or \
            (initial_no_cols != len(infected.columns)) or \
            (infected.drop("random")
             .frame_equal(initial_inf.drop("random")) == False):
            print("not equal!")
        return infected
        
        
    def external_exposure(self, t, day, P, rng):
        """
        Check for external import of infection.
        """
        if day % self.external_exposure_check_period == 0:
            external_strains = (self.strains
                                .sample(self.no_daily_external_strains))
            
            susceptibles = P.I.filter(
                (pl.col("no_of_strains") < 2) &\
                 (pl.col("vaccines").struct.field("no_of_doses") == 0)   
                    
                )
            susceptibles = susceptibles.with_columns(
                exposed_strains = external_strains.sample(susceptibles.height, 
                                            with_replacement = True,
                                            seed = rng.randint(9999)))
            
            
            infected = (
                susceptibles.filter(
                    (pl.col("exposed_strains").is_in("strain_list").is_not()) &\
                (pl.col("random") <= self.expernal_exposure_prob * (
                    P.I.height / susceptibles.height)
                    ) 
                ).select(["id", "age","strain_list", "random", "no_of_strains",
                         "exposed_strains", "endTimes", "exp_random",
                         "no_past_infections"])
            )
            
            infected = self.generate_duration_of_infection( day, infected, rng)
            infected = (infected.with_columns(
                        (pl.col("strain_list").list
                         .concat(pl.col("exposed_strains"))),
                        (pl.col("endTimes").list.concat( day +
                                   self.period * \
                                (pl.col("exp_random") /self.period).round())),
                        (pl.col("no_past_infections") + 1),
                        (((pl.col("strain_list").list.lengths()) + 1)
                          .cast(pl.Int32).alias("no_of_strains")),
                        )).drop("exposed_strains")
            
                           
            P.I = P.I.update(infected, on="id", how="left")
            
            return infected
        else:
            return None
    
    
    def generate_duration_of_infection(self, day, infected,rng):
        infected = infected.join(self.age_specific_duration_of_infections, 
                                 on="age", how="left")  
        
        mean_dur_of_infection = infected["mean_dur_of_infection"]
        random_dur_of_infection = rng.exponential(mean_dur_of_infection)
        infected = (infected.with_columns(
                    pl.lit(random_dur_of_infection).alias("exp_random"))
                    .drop("mean_dur_of_infection"))
        """
        import matplotlib.pyplot as plt

        ages = pl.DataFrame([
            pl.Series("age",rng.choice(range(0,100), 100000))])
        ages = ages.join(self.age_specific_duration_of_infections, 
                                 on="age", how="left")  
        random_dur_of_infection = rng.exponential(
            ages["mean_dur_of_infection"])
        ages = (ages.with_columns(
                    pl.lit(random_dur_of_infection).alias("exp_random"))
                    )
        median_dur = (ages.group_by("age").agg(pl.col("exp_random").median())
                        .sort("age"))
        
        #plotting antibody prob of transmission
        figsize=(7,3.5)#no in x axis, no in yaxis
        fig, ax = plt.subplots(figsize=figsize)
        
        
        ax.plot(ages["age"].to_list(), 
                    ages["exp_random"].to_list(), "o", markersize=2)
        ax.plot(median_dur["age"].to_list(), 
                    median_dur["exp_random"].to_list(),
                    "-", color="black", label="Median", lw=2)
        ax.set_xlabel('Age')
        ax.set_ylabel('Duration of Infection (days)')
        ax.legend()
        fig.savefig(os.path.join("output/", 
                  'dur_of_infection.pdf'),
                  bbox_inches = "tight",dpi=300)
        
           
                
        ax.plot(log_antibodies,
                 prob_of_transmission,
                 #color = "tab:blue"
                 label = "coefficient: %s"%age_logantibody_acq
                 )
        #plt.vlines(x=0,ymin=0, ymax= 1)
        ax.set_xlabel('Age')
        ax.set_ylabel('Duration of Infection')
        ax.legend()
        prob_of_transmission[0]
        prob_of_transmission[-1]
        
        fig.savefig(os.path.join(output_directory, 
                  'prob_of_acq.pdf'),
                  bbox_inches = "tight",dpi=300)
        
        """    
        return infected
       
        
        
    def update_ind_states(self, t, day, P):
        """
        Update disease status of all individuals.
        Returns a list of newly infectious ('symptomatic' individuals)
        
        """
        
        new_I = P.I.filter(pl.col("will_infected"))
        P.I = (P.I.drop("will_infected", "exposed_strains"))
        return new_I

    def update_observers(self, t, **kwargs):
        """
        Store observed data (if observers are switched on).
        """
        if self.obs_on:
            for observer in self.observers.values():
                observer.update(t,  **kwargs)

    ### Information  # # # # # # # # # # # # # # # # # # # # #

    def state_labels(self):
        """
        Return a list of state labels, in specified order.
        """
        return [v.label for v in sorted(
            self.states.values(), key=lambda x: x.order)]

    def state_colors(self):
        """
        Return a list of state labels, in specified order.
        """
        return [v.color for v in sorted(
            self.states.values(), key=lambda x: x.order)]

    def state_counts(self):
        """
        Return a list of state counts, in specified order.
        """
        return [v.count for v in sorted(
            self.states.values(), key=lambda x: x.order)]