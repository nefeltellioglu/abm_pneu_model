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
from itertools import chain
import tomllib
from collections import defaultdict
import time


from .antibody_levels import waning_ratio #create_vaccine_antibody_df, 
from .antibody_levels import create_vaccine_antibody_df
#from .antibody_levels_legacy import create_vaccine_antibody_df
from .disease_utils import looks_like_json_list

class Disease(object):
    """
    Disease class that updates disease state of a population.
    
    An agent-based model that simulated multi-strain pathogen transmission.
    
    :param 
    
    """
        
    def __init__(self, p, cmatrix, fname, mode):
        #########################
        #population parameters
        #########################
        
        # cmatrix: contact matrix 
        self.cmatrix = cmatrix
        self.transmission_rng = np.random.default_rng(p['transmission_seed'])
        self.vaccine_rng = np.random.default_rng(p['vaccine_seed'])
        
        self.disease_rng = np.random.default_rng(p['transmission_seed'])
        seeds = self.disease_rng.integers(0, 2**32 - 1, 
                                    size= p['clinical_model_per_simulation'], 
                                    dtype=np.uint32)
        self.disease_rngs = [np.random.default_rng(seed) for seed in seeds]
        #self.nprng = np.random.RandomState(rng.randint(0, 99999999))
        self.period = 365 // p['t_per_year']
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

        ###############
        #storage: h5file: HDF file for storing output of simulation
        self.h5file = tb.open_file(fname, mode)
        if mode in 'w':
            self.store_params(p)
            
        # obs_on: if True, record output on simulation
        self.obs_on = True

        # observers: data collection objects
        self.observers = {}
        self._load_strain_data(p)
        self._load_vaccine_data(p)
        self._load_disease_data(p)
    
    def _load_strain_data(self,p):
        #load strains
        strain_fname = os.path.join(p['resource_prefix'],
                                                p['strain_list'])
        a_s_specific_disease_multi_fname = os.path.join(p['resource_prefix'],
                             p['age_serotype_specific_disease_multipliers'])
        self.strains_df = pl.read_csv(
                strain_fname,
                comment_prefix="#",
                has_header=True)
        self.strains = pl.Series("strain", 
                                 values = self.strains_df["serotype"].to_list())
        #varying transmission rates
        self.serotype_to_trans_multiplier = dict(zip(
            self.strains_df["serotype"].to_list(),
            self.strains_df["transmission_multipliers"].to_list()
        ))
        self.trans_multiplier_to_serotype = defaultdict(list)
        for serotype, multiplier in self.serotype_to_trans_multiplier.items():
            self.trans_multiplier_to_serotype[str(multiplier)].append(serotype)
        
        grouped = defaultdict(list)
        
        # group serotypes by multiplier
        for serotype, multiplier in self.serotype_to_trans_multiplier.items():
            grouped[multiplier].append(serotype)
        
        # build the final list
        self.grouped_transmission_multipliers = [
            {
                "serotypes": serotypes,
                "trans_multiplier": multiplier,
            }
            for multiplier, serotypes in grouped.items()
        ]
        
        self.age_ST_specific_disease_multipliers = pl.read_csv(
                a_s_specific_disease_multi_fname,
                comment_prefix="#",
                has_header=True)
        self.age_specific_duration_of_infections = pl.read_csv(
                os.path.join(p['resource_prefix'],
               p['age_specific_duration_of_infections']),
                comment_prefix="#",
                has_header=True)
        
    def _load_vaccine_data(self,p):
        
        #adjust rollout days
        period = 365 // p["t_per_year"]
        
        # load vaccine data
        vaccine_fname = os.path.join(p['resource_prefix'],
                                                p['vaccine_list'])
        df = pl.read_csv(vaccine_fname, 
                         comment_prefix="#",
                         has_header=True)
        # Infer list fields from the first row
        sample = df.head(1).to_dicts()[0]
        
        LIST_FIELDS = [col for col, val in sample.items() if 
                       looks_like_json_list(val)]
        # Decode JSON list columns
        for col in LIST_FIELDS:
            s = df[col].map_elements(
                lambda x: json.loads(x) if isinstance(x, str) and x else x
            )
            df = df.with_columns(pl.Series(col, s))
        # Convert to dict-of-dicts keyed by "name"
        self.vaccines = {
            row["name"]: {k: v for k, v in row.items() if k != "name"}
            for row in df.to_dicts()
        }
        
        self.vaccs_only_protective_against_IPD = []
        
        for vaccine, value in self.vaccines.items():
            cur_vacc_seed = self.vaccine_rng.integers(0, 99999999)
            self.vaccines[vaccine]["rng"] = np.random.default_rng(cur_vacc_seed)
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
            
            self.vaccines[vaccine]["daily_schedule"] = \
                pl.Series(self.vaccines[vaccine]["daily_schedule"])
            self.vaccines[vaccine]["on_time_coverage_frac"] = \
                pl.Series(self.vaccines[vaccine]["on_time_coverage_frac"])
            self.vaccines[vaccine]["late_coverage_frac"] = \
                pl.Series(self.vaccines[vaccine]["late_coverage_frac"])
            self.vaccines[vaccine]["serotypes"] = \
                pl.Series("serotypes", values = 
                          sorted(set(self.strains_df
                    .filter(pl.col(self.vaccines[vaccine]["vaccine_given"])
                            )['serotype'])))
            self.vaccines[vaccine]["vaccine_given"] = \
                pl.Series(self.vaccines[vaccine]["vaccine_given"])
            self.vaccines[vaccine]["function"] = \
                pl.Series(self.vaccines[vaccine]["function"])
            if self.vaccines[vaccine]["is_only_protective_against_IPD"]:
                self.vaccs_only_protective_against_IPD.append(vaccine)
        # Map function names to actual functions
        self.vac_functions = {
            "find_paediatric": self.find_paediatric,
            "find_paediatric_catchup": self.find_paediatric_catchup,
            "find_adult": self.find_adult,
            "find_paediatric_final_dose": self.find_paediatric_final_dose,
        }

        self.vaccine_antibody_df, self.acceptable_vaccine_name_endings = \
                create_vaccine_antibody_df(self.vaccines, self.strains)
    
    def find_paediatric_final_dose(self, day, vaccine, prev_vacc, value, vac_rng, 
                        on_time_coverage, late_coverage, P):
        
        prev_vacc_doses = len(self.vaccines[prev_vacc]["daily_schedule"])
        vaccine_name_end = vaccine.split("_")[-1]   
        if vaccine_name_end not in self.acceptable_vaccine_name_endings:
            vaccine = vaccine.split(f"_{vaccine_name_end}")[0]
        
        vacc_target_group = (
                P.I.filter((pl.col("age")
                    .is_between(value["vaccination_age_range"][0],
                                value["vaccination_age_range"][1])
                    
                    ) & (
                (
        (pl.col("vaccines").struct.field("no_of_doses") == prev_vacc_doses) &
        (pl.col("vaccines").struct.field("vaccine_type") ==  prev_vacc) &
        (pl.col("vaccines").struct.field("on_time") == 1))
           )) .select(["id","age", "age_days", "vaccines"])
                    )   
        #on time first dose
        random_numbers = vac_rng.random(vacc_target_group.height)
        on_time_first_vacc = (vacc_target_group.filter(#on time 
            (pl.col("age_days") == value["daily_schedule"][0]) &
            (pl.col("age") == value["vaccination_age_range"][0]) &
             (pl.lit(random_numbers) <= on_time_coverage)
            ))
        late_first_vacc = (vacc_target_group.filter(#on time 
            (pl.col("age_days") == value["daily_schedule"][0]) &
            (pl.col("age") == value["vaccination_age_range"][0]) &
             ((on_time_coverage < pl.lit(random_numbers)) & \
              (pl.lit(random_numbers) <= \
              on_time_coverage + (late_coverage))
            )))
                           
        cont_vacc_target_group = (vacc_target_group.filter(#on time 
            (pl.col("age_days") != value["daily_schedule"][0]) |
            (pl.col("age") != value["vaccination_age_range"][0]) |
             (pl.lit(random_numbers) > (on_time_coverage + \
                                  (late_coverage)))
            ))
    
        on_time_first_vacc = (on_time_first_vacc.with_columns(
            pl.struct(
            (pl.col("vaccines").struct.field("no_of_doses") + 1),
                (pl.lit(1).cast(pl.Int64).alias("on_time")),
                (pl.lit(vaccine)).alias("vaccine_type"),
                pl.lit(day).alias("final_vaccine_time"),
                
                ).alias("vaccines")
            
            ))
        return on_time_first_vacc, late_first_vacc, cont_vacc_target_group
    
    def find_paediatric(self, day, vaccine, prev_vacc, value, vac_rng, 
                        on_time_coverage, late_coverage, P):
        
        vaccine_name_end = vaccine.split("_")[-1]   
        if vaccine_name_end not in self.acceptable_vaccine_name_endings:
            vaccine = vaccine.split(f"_{vaccine_name_end}")[0]
        
        vacc_target_group = (
            P.I.filter((pl.col("age")
                .is_between(value["vaccination_age_range"][0],
                            value["vaccination_age_range"][1])
                
                ) & 
                (pl.col("vaccines").struct.field("no_of_doses")
                     < len(value["daily_schedule"]))
                )
            .select(["id","age", "age_days", "vaccines"])
        )
                                   
        #on time first dose
        random_numbers = vac_rng.random(vacc_target_group.height)
        on_time_first_vacc = (vacc_target_group.filter(#on time 
            (pl.col("age_days") == value["daily_schedule"][0]) &
            (pl.col("age") == value["vaccination_age_range"][0]) &
             (pl.lit(random_numbers) <= on_time_coverage)
            ))
        late_first_vacc = (vacc_target_group.filter(#on time 
            (pl.col("age_days") == value["daily_schedule"][0]) &
            (pl.col("age") == value["vaccination_age_range"][0]) &
             ((on_time_coverage < pl.lit(random_numbers)) & \
              (pl.lit(random_numbers) <= \
              on_time_coverage + (late_coverage))
            )))
                           
        cont_vacc_target_group = (vacc_target_group.filter(#on time 
            (pl.col("age_days") != value["daily_schedule"][0]) |
            (pl.col("age") != value["vaccination_age_range"][0]) |
             (pl.lit(random_numbers) > (on_time_coverage + \
                                  (late_coverage)))
            ))
    
        on_time_first_vacc = (on_time_first_vacc.with_columns(
            pl.struct(
            (pl.col("vaccines").struct.field("no_of_doses") + 1),
                (pl.lit(1).cast(pl.Int64).alias("on_time")),
                (pl.lit(vaccine)).alias("vaccine_type"),
                #(pl.col("vaccines").struct.field("on_time") + 1),
                pl.lit(day).alias("final_vaccine_time"),
                ).alias("vaccines")
            ))
            
        return on_time_first_vacc, late_first_vacc, cont_vacc_target_group
    
    def find_paediatric_catchup(self, day, vaccine, prev_vacc, value, vac_rng, 
                                on_time_coverage, late_coverage, P):
        
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
                        "vaccines"])
        )
              
        #on time first dose
        random_numbers = vac_rng.random(vacc_target_group.height)
        on_time_first_vacc = (vacc_target_group.filter(#on time 
             (pl.col("age_days") == value["daily_schedule"][0]) &
             #(pl.col("age") == value["vaccination_age_range"][0]) &
              (pl.lit(random_numbers) <= on_time_coverage)
             ))
        late_first_vacc = (vacc_target_group.filter(#on time 
             (pl.col("age_days") == value["daily_schedule"][0]) &
             #(pl.col("age") == value["vaccination_age_range"][0]) &
              ((on_time_coverage < pl.lit(random_numbers)) & \
               (pl.lit(random_numbers) <= \
               on_time_coverage + (late_coverage))
             )))
                            
        cont_vacc_target_group = (vacc_target_group.filter(#on time 
             (pl.col("age_days") != value["daily_schedule"][0]) |
             #(pl.col("age") != value["vaccination_age_range"][0]) |
              (pl.lit(random_numbers) > (on_time_coverage + \
                                   (late_coverage)))
             ))
        on_time_first_vacc = (on_time_first_vacc.with_columns(
               
            pl.struct(
            (pl.col("vaccines").struct.field("no_of_doses") + 1),
    (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
                (pl.lit(vaccine)).alias("vaccine_type"),
                #(pl.col("vaccines").struct.field("on_time") + 1),
                pl.lit(day).alias("final_vaccine_time"),
                ).alias("vaccines")
                ))
        return on_time_first_vacc, late_first_vacc, cont_vacc_target_group
    
    def find_adult(self, day, vaccine, prev_vacc, value, vac_rng, 
                   on_time_coverage, late_coverage, P):
        
        vaccine_name_end = vaccine.split("_")[-1]   
        if vaccine_name_end not in self.acceptable_vaccine_name_endings:
            vaccine = vaccine.split(f"_{vaccine_name_end}")[0]
        vaccine = vaccine.strip("_1").strip("_2")
        
        vacc_age_group = (
            P.I.filter((pl.col("age")
                .is_between(value["vaccination_age_range"][0],
                            value["vaccination_age_range"][1])
                
                ) )
               .select(["id","age", "age_days", 
                        "vaccines"])
        )
        vacc_target_group = (
            vacc_age_group.filter(
                (pl.col("vaccines").struct.field("no_of_doses")
                     < len(value["daily_schedule"]))))
        
        #on time first dose
        random_numbers = vac_rng.random(vacc_target_group.height)
        on_time_first_vacc = (vacc_target_group.filter(
        (pl.col("age_days") == value["daily_schedule"][0]) &
              (pl.lit(random_numbers) <= on_time_coverage)
             ))
        late_first_vacc = (vacc_target_group.filter(#on time 
          (pl.col("age_days") == value["daily_schedule"][0]) &
              ((on_time_coverage < pl.lit(random_numbers)) & \
               (pl.lit(random_numbers) <= \
               on_time_coverage + (late_coverage))
             )))
                            
        cont_vacc_target_group = (vacc_target_group.filter(
           (pl.col("age_days") != value["daily_schedule"][0]) |
               (pl.lit(random_numbers) > (on_time_coverage + \
                                   (late_coverage)))
             ))
        
        on_time_first_vacc = (on_time_first_vacc.with_columns(
             pl.struct(
        (pl.col("vaccines").struct.field("no_of_doses") + 1),
        (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
                (pl.lit(vaccine)).alias("vaccine_type"),
                pl.lit(day).alias("final_vaccine_time"),
                ).alias("vaccines")
                ))
        return on_time_first_vacc, late_first_vacc, cont_vacc_target_group
    
    def give_late_and_following_vac_doses(self, day, vaccine,prev_vacc, 
                                   value, vac_rng,
                                   on_time_coverage, 
                                   late_coverage, 
                                   late_first_vacc, 
                                   cont_vacc_target_group, period):
        random_numbers = vac_rng.exponential(16, 
                            size = late_first_vacc.height)
        late_first_vacc = (late_first_vacc.with_columns(
                pl.struct(
                (pl.col("vaccines").struct.field("no_of_doses")),
                pl.Series("on_time",values = -1 * (
                    day + \
                ((1 + random_numbers // 1) * self.period) ),
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
            (pl.col("age").is_in(value["vaccination_age_range"][0]).not_()) |\
             (pl.col("age_days").is_in(value["daily_schedule"][1:]).not_())
             )
           )
        
        on_time_following_vacc = (on_time_following_vacc.with_columns(
                pl.struct(
                  (pl.col("vaccines").struct.field("no_of_doses") + 1),
                    (pl.lit(1).cast(pl.Int64).alias("on_time")),
                    (pl.lit(vaccine)).alias("vaccine_type"),
                    #(pl.col("vaccines").struct.field("on_time") + 1),
                    pl.lit(day).alias("final_vaccine_time"),
                    ).alias("vaccines")
                ))
        
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
             (pl.col("vaccines").struct.field("on_time") != -day) |
           (pl.col("vaccines").struct.field("vaccine_type")
            .is_in([vaccine#, value["previous_vacc"]
                    ]).not_())    
             )
                ))
        current_daily_schedules =  value["daily_schedule"]
        current_daily_schedules = (
                            current_daily_schedules.extend_constant(
                                value["daily_schedule"][-1], n=1))
        
        if late_vacc.height and vaccine.endswith("catchup"):
            late_vacc = (late_vacc.with_columns(
                pl.struct(
                ((pl.col("vaccines").struct.field("no_of_doses") + 1)
                .alias("no_of_doses")),
                 (-1 * (
                     day + \
                         current_daily_schedules.gather(
             late_vacc["vaccines"].struct.field("no_of_doses") -
             len(self.vaccines[prev_vacc]["daily_schedule"]) + 1)
                   - current_daily_schedules.gather(
             late_vacc["vaccines"].struct.field("no_of_doses") -
                 len(self.vaccines[prev_vacc]["daily_schedule"]))
                                 
                    )).cast(pl.Int64).alias("on_time"),   
                    #(pl.col("vaccines").struct.field("on_time")),
                    (pl.lit(vaccine)).alias("vaccine_type"),
                     pl.lit(day).alias("final_vaccine_time"),
                    ).alias("vaccines")
                ))
            
        elif late_vacc.height:
            late_vacc = (late_vacc.with_columns(
                pl.struct(
                ((pl.col("vaccines").struct.field("no_of_doses") + 1)
                .alias("no_of_doses")),
                    (-1 * (
                        day + \
                            current_daily_schedules.gather(
                late_vacc["vaccines"].struct.field("no_of_doses") + 1)
                      - current_daily_schedules.gather(
                    late_vacc["vaccines"].struct.field("no_of_doses"))
                       )).cast(pl.Int64).alias("on_time"),
                    #(pl.col("vaccines").struct.field("on_time")),
                    (pl.lit(vaccine)).alias("vaccine_type"),
                     pl.lit(day).alias("final_vaccine_time"),
                     ).alias("vaccines")
                
                ))
        return (late_first_vacc, on_time_following_vacc,
                    late_vacc, cont_vacc_target_group)
    def _load_disease_data(self,p):
        """
        Load disease data 
        """
        self.transmission_coef = p['transmission_coefficient']
        self.foi = pl.DataFrame()
        self.strain_distribution =  pl.DataFrame()
        self.reduction_in_susceptibility = \
                                p['reduction_in_susceptibility_coinfections']
        self.max_no_coinfections = p["max_no_coinfections"]
        self.external_exposure_prob = p["external_exposure_prob"]
        self.no_daily_external_strains = p["no_daily_external_strains"]
        self.noise_in_strain_distribution = p["noise_in_strain_distribution"]
        age_specific_prot_parameters_fname = os.path.join(p['resource_prefix'],
                                                p['age_specific_prot_parameters'])
        self.age_specific_prot_parameters = pl.read_csv(
                         age_specific_prot_parameters_fname, 
                         comment_prefix="#",
                         has_header=True)
        self.age_specific_prot_parameters_cols = \
            self.age_specific_prot_parameters.columns[2:]
        
        self.ipd_fraction_by_age_group = (self.age_specific_prot_parameters
                    .select("age_coef", "ipd_fraction_by_age_group")
                    .unique().sort("age_coef")["ipd_fraction_by_age_group"])
                    
        self.waning_halflife_day_adult = p["waning_halflife_day_adult"]
        self.waning_halflife_day_child = p["waning_halflife_day_child"]
        self.isPopUploaded = p["read_population"]
        self.pop_reading_address = p["pop_reading_address"]
            
        self.external_exposure_check_period = \
                int(p["t_per_year"] // p['external_exposure_check_per_year'])
            
    #############    
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
    #there is no epi_burn_in structure 

    def add_observers(self, *observers):
        """Add a new observer to the observers list."""
        for observer in observers:
            self.observers[observer.label] = observer
    
    def seed_infection(self, day, P, p):
        """Seed initial infection (set everyone else to susceptible)."""
        
        if self.isPopUploaded == False:
            
            initial_cases_fname = os.path.join(p['resource_prefix'],
                                                    p['initial_cases'])
            initial_infections = pl.read_csv(
                                            initial_cases_fname,
                                            comment_prefix="#",
                                            has_header=True)
            
            initial_infections = initial_infections.with_columns(
                pl.col('initial_prevalence_proportion') / 
                (pl.col('initial_prevalence_proportion').sum()))
            
             
            #if randomly assign == True, select strains with equal distribution
            if p["randomly_select_strains"]:
                sampled_strain_indexes = sorted(self.transmission_rng.choice(
                    len(self.strains),
                            size=p["randomly_no_initial_strains"],
                            replace=False
                            ))
                frac_values = 1 / p["randomly_no_initial_strains"]
                initial_props = [frac_values\
                                        if i in sampled_strain_indexes else 0\
                                        for i in range(len(self.strains))]
                initial_infections = initial_infections.with_columns(
                                        pl.Series(
                                            "initial_prevalence_proportion",
                                            values=initial_props
                                        ))

            rng = self.transmission_rng
            p_inf = p["infected_population_fraction"]
            
            # ------------------------------------------------------------------
            # Compute age-based infection probabilities (vectorised)
            # ------------------------------------------------------------------
            age = P.I["age"].to_numpy()
            
            infection_prob = np.where(
                age <= 2,
                2.0 * p_inf,
                p_inf
            )
            
            # clip to [0,1] to be safe
            infection_prob = np.clip(infection_prob, 0.0, 1.0)
            
            # ------------------------------------------------------------------
            # Sample who gets infected (single RNG call)
            # ------------------------------------------------------------------
            will_infected = rng.random(len(P.I)) < infection_prob
            
            # ------------------------------------------------------------------
            # Sample exposed strains ONLY for infected individuals
            # ------------------------------------------------------------------
            infected_idx = np.nonzero(will_infected)[0]
            n_infected = infected_idx.size
            
            if n_infected > 0:
                sampled_strains = self.strains[
                    rng.choice(
                        len(self.strains),
                        size=n_infected,
                        p=initial_infections["initial_prevalence_proportion"]
                    )
                ]
            
                mask = np.zeros(P.I.height, dtype=bool)
                mask[infected_idx] = True
                
                infected = P.I.filter(mask)
                #infected = P.I.filter(pl.col("id").is_in(infected_idx))
                infected = infected.with_columns(
                            pl.Series("exposed_strains", sampled_strains),
                            pl.lit(True)
                        ).select(["id","age", "strain_list", "endTimes",
                             "no_past_infections",
                             "exposed_strains"])
                infected = self.generate_duration_of_infection(day, infected, 
                                                        rng)
                
                infected = (infected.with_columns(
                            (pl.col("strain_list").list
                             .concat(pl.col("exposed_strains"))),
                            (pl.col("endTimes").list.concat(
                             self.period * (
                                pl.col("dur_of_infection") /self.period).round()
                                )),
                            (pl.col("no_past_infections") + 1),
                            (((pl.col("strain_list").list.len()) + 1)
                              .cast(pl.Int32).alias("no_of_strains"))
                            
                           )).drop("dur_of_infection")
                P.I = (P.I.update(infected, on="id", how="left"))
                """    
                
                prev_02 = (
                    P.I.filter(pl.col("age") <= 2)
                       .select((pl.col("no_of_strains") > 0).mean())
                       .item()
                )

                prev_older = (
                    P.I.filter(pl.col("age") > 2)
                       .select((pl.col("no_of_strains") > 0).mean())
                       .item()
                )

                print(f"Prevalence (age 0–2): {prev_02:.4f}")
                print(f"Prevalence (age >2): {prev_older:.4f}")
                """
             
    def calc_age_group_fois(self):
        #inf_fraction is sorted already
        """
        takes a sorted inf_fraction with length 15.
        It assumes that the inf_fraction is sorted based on age groups.
        """
        #self.transmission_coefficient_multipliers
        i = 0 
        for cur_serotype_group in \
            self.grouped_transmission_multipliers:
                serotype_group = str(cur_serotype_group['trans_multiplier'])
                multiplier = cur_serotype_group['trans_multiplier']
                cur_foi = [self.transmission_coef * multiplier * \
                    sum([x * y for x, y in \
                      zip(self.foi["inf_fraction_%s"%serotype_group],row)])\
                        for row in self.cmatrix.C]
                if i == 0: 
                    foi = cur_foi
                else:
                    foi = [sum(x) for x in zip(foi, cur_foi)]
                i += 1

        return pl.Series(name = "prob_infection",
            values = [(1 - np.exp(-cur_foi)) for cur_foi in foi])
        
    def calc_foi(self, day, P, cmatrix): 
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
                     .group_by("age_group").agg(
               pl.col("no_of_strains").sum(),
               pl.count().alias("total_inds") #number of people in each group
              )
              ).with_columns(
              inf_fraction = pl.col("no_of_strains")/(pl.col("total_inds") *\
                                                      self.max_no_coinfections)
              ).sort("age_group")
                      
         #varying transmission             
         strain_counts = ((P.I.select("age_group",
            pl.col("strain_list").alias("strain"))
             .explode("strain")
            ).group_by(["age_group", 'strain']).agg(pl.count())
             .filter(pl.col("strain") != "null")).with_columns(
                (pl.col("strain").replace(self.serotype_to_trans_multiplier )
                 ).alias("serotype_group")
               ).group_by("age_group", "serotype_group").agg(pl.sum("count"))
         
         
         strain_counts = strain_counts.pivot(values='count', 
                index='age_group', columns='serotype_group').sort("age_group")
         
         if len(strain_counts.columns) < (1 + 
                            len(self.trans_multiplier_to_serotype)):
             cols = list(self.trans_multiplier_to_serotype.keys()) 
             missing_cols = list(set(cols) - set(strain_counts.columns))
             strain_counts = strain_counts.with_columns(
                 [(pl.lit(0)).alias(missing_col) \
                  for missing_col in missing_cols]
                 )
         serotype_mult_group_columns = list(
             self.trans_multiplier_to_serotype.keys())
         
         self.foi = self.foi.join(
             strain_counts, on= "age_group", how = "left").with_columns(
                 (pl.col(serotype_mult_group_columns) / 
                  (pl.col("total_inds") * self.max_no_coinfections))
                 .name.prefix("inf_fraction_")
                 ).sort("age_group").fill_null(strategy="zero")
         
         missing_ages = [ i for i in list(range(len(self.cmatrix.age_classes))) 
                         if i not in self.foi["age_group"]]
         no_missing_ages = len(missing_ages)

         
         if no_missing_ages:
             schema = self.foi.schema
             missing_foi_df = pl.DataFrame(
                {
                    "age_group": missing_ages,
                    **{
                        col: [0] * no_missing_ages
                        for col in schema
                        if col != "age_group"
                    },
                },
                schema=schema,
             )
             self.foi = pl.concat([self.foi, missing_foi_df], rechunk=True,\
                            how = 'diagonal').sort("age_group")

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
                ).group_by(['strain']).agg(pl.count())
                 .filter(pl.col("strain") != "null")).with_columns(
                  pl.col("strain").replace(
                      self.serotype_to_trans_multiplier)
                  .cast(pl.Float64).alias("multipliers") 
                     ).with_columns(
                         (pl.col("multipliers") * pl.col("count"))
                         .alias("adj_count")   
                         )
             
             #remove null count & create distribution
             self.strain_distribution = (self.strain_distribution
                            .with_columns(
                                (self.strain_distribution['adj_count']/\
                         (self.strain_distribution['adj_count'].sum()))
                            .alias('fraction')).drop(
                                "adj_count", "count", "multipliers")
                ).sort("strain")
            
             noise = self.transmission_rng.random(self.strain_distribution.height)
             self.strain_distribution = (self.strain_distribution
                           .with_columns(
                               (pl.col("fraction") * (1 - noise * \
                            self.noise_in_strain_distribution))
                ).with_columns(
                    pl.col("fraction") * 1 / pl.col("fraction").sum())
                )  
             self._strain_array = self.strain_distribution["strain"].to_numpy()
             self._strain_cdf = self.strain_distribution["fraction"].to_numpy().cumsum()
             return True#self.foi["prob_infection"], self.strain_distribution
         else:
             return False #no inf individuals in the community
             
    #############  Updating # # # # # # # # # # # #  

    def update(self, t, day, P, t_per_year, rng):
        """
        Update the disease state of the population.
        """
        self.check_vaccines(t, day, P, t_per_year)
        
        is_population_infected = self.calc_foi(day, P, self.cmatrix)
        if is_population_infected:
            new_infections = self.check_exposure(t, day, P) 
            
        newly_infected_pop_external = self.external_exposure(t, day, P)
        
        if (new_infections.height and 
                isinstance(newly_infected_pop_external, pl.DataFrame)):
            new_infections = pl.concat([new_infections, 
                                        newly_infected_pop_external],
                                       rechunk=True, how = 'diagonal') 
        elif isinstance(newly_infected_pop_external, pl.DataFrame): 
            new_infections = newly_infected_pop_external
        
        if new_infections.height:
            self.check_disease(t, day, P, new_infections)
            
        self.check_recoveries(t, day, P)
            
        self.update_observers(t, disease=self, pop=P,
                              day=day,
                              #cases=cases,#cases=cases['infection'],
                              #introductions=introductions,
                              #new_I=new_I, 
                              rng=rng)
        return True
    
    def check_vaccines(self, t, day, P, t_per_year):
        """
        Check vacc schedules and apply on time and late vaccinations.
        
        """
        cur_year = day // 365 
        cur_day_in_year = day % 365
        period = self.period #day // t
        
        for vaccine, value in self.vaccines.items():
            #vaccine, value = list(self.vaccines.items())[0]
            #vacc_rollout_year = 0
           vac_rng = self.vaccines[vaccine]["rng"]
           if value["years"][0] <= cur_year <= value["years"][1]:
                vacc_rollout_year = cur_year - value["years"][0]
                on_time_coverage = value["on_time_coverage_frac"][\
                                vacc_rollout_year]
                late_coverage = value["late_coverage_frac"][\
                                vacc_rollout_year]
                prev_vacc = value["previous_vacc"]
                func = self.vac_functions[value["function"].name]
                (on_time_first_vacc, late_first_vacc, 
                 cont_vacc_target_group) = func(day, vaccine,prev_vacc, 
                                                value, vac_rng,
                                                on_time_coverage, 
                                                late_coverage, P) 
                
                (late_first_vacc, on_time_following_vacc,
                            late_vacc, cont_vacc_target_group) = (
                                self.give_late_and_following_vac_doses(day, 
                                               vaccine,prev_vacc, 
                                               value, vac_rng,
                                               on_time_coverage, 
                                               late_coverage, 
                                               late_first_vacc, 
                                               cont_vacc_target_group, 
                                               period))
                
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
                
    def check_recoveries(self, t, day, P):
        
        # Filter rows that have at least one endTime <= day
        recovered = P.I.filter(
            pl.col("endTimes").list.eval(pl.element() <= day).list.sum() > 0
        ).select(["id", "strain_list", "endTimes", "no_of_strains"])
        
        base_ids = recovered.select(["id"])
        # Explode the list columns into long format
        recovered_exploded = recovered.explode(["strain_list", "endTimes"])
        
        # Filter out elements where endTime <= day
        recovered_exploded = recovered_exploded.filter(
            pl.col("endTimes") > day
        )
        
        # Regroup back into list columns, aggregating by id
        recovered_grouped = recovered_exploded.group_by("id").agg([
            pl.col("strain_list").alias("strain_list"),
            pl.col("endTimes").alias("endTimes"),
        ])
        
        recovered_grouped = (
                base_ids
                .join(recovered_grouped, on="id", how="left")
                .with_columns([
                    pl.col("strain_list").fill_null(pl.lit([])),
                    pl.col("endTimes")
                    .fill_null(pl.lit([]).cast(pl.List(pl.Float64))),
                ])
            )
        
        # Update no_of_strains
        recovered_grouped = recovered_grouped.with_columns(
            pl.col("strain_list").list.len().cast(pl.Int32).alias("no_of_strains")
        )
        # Update the main DataFrame
        P.I = P.I.update(recovered_grouped, on="id", how="left")          
        
    def check_exposure_legacy(self, t, day, P):
        """
        Not used as it was quite slow
        
        """
        pop_size = P.I.height
        initial_pop = P.I
      
       
        #older version - this was quite slow
        prob_infection = pl.Series("prob_infection", 
                                   [self.foi["prob_infection"]])
        
        
        
        #assign strains to individuals to which are going to be exposed
        n = P.I.height  # faster than len(P.I)

        # sample using inverse CDF
        u = self.transmission_rng.random(n)
        idx = np.searchsorted(self._strain_cdf, u)
        sampled_strains = self._strain_array[idx]
        
        P.I = (
            P.I.with_columns(
                pl.Series("exposed_strains", sampled_strains)
            )
            .sort(["no_of_strains", "age_group"])
        )
        #identify individuals that are going to be infected by exposed strains
        random_numbers = self.transmission_rng.random(P.I.height)

        P.I = (P.I.with_columns(
                   (  pl.lit(random_numbers) <= (\
                (1 * (pl.col("no_of_strains") < self.max_no_coinfections)) *\
                    (prob_infection.list.gather(P.I["age_group"])[0] *\
                     (1 -  (self.reduction_in_susceptibility) * \
                          (pl.col("no_of_strains"))) * \
                            (1 - \
            pl.col("exposed_strains").is_in(pl.col("strain_list")))     
                                 )
                   )).alias("will_infected"),
               ))
        
        #add infected strains to "strain_list" and add endTime for them       
        infected = (
            P.I.filter(pl.col("will_infected"))
            .select(["id","age","age_days", "strain_list", "endTimes",
                     "no_past_infections", "vaccines", "quantile",
                     "exposed_strains"])
        )
        
        possibly_infected = infected.filter(
                        pl.col("vaccines").struct.field("no_of_doses") > 0)
        will_infected = infected.filter(
                        pl.col("vaccines").struct.field("no_of_doses") == 0)
        
        if possibly_infected.height:
            
            #vacc that do not provide any protection
            will_infected_not_protected = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                        self.vaccs_only_protective_against_IPD))
            
            
            possibly_infected = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                            self.vaccs_only_protective_against_IPD).not_())
            
            #if only vacc without any protection, then they will be infected regardless
            will_infected = pl.concat(
                                [will_infected, will_infected_not_protected],\
                                      rechunk=True, how = 'diagonal')
            
                
            possibly_infected = possibly_infected.join(
                self.vaccine_antibody_df, 
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                pl.col("vaccines").struct.field("no_of_doses"),
                "exposed_strains"],
                right_on = ["vaccine_type","no_of_doses",
                            "exposed_strains"],
                how="left").drop("vaccine_type",
                                 "no_of_doses", "exposed_strains_right") 
            
            possibly_infected = possibly_infected.join(
                                    self.age_specific_prot_parameters, 
                                     on= ["age", "age_days"],
                                       how="left")
            possibly_infected = possibly_infected.with_columns(
                            self.calc_log_antibodies_given_vacc(
                                possibly_infected,
                                day,
                                self.waning_halflife_day_adult,
                                self.waning_halflife_day_child,
                            )
                        )
            
            possibly_infected = possibly_infected.with_columns(
                    self.calc_prob_acq_givenlog_antibodies()
                )

            random_numbers = self.transmission_rng.random(possibly_infected.height) 
            P.vaccinated_acq_pop = (possibly_infected.with_columns(
               (pl.lit(random_numbers) <= pl.col("prob_of_transmission"))
               .alias("will_infected")
               ).filter(pl.col("meanlog")> -9)
                ).unnest("vaccines").select("id", "age","quantile",
                                        "will_infected",
                         "final_vaccine_time", "vaccine_type", "no_of_doses",
                         "prob_of_transmission", "meanlog", "sdlog",
                         "log_antibodies", "exposed_strains")
           
            will_infected_vacc = (possibly_infected.filter(
                ((pl.lit(random_numbers) <= pl.col("prob_of_transmission")
                  ).alias("will_infected")),
                )).drop("meanlog", "sdlog", self.age_specific_prot_parameters_cols)
                
            
            will_infected = pl.concat([will_infected, will_infected_vacc],\
                                      rechunk=True, how = 'diagonal')

        will_infected = self.generate_duration_of_infection(
                        day, will_infected, self.transmission_rng)
        
        will_infected = (will_infected.with_columns(
                    (pl.col("strain_list").list
                     .concat(pl.col("exposed_strains"))),
                    (pl.col("endTimes").list.concat( day +
                        self.period * \
                         (pl.col("dur_of_infection") / self.period).round())),
                    (pl.col("no_past_infections") + 1),
                    (((pl.col("strain_list").list.len()) + 1)
                      .cast(pl.Int32).alias("no_of_strains")),
                    )).drop("dur_of_infection")
        if pop_size != P.I.height:
            print("pop size has changed")

        P.I = P.I.update(will_infected, on="id", how="left")
        #check disease
        new_infections = (will_infected
                              .filter((pl.col("will_infected") == True))
                              .explode("strain_list", "endTimes")
                    .filter(pl.col("exposed_strains") == pl.col("strain_list"))
                        )
        #P.I = P.I.update(infected_pop, on="id", how="left")

        return new_infections
              
    def check_exposure(self, t, day, P):
        """
        TODO: function description
        
        """
        pop_size = P.I.height
        initial_pop = P.I
      
        #start = time.perf_counter()
        foi = self.foi.select("age_group", "prob_infection")
        
        # -------------------------------------------------
        # Keep only eligible individuals
        # -------------------------------------------------
        
        eligible = P.I.filter(
            pl.col("no_of_strains") < self.max_no_coinfections
        )
        
        # -------------------------------------------------
        # Count per age_group and join FOI
        # -------------------------------------------------
        
        group_counts = (
            eligible
            .group_by("age_group")
            .len()
            .join(foi, on="age_group", how="left")
        ).sort("age_group")
        
        # -------------------------------------------------
        # Binomial draw per group (FOI only)
        # -------------------------------------------------
        
        group_counts = group_counts.with_columns(
            pl.Series(
                "n_exposed",
                self.transmission_rng.binomial(
                    group_counts["len"].to_numpy(),
                    group_counts["prob_infection"].to_numpy()
                )
            )
        )
        
        # -------------------------------------------------
        # Vectorized random selection within groups
        # -------------------------------------------------
        
        eligible = (
            eligible
            .join(group_counts.select(["age_group", "n_exposed"]),
                  on="age_group",
                  how="left")
            .with_columns(
                pl.Series("rand", self.transmission_rng.random(eligible.height))
            )
            .with_columns(
                pl.col("rand")
                  .rank("ordinal")
                  .over("age_group")
                  .alias("rank_in_group")
            )
        )
        
        pre_selected = eligible.filter(
            pl.col("rank_in_group") <= pl.col("n_exposed")
        )
        
        # -------------------------------------------------
        # Apply reduction in susceptibility 
        # -------------------------------------------------
        
        selected = (
            pre_selected
            .with_columns(
                (
                    1
                    * (1 - self.reduction_in_susceptibility * pl.col("no_of_strains")
                    )
                )
                .clip(lower_bound=0.0)
                .alias("sus_factor")
            )
            .with_columns(
                pl.Series(
                    "rand_sus",
                    self.transmission_rng.random(pre_selected.height)
                )
            )
            .filter(
                pl.col("rand_sus") <= pl.col("sus_factor")
            )
        )
        # -------------------------------------------------
        # Assign exposed strains ONLY to selected
        # -------------------------------------------------
        
        n_selected = selected.height
        
        if n_selected > 0:
            u = self.transmission_rng.random(n_selected)
            idx = np.searchsorted(self._strain_cdf, u, side="right")
            sampled_strains = self._strain_array[idx]
        
            selected = selected.with_columns(
                pl.Series("exposed_strains", sampled_strains),
                pl.lit(True).alias("will_infected")
            ).filter(
                # Keep rows where exposed_strains is NOT already in strain_list
                ~pl.col("exposed_strains").is_in(pl.col("strain_list"))
            )
                    
            infected = selected.select(["id","age","age_days", "strain_list",
                                        "endTimes",
                     "no_past_infections", "vaccines", "quantile",
                     "exposed_strains", "will_infected"])
        else:
            # Create empty DataFrame with the same schema as P.I
            infected = pl.DataFrame(
                    schema={k: v for k, v in P.I.schema.items()}
                )
        
        possibly_infected = infected.filter(
                        pl.col("vaccines").struct.field("no_of_doses") > 0)
        will_infected = infected.filter(
                        pl.col("vaccines").struct.field("no_of_doses") == 0)
        
        if possibly_infected.height:
            
            #vacc that do not provide any protection
            will_infected_not_protected = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                        self.vaccs_only_protective_against_IPD))
            
            
            possibly_infected = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                            self.vaccs_only_protective_against_IPD).not_())
            
            #if only vacc without any protection, then they will be infected regardless
            will_infected = pl.concat(
                                [will_infected, will_infected_not_protected],\
                                      rechunk=True, how = 'diagonal')
            
                
            possibly_infected = possibly_infected.join(
                self.vaccine_antibody_df, 
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                pl.col("vaccines").struct.field("no_of_doses"),
                "exposed_strains"],
                right_on = ["vaccine_type","no_of_doses",
                            "exposed_strains"],
                how="left").drop("vaccine_type",
                                 "no_of_doses", "exposed_strains_right") 
           
            possibly_infected = possibly_infected.join(
                                    self.age_specific_prot_parameters, 
                                     on= ["age", "age_days"],
                                       how="left")
            possibly_infected = possibly_infected.with_columns(
                            self.calc_log_antibodies_given_vacc(
                                possibly_infected,
                                day,
                                self.waning_halflife_day_adult,
                                self.waning_halflife_day_child,
                            )
                        )
            
            possibly_infected = possibly_infected.with_columns(
                    self.calc_prob_acq_givenlog_antibodies()
                )

            random_numbers = self.transmission_rng.random(possibly_infected.height) 
            P.vaccinated_acq_pop = (possibly_infected.with_columns(
               (pl.lit(random_numbers) <= pl.col("prob_of_transmission"))
               .alias("will_infected")
               ).filter(pl.col("meanlog")> -9)
                ).unnest("vaccines").select("id", "age","quantile",
                                        "will_infected",
                         "final_vaccine_time", "vaccine_type", "no_of_doses",
                         "prob_of_transmission", "meanlog", "sdlog",
                         "log_antibodies", "exposed_strains")
           
            will_infected_vacc = (possibly_infected.filter(
                ((pl.lit(random_numbers) <= pl.col("prob_of_transmission")
                  ).alias("will_infected")),
                )).drop("meanlog", "sdlog", self.age_specific_prot_parameters_cols)
                
            
            will_infected = pl.concat([will_infected, will_infected_vacc],\
                                      rechunk=True, how = 'diagonal')

        will_infected = self.generate_duration_of_infection(
                        day, will_infected, self.transmission_rng)
        
        will_infected = (will_infected.with_columns(
                    (pl.col("strain_list").list
                     .concat(pl.col("exposed_strains"))),
                    (pl.col("endTimes").list.concat( day +
                        self.period * \
                         (pl.col("dur_of_infection") / self.period).round())),
                    (pl.col("no_past_infections") + 1),
                    (((pl.col("strain_list").list.len()) + 1)
                      .cast(pl.Int32).alias("no_of_strains")),
                    )).drop("dur_of_infection")
        if pop_size != P.I.height:
            print("pop size has changed")

        P.I = P.I.update(will_infected, on="id", how="left")
        #check disease
        new_infections = (will_infected
                              .filter((pl.col("will_infected") == True))
                              .explode("strain_list", "endTimes")
                    .filter(pl.col("exposed_strains") == pl.col("strain_list"))
                        )
        #P.I = P.I.update(infected_pop, on="id", how="left")

        return new_infections
    
    def calc_prob_acq_givenlog_antibodies(self) -> pl.Expr:
            
        return ((
                1
                /
                (
                    1
                    + pl.col("prob_acq_logantibody_scale")
                    * (
                        pl.col("prob_acq_logantibody_shape")
                        * (
                            pl.col("log_antibodies")
                            - pl.col("prob_acq_logantibody_shift")
                        )
                    ).exp()
                )
            )
            .alias("prob_of_transmission")
        )
        
    def calc_prob_dis_givenlog_antibodies(self) -> pl.Expr:
        return (
                (
                    pl.col("prob_dis_logantibody_scale")
                    /
                    (
                        1
                        +
                        pl.col("prob_dis_logantibody_age")
                        *
                        (
                            (   
                                pl.col("prob_dis_logantibody_shape")
                                *
                                (
                                    pl.col("log_antibodies")
                                    - pl.col("prob_dis_logantibody_additive")
                                    + pl.col("prob_dis_logantibody_adjust")
                                    * pl.col("prob_dis_logantibody_scale")
                                )
                            ).exp()
                        )
                    )
                ).alias("prob_of_disease")
            )
    
    def calc_log_antibodies_given_vacc(
        self,
        df: pl.DataFrame,
        day,
        waning_halflife_day_adult,
        waning_halflife_day_child,
    ) -> pl.Series:
    
        quantile = df["quantile"].to_numpy()
        sdlog = df["sdlog"].to_numpy()
        meanlog = df["meanlog"].to_numpy()
        
        antibodies = lognorm.ppf(
            quantile,
            np.sqrt(sdlog),
            loc=0,
            scale=np.exp(meanlog),
        )
    
        waning = waning_ratio(
            df["vaccines"].struct.field("vaccine_type"),
            day,
            df["vaccines"].struct.field("final_vaccine_time"),
            waning_halflife_day_adult,
            waning_halflife_day_child,
            df["age"],
        )
    
        result = np.where(
            meanlog > -9,
            np.log(antibodies * waning),
            meanlog,
        )
    
        return pl.Series("log_antibodies", result)
    def check_disease(self,t,day, P, infected):
        
        initial_inf = infected
        #infected = initial_inf
        initial_len = infected.height
        initial_no_cols = len(infected.columns)
        
        infected = infected.join(self.vaccine_antibody_df, 
        left_on = [pl.col("vaccines").struct.field("vaccine_type"),
        pl.col("vaccines").struct.field("no_of_doses"),
        "exposed_strains"],
        right_on = ["vaccine_type","no_of_doses", 
                    "exposed_strains"],
        how="left").drop("vaccine_type","no_of_doses", 
                         "exposed_strains_right")
        
        infected = infected.join(self.age_specific_prot_parameters, 
                                 on= ["age", "age_days"],
                                   how="left")
        
        infected = infected.join(self.age_ST_specific_disease_multipliers,
                                 left_on = ["age","exposed_strains"],
                                 right_on = ['age', 'strains'])
        
        #group them based on vaccine received
        infected_novacc = infected.filter(
                    pl.col("vaccines").struct.field("no_of_doses") == 0)
        
        infected_vacc = infected.filter(
                    pl.col("vaccines").struct.field("no_of_doses") > 0)
        
        #disagg vaccinated individuals based on 
        #vaccines that only protects against IPD
        infected_vacc_prot_aganst_IPD = infected_vacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    self.vaccs_only_protective_against_IPD))
        random_numbers = self.disease_rng.random(infected_vacc_prot_aganst_IPD.height)
        infected_vacc_against_IPD_protected = (infected_vacc_prot_aganst_IPD
                                               .filter(
                 pl.lit(random_numbers) <= 
                 pl.col("ipd_fraction_by_age_group")))
        
        infected_vacc_against_IPD_not_protected = (infected_vacc_prot_aganst_IPD
                                             .filter(
                 pl.lit(random_numbers) > pl.col("ipd_fraction_by_age_group")))
        
        infected_vacc = infected_vacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    self.vaccs_only_protective_against_IPD).not_())
        
        infected_vacc = pl.concat(
            [infected_vacc, infected_vacc_against_IPD_protected],\
                                  rechunk=True, how = 'diagonal')
        infected_novacc = pl.concat(
            [infected_novacc, infected_vacc_against_IPD_not_protected],\
                                  rechunk=True, how = 'diagonal')
            
        infected_vacc = infected_vacc.with_columns(
                        self.calc_log_antibodies_given_vacc(
                            infected_vacc,
                            day,
                            self.waning_halflife_day_adult,
                            self.waning_halflife_day_child,
                        )
                    )
        
        infected_novacc = infected_novacc.with_columns(pl.lit(-9)
                        .cast(pl.Float64()).alias("log_antibodies"))
        
        infected_vacc = infected_vacc.with_columns(
                self.calc_prob_dis_givenlog_antibodies()
            )
        infected_novacc = infected_novacc.with_columns(
                self.calc_prob_dis_givenlog_antibodies()
            )

        for cur_run, dis_rng in enumerate(self.disease_rngs):
            
            random_numbers = self.disease_rng.random(infected_vacc.height)
            cur_disease_pop_vacc = infected_vacc.filter(
                (pl.lit(random_numbers) <= (pl.col("prob_of_disease") 
                                            * pl.col("dis_multiplier"))
                  ))
            P.vaccinated_disease_pop = (infected_vacc.with_columns(
                (pl.lit(random_numbers) <= (pl.col("prob_of_disease") *
                                            pl.col("dis_multiplier"))
                  ).alias("will_develop_disease")
                )
            .filter(
                (pl.col("meanlog")> -9))
            ).unnest("vaccines").select("id", "age","age_coef","quantile",
                          "final_vaccine_time", "vaccine_type", "no_of_doses",
                          "prob_dis_logantibody_scale", "meanlog", "sdlog",
                          "log_antibodies", "exposed_strains",
                          "will_develop_disease")
            
            random_numbers = self.disease_rng.random(infected_novacc.height)
            cur_disease_pop_novacc = infected_novacc.filter(
              (pl.lit(random_numbers) <= (pl.col("prob_of_disease") *
                                          pl.col("dis_multiplier"))
                  ))
            
            infected_novacc_vacc_against_IPD = infected_novacc.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                        self.vaccs_only_protective_against_IPD))
                
            cur_disease_vacc_not_vacc_against_IPD = (cur_disease_pop_vacc.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                    self.vaccs_only_protective_against_IPD).not_()))
            cur_disease_vacc_vacc_against_IPD = (cur_disease_pop_vacc.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                        self.vaccs_only_protective_against_IPD)))
            
            cur_disease_novacc_not_vacc_against_IPD = (cur_disease_pop_novacc.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                    self.vaccs_only_protective_against_IPD).not_()))
            cur_disease_novacc_vacc_against_IPD = (cur_disease_pop_novacc.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                        self.vaccs_only_protective_against_IPD)))
            
            cur_disease_ipd_or_cap = pl.concat([cur_disease_vacc_not_vacc_against_IPD,
                                         cur_disease_novacc_not_vacc_against_IPD],
                                        how = 'diagonal')
            
            random_numbers = self.disease_rng.random(cur_disease_ipd_or_cap.height)
            cur_disease_ipd_or_cap = (cur_disease_ipd_or_cap.with_columns(
                     pl.when(pl.lit(random_numbers) <= 
            (self.ipd_fraction_by_age_group[cur_disease_ipd_or_cap["age_coef"]]))
                     .then(pl.lit("ipd"))
                     .otherwise(pl.lit("cap")).alias("disease") 
                                 ))
                                
            cur_disease_just_ipd = (cur_disease_vacc_vacc_against_IPD.with_columns(
                                pl.lit("ipd").alias("disease") ))
            
            cur_disease_just_cap = (cur_disease_novacc_vacc_against_IPD.with_columns(
                                pl.lit("cap").alias("disease")))
            
            
            cur_disease_pop = pl.concat([cur_disease_ipd_or_cap,
                                         cur_disease_just_ipd,
                                         cur_disease_just_cap],
                                        how = 'diagonal') 
            
            cur_disease_pop = cur_disease_pop.with_columns(
                pl.lit(cur_run).alias("clinical_model_no"))
            
            P.vaccinated_disease_pop = (P.vaccinated_disease_pop
                                    .join(cur_disease_pop.select("id", "disease"),
                                          on="id", how= "left"))
            period = day / t
            if day % (365) < period:
                #set the counter zero
                P.disease_pop = cur_disease_pop
                
            else:
                P.disease_pop = pl.concat([P.disease_pop, cur_disease_pop],
                                          how="diagonal")
                
        infected = infected.drop(self.age_specific_prot_parameters_cols, 
                                 "meanlog", "sdlog", "dis_multiplier")
        
        if (infected.height != initial_len) or \
            (initial_no_cols != len(infected.columns)) or \
            ((infected.sort("id")).equals(initial_inf.sort("id")) == False):
            print("not equal!")
        #return infected
        
    def external_exposure(self, t, day, P):
        """
        Check for external import of infection.
        """
        if day % self.external_exposure_check_period == 0:
            seed = self.transmission_rng.integers(0, 2**32 - 1)
            external_strains = (self.strains
                                .sample(self.no_daily_external_strains, 
                                        seed = seed))
            
            susceptibles = P.I.filter(
                (pl.col("no_of_strains") < 2) &\
                 (pl.col("vaccines").struct.field("no_of_doses") == 0)   
                )
                
            susceptibles = susceptibles.with_columns(
                exposed_strains = external_strains.sample(susceptibles.height, 
                                            with_replacement = True,
                                 seed = self.transmission_rng.integers(9999)))
            random_numbers = self.transmission_rng.random(susceptibles.height)
            infected = (
                susceptibles.filter(
                    (pl.col("exposed_strains").is_in("strain_list").not_()) &\
                (pl.lit(random_numbers) <= self.external_exposure_prob * (
                    P.I.height / susceptibles.height)
                    ) 
                ).with_columns(pl.lit(True).alias('will_infected'))
                    .select(["id", "age", 'age_days','days_at_death',
                          'age_group', 'quantile', 'vaccines', 'no_of_strains',
                        'strain_list',   "endTimes", "no_past_infections", 
                        "exposed_strains", "will_infected"])
            )
            
            infected = self.generate_duration_of_infection( day, infected, 
                                                        self.transmission_rng)
            newly_infected = (infected.with_columns(
                        (pl.col("strain_list").list
                         .concat(pl.col("exposed_strains"))),
                        (pl.col("endTimes").list.concat( day +
                                   self.period * \
                                (pl.col("dur_of_infection") /self.period).round())),
                        (pl.col("no_past_infections") + 1),
                        (((pl.col("strain_list").list.len()) + 1)
                          .cast(pl.Int32).alias("no_of_strains")),
                        )).drop("dur_of_infection")
            infected = newly_infected.drop("exposed_strains", 'will_infected')
            
                           
            P.I = P.I.update(infected, on="id", how="left")
            newly_infected = (newly_infected
                                  .filter((pl.col("will_infected") == True))
                                  .explode("strain_list", "endTimes")
                         .filter(
                             pl.col("exposed_strains") == pl.col("strain_list"))
                            )
            return newly_infected
        else:
            return None
    
    def generate_duration_of_infection(self, day, infected, rng):
        infected = infected.join(self.age_specific_duration_of_infections, 
                                 on="age", how="left")  
        
        mean_dur_of_infection = infected["mean_dur_of_infection"]
        random_dur_of_infection = rng.exponential(mean_dur_of_infection)
        infected = (infected.with_columns(
                    pl.lit(random_dur_of_infection).alias("dur_of_infection"))
                    .drop("mean_dur_of_infection"))

        return infected


    def update_observers(self, t, **kwargs):
        """
        Store observed data (if observers are switched on).
        """
        if self.obs_on:
            for observer in self.observers.values():
                observer.update(t,  **kwargs)
