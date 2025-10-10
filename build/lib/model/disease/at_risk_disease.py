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
from .vaccine import create_vaccine_antibody_df, waning_ratio
from .disease import Disease

class AtRiskDisease(Disease):
    """
    Disease class that updates disease state of a population.
    
    An agent-based model that simulated multi-strain pathogen transmission.
    
    :param 
    
    """
    def __init__(self, p, cmatrix,rng, fname, mode):
        super(AtRiskDisease, self).__init__(p, cmatrix,rng, fname, mode)
    
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
             ]) """   
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
        
        ########################at_risk data loading#####################
        
        at_risk_perc = pl.read_csv(os.path.join(p['resource_prefix'],
                                 "at_risk/at_risk_percentages.csv"),
                                        truncate_ragged_lines=True)
        at_risk_perc = at_risk_perc.with_columns(
            (pl.col("at_risk_tier_1")/100),
            ((pl.col("at_risk_tier_1") + pl.col("at_risk_tier_2"))/100)
            .alias("at_risk_tier_2"),)
        
        at_risk_perc = at_risk_perc.filter(
            pl.col("Ind_status") == "nonindigenous").drop("Ind_status")
        
        age_group_list = []
        self.at_risk_changing_ages = []
        for cur_group in at_risk_perc["age_group"].to_list():
            #cur_group = at_risk_perc["age_group"].to_list()[0]
            if cur_group == at_risk_perc["age_group"].to_list()[0]:
                repeat = int(
                    cur_group.split("–")[0]) 
                age_group_list = age_group_list + [""] * repeat
            
            repeat = \
            int(cur_group.split("–")[1]) - \
            int(cur_group.split("–")[0]) + 1
            age_group_list = age_group_list + ([cur_group] * repeat)
            
            
            self.at_risk_changing_ages.append(int(cur_group.split("–")[0]))
            
            if cur_group == at_risk_perc["age_group"].to_list()[-1]:
                repeat = 102 - \
                int(cur_group.split("–")[1]) - 1
                age_group_list = age_group_list + [""] * repeat
                self.at_risk_changing_ages.append(
                    int(cur_group.split("–")[1]) + 1)
            
        self.at_risk_changing_ages = pl.DataFrame([
            pl.Series("age", self.at_risk_changing_ages),
              pl.Series("at_risk_tier_1",
                        at_risk_perc["at_risk_tier_1"].to_list() + [0]),
              pl.Series("at_risk_tier_2",
                        at_risk_perc["at_risk_tier_2"].to_list() + [0]),
              
             ]) 
        self.at_risk_percentages = pl.DataFrame([
            pl.Series("age", list(range(0,102))),
             pl.Series("at_risk_age", age_group_list),
             ])  
        self.at_risk_percentages = self.at_risk_percentages.join(
            at_risk_perc, left_on= "at_risk_age", right_on= "age_group",
            how = "left"
            )
        
    
         
    
    #############  Updating # # # # # # # # # # # #
    
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
        
        
        
        
    