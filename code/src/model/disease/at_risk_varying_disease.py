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
from .vaccine import create_vaccine_antibody_df, waning_ratio, roll_v116
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
        print(p["prefix"])
        
        self.strains
        self.update_inital_population = False
        #varying transmission rates
        self.transmission_coefficient_multipliers1 = \
            p['transmission_coefficient_multipliers']
        self.serotype_groups1 = { "pcv7": ['4',
                                    '6B', '9V', '14', '18C', '19F', '23F'],
                                "pcv13": ['1', '3', '5', '6A', '7F',  '19A'],
                                "nonpcv13": ['33A', '25F', '15A', '7B', '41A',
                    '40', '44', '11E', '18B', '7C', '11D', '34', '48', '35B', 
                    '20B', '20A', '47A', '23A', '11B', '17F', '13', '15B',
                    '39', '31', '41F', '12B', '32A', '32F', '35A', '35D', 
                    '16F', '9L', '15C', '6D', '24F', '18A', '21', '24C', '22A',
                    '6C', '28A', '10B', '47F', '2', '9N', '7D', '22F', '29',
                    '33D', '46', '45', '10A', '24B', '11C', '33C', '35C', '20',
                    '23B', '28F', '12F', '43', '7A', '37', '15F', '24A', '9A',
                    '8', '18F', '33F', '19C', '10D', '27', '33B', '35F', '42',
           '6H', '10F', '38', '16A', '25A', '36', '19B', '10C', '17A', '11A']}
        
        #print(len(serotype_groups1["nonpcv13"]))
        #print(len(serotype_groups1["pcv13"]))
        #print(len(serotype_groups1["pcv7"]))
        
        ppv23_serotypes = ["1", "2", "3", "4", "5", "6B", "7F", "8", "9N",
                           "9V", "10A", "11A", "12F", "14", "15B", "17F", 
                           "18C", "19A", "19F", "20", "22F", "23F", "33F"]
        pcv13_ppv23 = list(set(self.serotype_groups1["pcv13"]
                               ).union( set(ppv23_serotypes)))
        pcv13_ppv23_nonpcv7 = list(set(pcv13_ppv23
                               ) - (set(self.serotype_groups1["pcv7"])))
        others = (list((set(self.serotype_groups1["nonpcv13"]) - set(
                                                    ppv23_serotypes))))
        
        ppv23_nonpcv13_nonpcv7 = list((set(ppv23_serotypes) -
                                  (set(self.serotype_groups1["pcv7"]))) -
                                  (set(self.serotype_groups1["pcv13"])))
        
        others = (list((set(self.serotype_groups1["nonpcv13"]) - set(
                                                    ppv23_serotypes))))
        self.ppv23_nonpcv13 = ppv23_nonpcv13_nonpcv7
        
        #non_ppv23_nonpcv13 = list(set(self.serotype_groups1["nonpcv13"]) -\
        #                                    set( pcv13_ppv23_nonpcv7))
        
        self.serotype_groups1 = { "pcv7": self.serotype_groups1["pcv7"],
                                 "pcv13": self.serotype_groups1["pcv13"],
                                 "ppv23": ppv23_nonpcv13_nonpcv7,
                                "nonppv23": others}
        
        
        
        self.serotype_groups = {}
        for keys,values in self.serotype_groups1.items():
            for i in values:
                self.serotype_groups[i] = keys
        self.transmission_coefficient_multipliers = {}
        for keys,values in self.serotype_groups.items():
            self.transmission_coefficient_multipliers[keys] = \
                    self.transmission_coefficient_multipliers1[values]
        
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
        self.vaccine_antibody_df.sort("exposed_strains","vaccine_type", "no_of_doses")
        
        #############
        #disease outcome multipliers
        self.disease_outcome_multipliers = p["disease_outcome_multipliers"]
        
        
        self.v116_only_strains = pl.Series("v116_only", 
                 ['15C', '15A', '24F', '31', '35B', '23A', '23B', '16F'])
        
        self.vt_strains =  pl.Series("vt_strains", 
                 ['6B', '11A', '31', '12F', '10A', '3', '19F', '6A', '15A',
                  '24F', '33F', '35B', '8', '1', '5', '7F', '15B', '20', 
                  '23B', '2', '19A', '23F', '9V', '9N', '15C', '18C', 
                  '22F', '14', '23A', '16F', '4', '17F']
                 )
        self.nvt_strains = pl.Series("nvt_strains", 
                 list(set(self.strains.to_list()) - 
                      set(self.vt_strains.to_list())))
        
        no_strains = len(self.v116_only_strains.to_list() + 
                         self.nvt_strains.to_list() + self.ppv23_nonpcv13)
        no_ages =  102
        ages_ppv23 = list(range(5,102))
        ages_nvt = list(range(0,102))
        ages_v116 = list(range(0,102))
        no_ages_ppv23 = len(ages_ppv23)
        no_ages_nvt = len(ages_nvt)
        no_ages_v116 = len(ages_v116)
        given_disease_outcome_multipliers = pl.DataFrame([ 
            pl.Series("age", 
                       ages_v116 *  len(self.v116_only_strains.to_list()) + 
                       ages_nvt *  len(self.nvt_strains.to_list()) + 
                       ages_ppv23 *  len(self.ppv23_nonpcv13)),
            pl.Series("strains", 
                      [ ele for ele in 
                      self.v116_only_strains.to_list() for i in ages_v116] +
                     [ ele for ele in 
                    (self.nvt_strains.to_list()) for i in ages_nvt] +
                     [ ele for ele in 
                    (self.ppv23_nonpcv13) for i in ages_ppv23] 
                      ),
                     # p["strains_with_higher_disease_outcome"] *\
                     # sum(no_ages)),
            pl.Series("multiplier", 
                      (([self.disease_outcome_multipliers["v116_only"]/2] * 50 +
                       [self.disease_outcome_multipliers["v116_only"]] * (no_ages_v116 - 50))
                       * len(self.v116_only_strains.to_list()) +
                     [self.disease_outcome_multipliers["nvt"]] * no_ages_nvt * len(self.nvt_strains.to_list()) +
                     [self.disease_outcome_multipliers["ppv23_nonpcv13"]] * no_ages_ppv23 * len(self.ppv23_nonpcv13))
                      )
            
            ]).sort("age", "strains")
        
        """
        for nvt in self.nvt_strains:
            multiplier = given_disease_outcome_multipliers.filter(pl.col("strains")== nvt)['multiplier'].unique() 
            
            if (len(multiplier) != 1 ) or (multiplier[0] != self.disease_outcome_multipliers["nvt"]):
                print(multiplier)
        
        for v116 in self.v116_only_strains:
            multiplier = given_disease_outcome_multipliers.filter(pl.col("strains")== v116)['multiplier'].unique() 
            
            if (len(multiplier) != 1 ) or (multiplier[0] != self.disease_outcome_multipliers["v116_only"]):
                print(multiplier)
        """
        self.disease_outcome_multipliers = pl.DataFrame([
            pl.Series("age", list(range(0,102)) * len(self.strains)),
            pl.Series("strains", self.strains.to_list() * 102),
            pl.Series("multiplier", [1] * len(self.strains) * 102 )
            ])
        self.disease_outcome_multipliers = \
            (self.disease_outcome_multipliers.update(
            given_disease_outcome_multipliers, on = ["age", "strains"], 
            how= "left"))
            
        ########################at_risk data loading#####################
        
        at_risk_perc = pl.read_csv(os.path.join(p['resource_prefix'],
                                 "disease/at_risk_percentages.csv"),
                                        truncate_ragged_lines=True)
        at_risk_perc = at_risk_perc.with_columns(
            (pl.col("at_risk_tier_1")/100),
            ((pl.col("at_risk_tier_1") + pl.col("at_risk_tier_2"))/100)
            .alias("at_risk_tier_2"),)
        
        at_risk_perc = at_risk_perc.filter(
            pl.col("Ind_status") == "nonindigenous").drop("Ind_status")
        
            
        at_risk_perc = at_risk_perc.with_columns(
                pl.Series("ages", 
                      p["at_risk_age_groups"]),
                pl.Series("tier_1_multipliers", 
                          p["prob_dis_at_risk_tier1_multipliers"]),
                pl.Series("tier_2_multipliers", 
                          p["prob_dis_at_risk_tier2_multipliers"]),
                pl.Series("no_risk_multipliers", 
                          p["prob_dis_at_risk_no_risk_multipliers"]),
                )
       
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
              pl.Series("tier_1_multipliers",
                        p["prob_dis_at_risk_tier1_multipliers"] + [0]),
              pl.Series("tier_2_multipliers",
                        p["prob_dis_at_risk_tier2_multipliers"] + [0]),
              pl.Series("no_risk_multipliers",
                        p["prob_dis_at_risk_no_risk_multipliers"] + [0]),
              
             ]) 
        
        self.at_risk_percentages = pl.DataFrame([
            pl.Series("age", list(range(0,102))),
             pl.Series("at_risk_age", age_group_list),
             ])  
        self.at_risk_percentages = (self.at_risk_percentages.join(
            at_risk_perc, left_on= "at_risk_age", right_on= "age_group",
            how = "left"
            ))
        self.at_risk_multipliers = pl.DataFrame([
            pl.Series("age", list(range(0,102))* 3),
            pl.Series("at_risk", 
                      [0] * 102 + [1] * 102 + [2] * 102 ).cast(pl.Int32),
            pl.Series("at_risk_multipliers",
                self.at_risk_percentages["no_risk_multipliers"].to_list() +
                self.at_risk_percentages["tier_1_multipliers"].to_list() +
                self.at_risk_percentages["tier_2_multipliers"].to_list() ) 
            ]).fill_null(1)
        
        self.at_risk_percentages = self.at_risk_percentages.drop(
                                "tier_1_multipliers", "tier_2_multipliers",
                                     "no_risk_multipliers")
        self.at_risk_vaccine_target_groups = p["at_risk_vaccine_target_group"]
        
        self.prev_vacc_list = pl.DataFrame(
            {"vaccine": (["atrisk_ppv23_adult"] * 96 +
             ["atrisk_ppv23_child"] * 96 +
             ["atrisk_ppv23_adult_2"] * 96 + 
             ["atrisk_ppv23_child_2"] * 96
             ) ,
             "age": list(range(5,101)) * 4,
             "prev_eff_vaccine": \
            ["atrisk_pcv13_child"] * 14  + ["atrisk_pcv13_adult"] * 82 +
              ["atrisk_pcv13_child"] * 96 +
              ["atrisk_pcv13_child"] * 19 + ["atrisk_pcv13_adult"] * 77 +
              ["atrisk_pcv13_child"] * 96,
             "substract_vacc_time":  ([365] * 96 +
              [365] * 96 +
              [365 * 6] * 96 + 
              [365 * 6] * 96) ,})
        
        if "atrisk_ppv23_adult" in self.vaccines:
            
            self.prev_vacc_list = pl.DataFrame(
                {
                 "vaccine": (["atrisk_ppv23_adult"] * 83 +
                             ["atrisk_ppv23_adult_2"] * 83 ),
                 "age": list(range(18,101)) * 2,
                 
                 "prev_eff_vaccine": \
                ([self.vaccines["atrisk_ppv23_adult"]["previous_vacc"]] * 83 +
                [self.vaccines["atrisk_ppv23_adult"]["previous_vacc"]] * 83),
                
                 "substract_vacc_time":  ([365] * 83 +
                                          [365 * 6] * 83)
                 })    
                
                
    
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
            if (vaccine.endswith(("_entrybirthcohort"))):
                vaccine = vaccine.replace("_entrybirthcohort", "")
                #.strip("_entrybirthcohort")
                
            if (vaccine in ["V116_adult", "atrisk_V116_adult"]) & \
                (value["years"][0] <= cur_year <= value["years"][1]):
                P = roll_v116(P, vaccine, value, t, day, 
                              t_per_year, self.vaccines, 
                              self.at_risk_vaccine_target_groups, rng)
                    
            elif value["years"][0] <= cur_year <= value["years"][1]:
                vacc_rollout_year = cur_year - value["years"][0]
                on_time_coverage = value["on_time_coverage_frac"][\
                                vacc_rollout_year]
                late_coverage = value["late_coverage_frac"][\
                                vacc_rollout_year]
                #time to vaccinate individuals 
                if (vaccine in ["atrisk_ppv23_adult", "atrisk_ppv23_child",
                                "atrisk_ppv23_adult_2", "atrisk_ppv23_child_2"]):
                    
                    if (vaccine in ["atrisk_ppv23_adult", 
                                    "atrisk_ppv23_child"]):
                        vacc_prefix = (self.vaccines["atrisk_ppv23_adult"]
                                       ["previous_vacc"].replace("_adult", "")
                                       .replace("_child", ""))
                        #find pcv13 vaccinated atrisk indvs
                        prev_vaccines = [f"{vacc_prefix}_adult",
                                         f"{vacc_prefix}_child"]
                           
                    else:
                        #vaccine = vaccine.strip("_1").strip("_2")
                        
                        prev_vaccines = ["atrisk_ppv23_adult", 
                                         "atrisk_ppv23_child"]
                        
                        
                    vacc_target_group = (
                        P.I.filter((pl.col("age")
                            .is_between(value["vaccination_age_range"][0],
                                value["vaccination_age_range"][1])) & 
                                (pl.col("at_risk").is_in(
                        self.at_risk_vaccine_target_groups)) & 
                      #(pl.col("vaccines").struct.field("no_of_doses") == 1) & \
                            (pl.col("vaccines").struct.field("vaccine_type")
                             .is_in(prev_vaccines)) &\
                            ((day - 
              pl.col("vaccines").struct.field("final_vaccine_time")) == (364 +
              364 * 4 * (vaccine in ["atrisk_ppv23_adult_2",
                                     "atrisk_ppv23_child_2"])))
                                         )
                           .select(["id","age", "age_days", 
                                    "vaccines", "random"])
                    )
                    """if (vacc_target_group.height > 0):
                        print("here")
                        if (vaccine in ["atrisk_ppv23_adult_2", 
                                        "atrisk_ppv23_child_2"]):
                            print("here")"""
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
                                        
                    cont_vacc_target_group = (vacc_target_group.filter(
                       (pl.col("age_days") != value["daily_schedule"][0]) |
                           (pl.col("random") > (on_time_coverage + \
                                               (late_coverage)))
                         ))
                    on_time_first_vacc = (on_time_first_vacc.with_columns(
                            (pl.lit(rng.rand(on_time_first_vacc.height))
                                 .alias("random")),
                        pl.struct(
                    pl.lit(1).alias("no_of_doses"),
                    #(pl.col("vaccines").struct.field("no_of_doses") + 1),
            (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
                            (pl.lit(vaccine)).alias("vaccine_type"),
                            pl.lit(day).alias("final_vaccine_time"),
                            ).alias("vaccines")
                            ))
                    
                        
                elif (vaccine.endswith(("_1", "_2", "_adult"))) & \
                        (not vaccine.startswith(("atrisk"))):
                    vaccine = vaccine.strip("_1").strip("_2")
                    vacc_age_group = (
                        P.I.filter((pl.col("age")
                            .is_between(value["vaccination_age_range"][0],
                                        value["vaccination_age_range"][1])
                            ) &
                        #((pl.col("age") > 69) | (pl.col("at_risk")
                        # .is_in(self.at_risk_vaccine_target_groups).is_not()))
                        (pl.col("vaccines").struct.field("vaccine_type").str
                                 .starts_with(("atrisk")).is_not())
                        )
                           .select(["id","age", "age_days", 
                                    "vaccines", "random"])
                    )
                    vacc_target_group = (
                        vacc_age_group.filter(
                            ((pl.col("vaccines").struct.field("no_of_doses")
                                 < len(value["daily_schedule"])) |
                            (pl.col("vaccines").struct.field("vaccine_type")
                                 != vaccine))
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
                    pl.lit(1).alias("no_of_doses"),
                    #(pl.col("vaccines").struct.field("no_of_doses") + 1),
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
                elif vaccine.startswith("atrisk"):
                    """if (vaccine in ["atrisk_ppv23_adult_2", 
                                    "atrisk_ppv23_child_2"]):
                        print("here")"""
                    vacc_age_group = (
                        P.I.filter((pl.col("age")
                            .is_between(value["vaccination_age_range"][0],
                                        value["vaccination_age_range"][1])
                            
                            ) &
                            (pl.col("at_risk")
                             .is_in(self.at_risk_vaccine_target_groups)))
                           .select(["id","age", "age_days", 
                                    "vaccines", "random"])
                    )
                    #make sure they don't receive the vaccine beforehand
                    vacc_target_group = (
                        vacc_age_group.filter(
                        ((pl.col("vaccines")
                         .struct.field("vaccine_type") != vaccine)) &
                        ((pl.col("vaccines")
                         .struct.field("vaccine_type")).str
                         .starts_with("atrisk").is_not())
                                     ))
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
                    pl.lit(1).alias("no_of_doses"),
                    #(pl.col("vaccines").struct.field("no_of_doses") + 1),
            (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
                            (pl.lit(vaccine)).alias("vaccine_type"),
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
        #self.transmission_coefficient_multipliers
        i = 0 
        for serotype_group, multiplier in \
            self.transmission_coefficient_multipliers1.items():
                cur_foi = [self.transmission_coef * multiplier * \
                    sum([x * y for x, y in \
                      zip(self.foi["inf_fraction_%s"%serotype_group],row)])\
                        for row in self.cmatrix.C]
                if i == 0: 
                    foi = cur_foi
                else:
                    foi = [sum(x) for x in zip(foi, cur_foi)]
                    
                i += 1
        
        #foi =  [self.transmission_coef *\
        #        sum([x * y for x, y in zip(self.foi["inf_fraction"],row)])\
        #        for row in self.cmatrix.C]
            
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
                      
         #varying transmission             
         strain_counts = ((P.I.select("age_group",
            pl.col("strain_list").alias("strain"))
             .explode("strain")
            ).groupby(["age_group", 'strain']).agg(pl.count())
             .filter(pl.col("strain") != "null")).with_columns(
                (pl.col("strain").map_dict(self.serotype_groups)
                 ).alias("serotype_group")
               ).group_by("age_group", "serotype_group").agg(pl.sum("count"))
         
         
         strain_counts = strain_counts.pivot(values='count', 
                index='age_group', columns='serotype_group').sort("age_group")
         
         if len(strain_counts.columns) < (1 + 
                            len(self.transmission_coefficient_multipliers1)):
             cols = list(self.transmission_coefficient_multipliers1.keys()) 
             missing_cols = list(set(cols) - set(strain_counts.columns))
             strain_counts = strain_counts.with_columns(
                 [(pl.lit(0)).alias(missing_col) \
                  for missing_col in missing_cols]
                 )
                 
         
         self.foi = self.foi.join(
             strain_counts, on= "age_group", how = "left").with_columns(
                 (pl.col("pcv7")/ (pl.col("total_inds") *\
                     self.max_no_coinfections)).alias("inf_fraction_pcv7"),
                (pl.col("pcv13")/ (pl.col("total_inds") *\
                    self.max_no_coinfections)).alias("inf_fraction_pcv13"),
                (pl.col("ppv23")/ (pl.col("total_inds") *\
                    self.max_no_coinfections)).alias("inf_fraction_ppv23"),
                    
                    
                (pl.col("nonppv23")/ (pl.col("total_inds") *\
                    self.max_no_coinfections)).alias("inf_fraction_nonppv23"),
                 ).sort("age_group").fill_null(strategy="zero")
                  
         missing_ages = [ i for i in list(range(16)) if i not in 
                         self.foi["age_group"]]
         
         if len(missing_ages):
             missing_foi_df = pl.DataFrame({
                 "age_group" : missing_ages,
                 "no_of_strains": [0]* len(missing_ages),
                 "total_inds": [0]* len(missing_ages),
                 "inf_fraction": [0]* len(missing_ages),
                 "pcv13": [0]* len(missing_ages),
                 "ppv23": [0]* len(missing_ages),
                 "nonppv23": [0]* len(missing_ages),
                 "pcv7": [0]* len(missing_ages),
                 "inf_fraction_pcv7": [0]* len(missing_ages),
                 "inf_fraction_pcv13": [0]* len(missing_ages),
                 "inf_fraction_nonpcv13": [0]* len(missing_ages),
    
                 }).with_columns(pl.col("age_group").cast(pl.Int64),
                            pl.col("no_of_strains").cast(pl.Int32),
                            pl.col("total_inds").cast(pl.UInt32),
                            pl.col("inf_fraction").cast(pl.Float64),
                            pl.col("nonpcv13").cast(pl.UInt32),
                            pl.col("pcv7").cast(pl.UInt32),
                            pl.col("pcv13").cast(pl.UInt32),
                            pl.col("inf_fraction_pcv7").cast(pl.Float64),
                            pl.col("inf_fraction_pcv13").cast(pl.Float64),
                            pl.col("inf_fraction_nonppv23").cast(pl.Float64),
                            pl.col("inf_fraction_nonppv23").cast(pl.Float64),
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
                ).groupby(['strain']).agg(pl.count())
                 .filter(pl.col("strain") != "null")).with_columns(
                  pl.col("strain").map_dict(
                      self.transmission_coefficient_multipliers)
                      .alias("multipliers") 
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
        
        #recovered.filter(pl.col("strain_list").list.len() != pl.col("endTimes").list.len())
       
        
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
            
            #vacc with ppv23
            will_infected_ppv23 = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                        ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"]))
            #TODO: update here for at_risk
            
            
                
            possibly_infected_atrisk_ppv23 = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                            ["atrisk_ppv23_adult",
                             "atrisk_ppv23_child",
                             "atrisk_ppv23_adult_2","atrisk_ppv23_child_2"]))
            """if (len(possibly_infected_atrisk_ppv23["vaccines"]
                    .struct.field("vaccine_type").unique()) > 0):
                print("here")"""
            possibly_infected_atrisk_ppv23 = (possibly_infected_atrisk_ppv23.
                                    join(self.prev_vacc_list,
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                "age"],right_on = ["vaccine","age"], how= "left"))
            
            possibly_infected_atrisk_ppv23 = (possibly_infected_atrisk_ppv23
                                .join(self.vaccine_antibody_df, 
                left_on = ["prev_eff_vaccine",
                pl.col("vaccines").struct.field("no_of_doses"),
                pl.col("exposed_strains")],
                right_on = ["vaccine_type","no_of_doses",
                            "exposed_strains"],
                how="left").drop("vaccine_type","no_of_doses") )
            
            log_antibodies_atrisk_ppv23 = pl.DataFrame([
               pl.Series("meanlog" , possibly_infected_atrisk_ppv23["meanlog"]),
               pl.Series("log_antibodies" ,lognorm.ppf(
                   possibly_infected_atrisk_ppv23["quantile"], 
                            possibly_infected_atrisk_ppv23["sdlog"], 
                       loc=possibly_infected_atrisk_ppv23["meanlog"],
                       scale=1,
                       )),
               pl.Series("antibodies" ,
                    lognorm.ppf(possibly_infected_atrisk_ppv23["quantile"], 
                        np.sqrt(possibly_infected_atrisk_ppv23["sdlog"]), 
                       loc=0,
                       scale=np.exp(possibly_infected_atrisk_ppv23["meanlog"]),
                       )),
               pl.Series("waning_ratio" ,
               waning_ratio(
               possibly_infected_atrisk_ppv23["prev_eff_vaccine"], 
               day, 
         (possibly_infected_atrisk_ppv23["vaccines"].struct.field("final_vaccine_time") -
              possibly_infected_atrisk_ppv23["substract_vacc_time"]),
             self.waning_halflife_day_adult, self.waning_halflife_day_child,
             possibly_infected_atrisk_ppv23["age"]))
               ])
            #if log_antibodies_atrisk_ppv23.height:
            #    print("here")
            log_antibodies_atrisk_ppv23 = log_antibodies_atrisk_ppv23.with_columns(
               pl.when(pl.col("meanlog") > -9)
               .then( np.log((pl.col("antibodies") * 
                     pl.col("waning_ratio"))))
               .otherwise(pl.col("meanlog")).alias("waning_log_antibodies")
               )["waning_log_antibodies"]
           
           
            prob_of_transmission_atrisk_ppv23 = ( 1 / \
                           (1 + self.prob_acq_logantibody_scale * \
                           np.exp( self.prob_acq_logantibody_shape * \
            (log_antibodies_atrisk_ppv23 - self.prob_acq_logantibody_shift))))   
            
            possibly_infected_atrisk_ppv23 = (possibly_infected_atrisk_ppv23
                            .drop('prev_eff_vaccine', 'substract_vacc_time'))
            
            
            
            possibly_infected = possibly_infected.filter(
                        pl.col("vaccines").struct.field("vaccine_type").is_in(
                            ["atrisk_ppv23_adult_2","atrisk_ppv23_adult",
                             "atrisk_ppv23_child_2", "atrisk_ppv23_child",
                             "ppsv23_adult", "ppsv23_1", 
                             "ppsv23_2", "ppsv23"]).is_not())
            
            #if only vacc with ppv23, then they will be infected regardless
            will_infected = pl.concat(
                                [will_infected, will_infected_ppv23],\
                                      rechunk=True, how = 'diagonal')
            
            
            #if vacc with ppv23 after prev adult vacc doses, 
            #then they will be infected based on 
            #the antibody level from prev dose  
            """possibly_infected_atrisk_ppv23_adult_1 = \
                (possibly_infected_atrisk_ppv23_adult_1.join(
                    self.vaccine_antibody_df, 
                    left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                    pl.col("vaccines").struct.field("no_of_doses"),
                    pl.col("exposed_strains")],
                    right_on = ["vaccine_type","no_of_doses",
                                "exposed_strains"],
                    how="left").drop("vaccine_type","no_of_doses") )"""
            #possibly_infected.filter(pl.col("id").is_duplicated())

            
            
            
            possibly_infected = possibly_infected.join(
                self.vaccine_antibody_df, 
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                pl.col("vaccines").struct.field("no_of_doses"),
                pl.col("exposed_strains")],
                right_on = ["vaccine_type","no_of_doses",
                            "exposed_strains"],
                how="left").drop("vaccine_type","no_of_doses") 
            
            log_antibodies = pl.DataFrame([
                pl.Series("meanlog" , possibly_infected["meanlog"]),
                pl.Series("log_antibodies" ,lognorm.ppf(possibly_infected["quantile"], 
                                         possibly_infected["sdlog"], 
                        loc=possibly_infected["meanlog"],
                        scale=1,
                        )),
                pl.Series("antibodies" ,lognorm.ppf(possibly_infected["quantile"], 
                                         np.sqrt(possibly_infected["sdlog"]), 
                        loc=0,
                        scale=np.exp(possibly_infected["meanlog"]),
                        )),
                pl.Series("waning_ratio" ,
                waning_ratio(
                possibly_infected["vaccines"].struct.field("vaccine_type"), 
                day, 
              possibly_infected["vaccines"].struct.field("final_vaccine_time"),
              self.waning_halflife_day_adult, self.waning_halflife_day_child,
              possibly_infected["age"]))
                ])
            
            log_antibodies = log_antibodies.with_columns(
                pl.when(pl.col("meanlog") > -9)
                .then( np.log((pl.col("antibodies") * 
                      pl.col("waning_ratio"))))
                .otherwise(pl.col("meanlog")).alias("waning_log_antibodies")
                )["waning_log_antibodies"]
            
            
            prob_of_transmission = ( 1 / \
                            (1 + self.prob_acq_logantibody_scale * \
                            np.exp( self.prob_acq_logantibody_shape * \
                          (log_antibodies - self.prob_acq_logantibody_shift))))  
            
            P.vaccinated_acq_pop = (pl.concat([
                (possibly_infected.with_columns(
               prob_of_transmission.alias("rel_prob_transmission"),
               log_antibodies,
               (pl.col("random") <= prob_of_transmission)
               .alias("will_infected")
               ).filter(pl.col("meanlog")> -9)),
                
                (possibly_infected_atrisk_ppv23.with_columns(
               prob_of_transmission_atrisk_ppv23.alias("rel_prob_transmission"),
               log_antibodies_atrisk_ppv23,
               (pl.col("random") <= prob_of_transmission_atrisk_ppv23)
               .alias("will_infected")
               ).filter(pl.col("meanlog")> -9))
                ],rechunk=True, how = 'diagonal' )
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
            
            will_infected_vacc_atrisk_ppv23 = (possibly_infected_atrisk_ppv23.filter(
                ((pl.col("random") <= (prob_of_transmission_atrisk_ppv23)
                  ).alias("will_infected")),
                ))
            will_infected_vacc_atrisk_ppv23 = will_infected_vacc_atrisk_ppv23.with_columns(
                (pl.lit(rng.rand(will_infected_vacc_atrisk_ppv23.height))
                 .alias("random"))).drop("meanlog", "sdlog")
                    
            not_infected = (possibly_infected.filter(
                ~pl.col("id").is_in(will_infected_vacc["id"])))
             
            not_infected = not_infected.with_columns(
                            (pl.lit(rng.rand(not_infected.height))
                             .alias("random")))
            not_infected_atrisk_ppv23 = (possibly_infected_atrisk_ppv23.filter(
                ~pl.col("id").is_in(will_infected_vacc_atrisk_ppv23["id"])))
             
            not_infected_atrisk_ppv23 = not_infected_atrisk_ppv23.with_columns(
                            (pl.lit(rng.rand(not_infected_atrisk_ppv23.height))
                             .alias("random")))
            
            will_infected = pl.concat([will_infected, will_infected_vacc,
                                       will_infected_vacc_atrisk_ppv23],\
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
            
            updated_infections = pl.concat([will_infected, not_infected, 
                                            not_infected_atrisk_ppv23],\
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
        
        infected = infected.join(self.disease_outcome_multipliers,
                                 left_on = ["age","exposed_strains"],
                                 right_on = ['age', 'strains'])
        
        #group them based on vaccine received
        infected_novacc = infected.filter(
                    pl.col("vaccines").struct.field("no_of_doses") == 0)
        
        infected_vacc = infected.filter(
                    pl.col("vaccines").struct.field("no_of_doses") > 0)
        
        #disagg vacc individuals based on ppv23
        infected_vacc_ppv23 = infected_vacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"]))
        
        infected_vacc_ppv23_protected = (infected_vacc_ppv23.filter(
                 pl.col("random") <= (self.ipd_fraction_by_age_group[
                     infected_vacc_ppv23["age_coef"]])))
        infected_vacc_ppv23_protected = (infected_vacc_ppv23_protected
                     .with_columns(
                    (pl.lit(rng.rand(infected_vacc_ppv23_protected.height))
                    .alias("random")),))
        
        infected_vacc_ppv23_not_protected = (infected_vacc_ppv23.filter(
                 pl.col("random") > (self.ipd_fraction_by_age_group[
                     infected_vacc_ppv23["age_coef"]])))
        infected_vacc_ppv23_not_protected = (infected_vacc_ppv23_not_protected
                     .with_columns(
                (pl.lit(rng.rand(infected_vacc_ppv23_not_protected.height))
                    .alias("random")),))
        
        
        ####at-risk vacc separate as ipd and cap
        infected_vacc_atrisk_ppv23 = (infected_vacc.drop("meanlog", "sdlog")
                .join(self.vaccine_antibody_df, 
            left_on = [pl.col("vaccines").struct.field("vaccine_type"),
            pl.col("vaccines").struct.field("no_of_doses"),
            pl.col("exposed_strains")],
            right_on = ["vaccine_type","no_of_doses",
                        "exposed_strains"],
            how="left").drop("vaccine_type","no_of_doses").filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                        ["atrisk_ppv23_adult",
                         "atrisk_ppv23_child",
                     "atrisk_ppv23_adult_2","atrisk_ppv23_child_2"])).join(
                    self.prev_vacc_list,
                    left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                    "age"],right_on = ["vaccine","age"], how= "left").join(
                    self.vaccine_antibody_df, 
                        left_on = ["prev_eff_vaccine",
                        pl.col("vaccines").struct.field("no_of_doses"),
                        pl.col("exposed_strains")],
                        right_on = ["vaccine_type","no_of_doses",
                                    "exposed_strains"],
                        how="left",
                    suffix = "_prev_vacc")
                        .drop("vaccine_type","no_of_doses"))
        
                
        infected_vacc_atrisk_ppv23_ipd = (infected_vacc_atrisk_ppv23.filter(
                 pl.col("random") <= (self.ipd_fraction_by_age_group[
                     infected_vacc_atrisk_ppv23["age_coef"]])))
        infected_vacc_atrisk_ppv23_ipd = (infected_vacc_atrisk_ppv23_ipd
                     .with_columns(
                    (pl.lit(rng.rand(infected_vacc_atrisk_ppv23_ipd.height))
                    .alias("random")),)).join(self.prev_vacc_list,
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                        "age"],right_on = ["vaccine","age"], how= "left")
        infected_vacc_atrisk_ppv23_cap = (infected_vacc_atrisk_ppv23.filter(
                 pl.col("random") > (self.ipd_fraction_by_age_group[
                     infected_vacc_atrisk_ppv23["age_coef"]])))
        infected_vacc_atrisk_ppv23_cap = (infected_vacc_atrisk_ppv23_cap
                     .with_columns(
                (pl.lit(rng.rand(infected_vacc_atrisk_ppv23_cap.height))
                    .alias("random")),)).join(self.prev_vacc_list,
                 left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                        "age"],right_on = ["vaccine","age"], how= "left")
        #####  IPD  ######
        log_antibodies_atrisk_ppv23_ipd = pl.DataFrame([
            pl.Series("meanlog" , infected_vacc_atrisk_ppv23_ipd["meanlog"]),
            pl.Series("log_antibodies" ,lognorm.ppf(infected_vacc_atrisk_ppv23_ipd["quantile"], 
                                     infected_vacc_atrisk_ppv23_ipd["sdlog"], 
                    loc=infected_vacc_atrisk_ppv23_ipd["meanlog"],
                    scale=1,
                    )),
            pl.Series("antibodies" ,lognorm.ppf(infected_vacc_atrisk_ppv23_ipd["quantile"], 
                                     np.sqrt(infected_vacc_atrisk_ppv23_ipd["sdlog"]), 
                    loc=0,
                    scale=np.exp(infected_vacc_atrisk_ppv23_ipd["meanlog"]),
                    )),
                pl.Series("waning_ratio" ,
                waning_ratio(
                infected_vacc_atrisk_ppv23_ipd["vaccines"].struct.field("vaccine_type"), 
                day, 
              infected_vacc_atrisk_ppv23_ipd["vaccines"].struct.field("final_vaccine_time"),
              self.waning_halflife_day_adult, self.waning_halflife_day_child,
              infected_vacc_atrisk_ppv23_ipd["age"]
              )),
            pl.Series("meanlog_prev" , infected_vacc_atrisk_ppv23_ipd["meanlog_prev_vacc"]),
            pl.Series("log_antibodies_prev" ,lognorm.ppf(infected_vacc_atrisk_ppv23_ipd["quantile"], 
                                     infected_vacc_atrisk_ppv23_ipd["sdlog_prev_vacc"], 
                    loc=infected_vacc_atrisk_ppv23_ipd["meanlog_prev_vacc"],
                    scale=1,
                    )),
            pl.Series("antibodies_prev" ,lognorm.ppf(infected_vacc_atrisk_ppv23_ipd["quantile"], 
                                     np.sqrt(infected_vacc_atrisk_ppv23_ipd["sdlog_prev_vacc"]), 
                    loc=0,
                    scale=np.exp(infected_vacc_atrisk_ppv23_ipd["meanlog_prev_vacc"]),
                    )),
            pl.Series("waning_ratio_prev" ,
                waning_ratio(
                infected_vacc_atrisk_ppv23_ipd["prev_eff_vaccine"], 
                day, 
              (infected_vacc_atrisk_ppv23_ipd["vaccines"]
               .struct.field("final_vaccine_time") - 
              infected_vacc_atrisk_ppv23_ipd["substract_vacc_time"]),
              self.waning_halflife_day_adult, self.waning_halflife_day_child,
              infected_vacc_atrisk_ppv23_ipd["age"]
              )), ])
        
        log_antibodies_atrisk_ppv23_ipd = (log_antibodies_atrisk_ppv23_ipd
                                .with_columns(
            pl.when(pl.col("meanlog") > -9)
            .then( np.log((pl.col("antibodies") * 
                  pl.col("waning_ratio"))))
            .otherwise(pl.col("meanlog")).alias("waning_log_antibodies"),
            pl.when(pl.col("meanlog_prev") > -9)
            .then( np.log((pl.col("antibodies_prev") * 
                  pl.col("waning_ratio_prev"))))
            .otherwise(pl.col("meanlog_prev"))
            .alias("waning_log_antibodies_prev"),
            ).with_columns(
                pl.max_horizontal(['waning_log_antibodies',
                        'waning_log_antibodies_prev'])
                .alias('waning_log_antibodies')
                )["waning_log_antibodies"])
                
        prob_of_disease_atrisk_ppv23_ipd = ( 
           self.prob_dis_logantibody_scale[
               infected_vacc_atrisk_ppv23_ipd["age_coef"]] / ( 1 + \
                self.prob_dis_logantibody_age[
                    infected_vacc_atrisk_ppv23_ipd["age_coef"]] * \
                    np.exp( self.prob_dis_logantibody_shape * \
                   (log_antibodies_atrisk_ppv23_ipd - 
                    #self.prob_dis_logantibody_shift
                    -1.7 + 1250 *\
                   self.prob_dis_logantibody_scale[
                       infected_vacc_atrisk_ppv23_ipd["age_coef"]]
                   
                    ))))
            
        cur_disease_atrisk_ppv23_ipd = (infected_vacc_atrisk_ppv23_ipd
                .join(self.at_risk_multipliers, on= ["age", "at_risk"], 
                    how = "left").filter((pl.col("random") <= 
            (prob_of_disease_atrisk_ppv23_ipd * pl.col("at_risk_multipliers")
             * pl.col("multiplier")) 
              )))
        cur_disease_atrisk_ppv23_ipd = (cur_disease_atrisk_ppv23_ipd
                    .with_columns(pl.lit("ipd").alias("disease") ))
                                          
        #####  CAP  ######
                                 
        log_antibodies_atrisk_ppv23_cap = pl.DataFrame([
            #pl.Series("meanlog" , infected_vacc_atrisk_ppv23_cap["meanlog"]),
            #pl.Series("log_antibodies" ,lognorm.ppf(infected_vacc_atrisk_ppv23_cap["quantile"], 
            #                         infected_vacc_atrisk_ppv23_cap["sdlog"], 
            #        loc=infected_vacc_atrisk_ppv23_cap["meanlog"],
            #        scale=1,
            #        )),
            #    pl.Series("waning_ratio" ,
            #    waning_ratio(
            #    infected_vacc_atrisk_ppv23_cap["vaccines"].struct.field("vaccine_type"), 
            #    day, 
            #  infected_vacc_atrisk_ppv23_cap["vaccines"].struct.field("final_vaccine_time"),
            #  self.waning_halflife_day_adult, self.waning_halflife_day_child,
            #  infected_vacc_atrisk_ppv23_cap["age"]
            #  )),
            pl.Series("meanlog_prev" , infected_vacc_atrisk_ppv23_cap["meanlog_prev_vacc"]),
            pl.Series("log_antibodies_prev" ,lognorm.ppf(infected_vacc_atrisk_ppv23_cap["quantile"], 
                                     infected_vacc_atrisk_ppv23_cap["sdlog_prev_vacc"], 
                    loc=infected_vacc_atrisk_ppv23_cap["meanlog_prev_vacc"],
                    scale=1,
                    )),
            pl.Series("antibodies_prev" ,lognorm.ppf(infected_vacc_atrisk_ppv23_cap["quantile"], 
                                     np.sqrt(infected_vacc_atrisk_ppv23_cap["sdlog_prev_vacc"]), 
                    loc=0,
                    scale=np.exp(infected_vacc_atrisk_ppv23_cap["meanlog_prev_vacc"]),
                    )),
            pl.Series("waning_ratio_prev" ,
                waning_ratio(
                infected_vacc_atrisk_ppv23_cap["prev_eff_vaccine"], 
                day, 
              (infected_vacc_atrisk_ppv23_cap["vaccines"]
               .struct.field("final_vaccine_time") - 
              infected_vacc_atrisk_ppv23_cap["substract_vacc_time"]),
              self.waning_halflife_day_adult, self.waning_halflife_day_child,
              infected_vacc_atrisk_ppv23_cap["age"]
              )), ])
        #if log_antibodies_atrisk_ppv23_cap.height and day >= 8015:
        #    print("here")
        log_antibodies_atrisk_ppv23_cap = (log_antibodies_atrisk_ppv23_cap
                                .with_columns(
            #pl.when(pl.col("meanlog") > -9)
            #.then( (pl.col("log_antibodies") + 9) * \
            #      (pl.col("waning_ratio")) - 9)
            #.otherwise(pl.col("meanlog")).alias("waning_log_antibodies"),
            pl.when(pl.col("meanlog_prev") > -9)
            .then( np.log((pl.col("antibodies_prev") * 
                  pl.col("waning_ratio_prev"))))
            .otherwise(pl.col("meanlog_prev"))
            .alias("waning_log_antibodies_prev"),
            )#.drop("waning_log_antibodies")
            .rename({
                "waning_log_antibodies_prev": "waning_log_antibodies"
                })["waning_log_antibodies"])
            

            
        prob_of_disease_atrisk_ppv23_cap = ( 
           self.prob_dis_logantibody_scale[
               infected_vacc_atrisk_ppv23_cap["age_coef"]] / ( 1 + \
                self.prob_dis_logantibody_age[
                    infected_vacc_atrisk_ppv23_cap["age_coef"]] * \
                    np.exp( self.prob_dis_logantibody_shape * \
                   (log_antibodies_atrisk_ppv23_cap - 
                    #self.prob_dis_logantibody_shift
                    -1.7 + 1250 *\
                   self.prob_dis_logantibody_scale[
                       infected_vacc_atrisk_ppv23_cap["age_coef"]]
                   
                    ))))
        cur_disease_atrisk_ppv23_cap = (infected_vacc_atrisk_ppv23_cap
                .join(self.at_risk_multipliers, on= ["age", "at_risk"], 
                    how = "left").filter((pl.col("random") <= 
            (prob_of_disease_atrisk_ppv23_cap * pl.col("at_risk_multipliers")
             * pl.col("multiplier")) 
              )))
                                                        
        
        
        """cur_disease_atrisk_ppv23_cap = (cur_disease_atrisk_ppv23_cap.with_columns(
                            (pl.lit(rng.rand(cur_disease_atrisk_ppv23_cap.height))
                             .alias("random")),
                            ).with_columns(
                     pl.when(pl.col("random") <= (self.ipd_fraction_by_age_group[
                         cur_disease_atrisk_ppv23_cap["age_coef"]])
                              )
                     .then("ipd")
                     .otherwise("cap").alias("disease") 
                                 )).filter(pl.col("disease") == "cap")"""
        cur_disease_atrisk_ppv23_cap = (cur_disease_atrisk_ppv23_cap.with_columns(
                            (pl.lit(rng.rand(cur_disease_atrisk_ppv23_cap.height))
                             .alias("random")),
                            ).with_columns(
                     pl.lit("cap").alias("disease") 
                                 ))
        
        
        
        
        
        ######### rest of the vacc individuals
        infected_vacc = infected_vacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    ["ppsv23_adult", "ppsv23_1", 
                     "ppsv23_2", "ppsv23",
                     "atrisk_ppv23_adult","atrisk_ppv23_child",
                     "atrisk_ppv23_adult_2","atrisk_ppv23_child_2"]).is_not())
        
        
        
        
        
        infected_vacc = pl.concat(
                            [infected_vacc, infected_vacc_ppv23_protected],\
                                  rechunk=True, how = 'diagonal')
        infected_novacc = pl.concat(
                      [infected_novacc, infected_vacc_ppv23_not_protected],\
                                  rechunk=True, how = 'diagonal')
        
            
        """if (infected.filter(
                    pl.col("vaccines").struct.field("vaccine_type") == 
                    "ppsv23_adult").height > 0):
            print("here")
            print(infected.join(self.vaccine_antibody_df, 
                left_on = [pl.col("vaccines").struct.field("vaccine_type"),
                pl.col("vaccines").struct.field("no_of_doses"),
                pl.col("exposed_strains")],
                right_on = ["vaccine_type","no_of_doses", 
                            "exposed_strains"],
                how="left").drop("vaccine_type","no_of_doses").filter(
                        (pl.col("vaccines").struct.field("vaccine_type") == 
                        "ppsv23_adult") & (pl.col("exposed_strains").
                is_in(self.vaccines["ppsv23_adult"]["serotypes"]))))"""
        
        
        log_antibodies_vacc = pl.DataFrame([
            pl.Series("meanlog" , infected_vacc["meanlog"]),
            pl.Series("log_antibodies" ,lognorm.ppf(infected_vacc["quantile"], 
                                     infected_vacc["sdlog"], 
                    loc=infected_vacc["meanlog"],
                    scale=1,
                    )),
            pl.Series("antibodies" ,lognorm.ppf(infected_vacc["quantile"], 
                                     np.sqrt(infected_vacc["sdlog"]), 
                    loc=0,
                    scale=np.exp(infected_vacc["meanlog"]),
                    )),
                pl.Series("waning_ratio" ,
                waning_ratio(
                infected_vacc["vaccines"].struct.field("vaccine_type"), 
                day, 
              infected_vacc["vaccines"].struct.field("final_vaccine_time"),
              self.waning_halflife_day_adult, self.waning_halflife_day_child,
              infected_vacc["age"]
              ))
                ])
        
        log_antibodies_vacc = log_antibodies_vacc.with_columns(
            pl.when(pl.col("meanlog") > -9)
            .then( np.log((pl.col("antibodies") * 
                  pl.col("waning_ratio"))))
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
        
        
        
        
           
        cur_disease_pop_vacc = (infected_vacc.join(self.at_risk_multipliers, 
                             on= ["age", "at_risk"], how = "left").filter(
            (pl.col("random") <= 
             (prob_of_disease_vacc * pl.col("at_risk_multipliers") *
              pl.col("multiplier")) 
              )))
        #P.vaccinated_disease_po
        P.vaccinated_disease_pop = ((infected_vacc
                .join(self.at_risk_multipliers, on= ["age", "at_risk"],
                    how = "left")).with_columns(
            prob_of_disease_vacc,
            log_antibodies_vacc,
            (pl.col("random") <= (prob_of_disease_vacc *
                                  pl.col("at_risk_multipliers") *
                                  pl.col("multiplier"))
              ).alias("will_develop_disease")
            )
        .filter(
            (pl.col("meanlog")> -9))
        ).unnest("vaccines").select("id", "age","age_coef","quantile",
                      "final_vaccine_time", "vaccine_type", "no_of_doses",
                      "prob_dis_logantibody_scale", "meanlog", "sdlog",
                      "waning_log_antibodies", "exposed_strains",
                      "will_develop_disease", "at_risk")
        
        
        cur_disease_pop_novacc = infected_novacc.join(self.at_risk_multipliers, 
                             on= ["age", "at_risk"], how = "left").filter(
            (pl.col("random") <= 
             (prob_of_disease_novacc * pl.col("at_risk_multipliers"))
              ))
        
        infected_novacc_ppv23 = infected_novacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"]))
            
        
        cur_disease_vacc_not_ppv23 = (cur_disease_pop_vacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"]).is_not()))
        cur_disease_vacc_ppv23 = (cur_disease_pop_vacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"])))
        
        cur_disease_novacc_not_ppv23 = (cur_disease_pop_novacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"]).is_not()))
        cur_disease_novacc_ppv23 = (cur_disease_pop_novacc.filter(
                    pl.col("vaccines").struct.field("vaccine_type").is_in(
                    ["ppsv23_adult", "ppsv23_1", "ppsv23_2", "ppsv23"])))
        
        cur_disease_ipd_or_cap = pl.concat([cur_disease_vacc_not_ppv23,
                                     cur_disease_novacc_not_ppv23],
                                    how = 'diagonal')
        
        cur_disease_ipd_or_cap = (cur_disease_ipd_or_cap.with_columns(
                        (pl.lit(rng.rand(cur_disease_ipd_or_cap.height))
                         .alias("random")),
                        ).with_columns(
                 pl.when(pl.col("random") <= (self.ipd_fraction_by_age_group[
                     cur_disease_ipd_or_cap["age_coef"]])
                          )
                 .then("ipd")
                 .otherwise("cap").alias("disease") 
                             ))
                            
        cur_disease_just_ipd = (cur_disease_vacc_ppv23.with_columns(
                            pl.lit("ipd").alias("disease") ))
        
        cur_disease_just_cap = (cur_disease_novacc_ppv23.with_columns(
                            (pl.lit(rng.rand(cur_disease_novacc_ppv23.height))
                             .alias("random")),
                            ).with_columns(
                     #pl.when(pl.col("random") <= (self.ipd_fraction_by_age_group[
                     #    cur_disease_novacc_ppv23["age_coef"]])
                     #         )
                     #.then("ipd")
                     #.otherwise("cap").alias("disease") 
                     #            )).filter(pl.col("disease") == "cap")
                     pl.lit("cap").alias("disease")
                     ))
        
        cur_disease_pop = pl.concat([cur_disease_ipd_or_cap,
                                     cur_disease_just_ipd,
                                     cur_disease_just_cap,
                                     cur_disease_atrisk_ppv23_ipd,
                                     cur_disease_atrisk_ppv23_cap],
                                    how = 'diagonal')    
        
        
        
        """cur_disease_pop = (cur_disease_pop.with_columns(
                        (pl.lit(rng.rand(cur_disease_pop.height))
                         .alias("random")),
                        ).with_columns(
                 pl.when(pl.col("random") <= (self.ipd_fraction_by_age_group[
                     cur_disease_pop["age_coef"]])
                          )
                 .then("ipd")
                 .otherwise("cap").alias("disease") 
                             ))"""
        
        P.vaccinated_disease_pop = (P.vaccinated_disease_pop
                                .join(cur_disease_pop.select("id", "disease"),
                                      on="id", how= "left"))
        
        if day % (364) == 0:
            #set the counter zero
            P.disease_pop = cur_disease_pop
            
        else:
            P.disease_pop = pl.concat([P.disease_pop, cur_disease_pop],
                                      how="diagonal")
        infected =( infected.with_columns(
                        (pl.lit(rng.rand(infected.height))
                         .alias("random")))
                            .drop("age_coef", "meanlog","sdlog", "multiplier"))
        
        
        if (infected.height != initial_len) or \
            (initial_no_cols != len(infected.columns)) or \
            (infected.drop("random")
             .frame_equal(initial_inf.drop("random")) == False):
            print("not equal!")
        return infected
        
    

        
        
        
    