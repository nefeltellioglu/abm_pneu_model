"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class DiseaseObserver(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.interval = interval
        self.sumdata = pl.DataFrame({})
        self.observed_day = self.interval - 1
        self.vaccines = vaccines
        self.no_age_groups = 10
        
        #pl.Series("strain_list", list(np.sort(strains))))
        desc = {'t': tb.UInt32Col(pos=0),
                'cap_per_100k': tb.UInt32Col(pos=1),
                'ipd_per_100k': tb.UInt32Col(pos=2),
                'no_strains_in_ipd': tb.UInt32Col(pos=3),
                "no_PCV7_strains_in_ipd": tb.UInt32Col(pos=4),
                "no_PCV13_strains_in_ipd": tb.UInt32Col(pos=5),
                "no_PPSV23_strains_in_ipd": tb.UInt32Col(pos=6),
                "no_PPSV23_notPCV13_strains_in_ipd": tb.UInt32Col(pos=7),
            "fraction_prevalence_PCV7_strains_in_ipd": tb.Float64Col(pos=8),
            "fraction_prevalence_PCV13_strains_in_ipd": tb.Float64Col(pos=9),
            "fraction_prevalence_PPSV23_strains_in_ipd": tb.Float64Col(pos=10),
   "fraction_prevalence_PPSV23_notPCV13_strains_in_ipd": tb.Float64Col(pos=11),
        "no_of_cases_per_age_group": tb.UInt32Col(pos=12,
                                                  shape=(self.no_age_groups)), 
        "cases_per_100k_per_age_group": tb.Float64Col(pos=13,
                                                    shape=(self.no_age_groups))
        }
       
        self.age_0_conversion = pl.DataFrame([
           pl.Series("age", [0] * 364),
           pl.Series("age_days", list(range(364))),
           pl.Series("age_coef", [0] * 30 + [1] * 60 + [2] * 60 + \
                               [3] * (180 + 34)), #+ [-1] * 34 ),
           ]) 
        
        self.age1_conversion = pl.DataFrame([
            pl.Series("age", list(range(1,102))),
             pl.Series("age_coef", [4] * 4 + [5] * 10 + [6] * 10 + \
                                   [7] * 25 + [8] * 15 + [9] * 37 
                       ), #+ [-1] * 34 ),
             ])
            
        ppsv23 = self.vaccines["ppsv23_1"]["serotypes"]
        indexes = (~self.vaccines["ppsv23_1"]["serotypes"].is_in(self.vaccines["pcv13_30"]["serotypes"])).to_list()
        indexes =[i for i, x in enumerate(indexes) if x]
        
        self.ppsv23_not_pcv13_strains = ppsv23[indexes]
        super(DiseaseObserver, self).__init__(h5file =h5file, \
                            label = 'disease', description = desc,
                            title = 'DiseaseObserver')
        
    def update_age_conversion(self):
        self.age_0_conversion = pl.DataFrame([
           pl.Series("age", [0] * 360),
           pl.Series("age_days", list(range(360))),
           pl.Series("age_coef_by_age", [0] * 60 + [1] * 60 + [2] * 60 + \
                               [3] * (180)), #+ [-1] * 34 ),
           ]) 
    
    def update(self, t, pop, **kwargs):
        
        if t % self.interval == self.observed_day:
            
            year = t / self.interval
            #agg data
            disease_counts = pop.disease_pop.group_by(
                                        "disease").agg(pl.count())
            disease_age_groups = pl.DataFrame([
                pl.Series("age_coef", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                ]).update(
                    pop.disease_pop.group_by("age_coef").agg(pl.count()),
                    on = "age_coef", how= "left" ).sort("age_coef")
            if pop.disease_pop.height and \
                pop.disease_pop["age_coef"].max() > 17:
                print("obs_disese: wrong age categories")
            
            #if pop.disease_pop.height:
            #    print("here")
            #create age groups for the pop   
            
            
            cur_pop = pop.I.join(self.age_0_conversion,
                                 on= ["age", "age_days"],how="left")
            cur_pop = cur_pop.update(self.age1_conversion,
                                 on= ["age"],how="left")
            
            cur_pop = cur_pop.filter(
                (pl.col("age_coef") >= 0) |
                ((pl.col("age") > 0)) |
                ((pl.col("age") == 0) &
                 (pl.col("age_days") <= 47)))
            
            
            
            #pop.I.filter(pl.col("age_coef") == 16)
            popsize_age_groups = pl.DataFrame([
                pl.Series("age_coef", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                ]).update(
                    cur_pop.group_by("age_coef").agg(pl.count()),
                    on = "age_coef", how= "left" ).sort("age_coef")
            
            frac_cases_age_groups = ((disease_age_groups["count"] / 
                     popsize_age_groups["count"]) * 100_000).round(2).to_list()
            
            ipd_cases = pop.disease_pop.filter(pl.col("disease") == "ipd")
            #sum(ipd_cases["exposed_strains"]
            #    .is_in(self.vaccines["pcv7"]["serotypes"]))
            #if len(target_group):
            #    for row in target_group.rows(named=True):
            self.row['t'] = year
            self.row['cap_per_100k'] = (disease_counts.filter(
                                        pl.col("disease") == "cap"
                                        )["count"][0] /
                                cur_pop.height) * 100_000
            self.row['ipd_per_100k'] = 0.0 if \
             (disease_counts.filter(pl.col("disease") == "ipd").height == 0) \
                 else (disease_counts.filter(
                              pl.col("disease") == "ipd")["count"][0] /
                                cur_pop.height) * 100_000
            self.row['no_strains_in_ipd'] = \
                                    ipd_cases["exposed_strains"].n_unique()
            self.row['no_PCV7_strains_in_ipd'] = \
                                sum(ipd_cases["exposed_strains"].unique()
                                .is_in(self.vaccines["pcv7"]["serotypes"]))
            self.row['no_PCV13_strains_in_ipd'] = \
                                    sum(ipd_cases["exposed_strains"].unique()
                                .is_in(self.vaccines["pcv13_30"]["serotypes"]))
            self.row['fraction_prevalence_PCV7_strains_in_ipd'] = \
                        sum(ipd_cases["exposed_strains"]
                    .is_in(self.vaccines["pcv7"]["serotypes"])) / \
                          ipd_cases.height  if ipd_cases.height  else 0
            self.row['fraction_prevalence_PCV13_strains_in_ipd'] = \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.vaccines["pcv13_30"]["serotypes"])) / \
                         ipd_cases.height if ipd_cases.height  else 0
                             
            self.row["no_of_cases_per_age_group"] = \
                                disease_age_groups["count"].to_list()
            self.row["cases_per_100k_per_age_group"] = frac_cases_age_groups
            
            self.row["no_PPSV23_strains_in_ipd"] =  sum(
                        ipd_cases["exposed_strains"].unique()
                        .is_in(self.vaccines["ppsv23_1"]["serotypes"]))
            
            self.row["no_PPSV23_notPCV13_strains_in_ipd"] = sum(
                        ipd_cases["exposed_strains"].unique()
                        .is_in(self.ppsv23_not_pcv13_strains))
            self.row["fraction_prevalence_PPSV23_strains_in_ipd"] =  \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.vaccines["ppsv23_1"]["serotypes"])) / \
                         ipd_cases.height if ipd_cases.height  else 0
                             
            self.row["fraction_prevalence_PPSV23_notPCV13_strains_in_ipd"] = \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.ppsv23_not_pcv13_strains)) / \
                         ipd_cases.height if ipd_cases.height  else 0
            
            self.row.append()
            self.h5file.flush()
                

    
    
            
    