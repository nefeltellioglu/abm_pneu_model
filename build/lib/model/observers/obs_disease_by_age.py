"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class DiseaseObserverByAge(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.interval = interval
        self.sumdata = pl.DataFrame({})
        self.observed_day = self.interval - 1
        self.vaccines = vaccines
        self.no_age_groups = 106
        
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
        "no_of_cap_per_age_group": tb.UInt32Col(pos=12,
                                                  shape=(self.no_age_groups)), 
        "no_of_ipd_per_age_group": tb.UInt32Col(pos=13,
                                                  shape=(self.no_age_groups)), 
        "total_inds_per_age_group": tb.UInt32Col(pos=14,
                                                  shape=(self.no_age_groups)), 
        "cases_per_100k_per_age_group": tb.Float64Col(pos=15,
                                                    shape=(self.no_age_groups)),
        "no_of_cases_per_age_group": tb.Float64Col(pos=16,
                                                    shape=(self.no_age_groups)),
        "no_PCV15_strains_in_ipd": tb.UInt32Col(pos=17),
        "no_PCV20_strains_in_ipd": tb.UInt32Col(pos=18),
        "fraction_prevalence_PCV15_strains_in_ipd": tb.Float64Col(pos=19),
        "fraction_prevalence_PCV20_strains_in_ipd": tb.Float64Col(pos=20),
        "no_PCV7_strains_in_ipd_per_age_group":  tb.UInt32Col(pos=21,
                                                  shape=(self.no_age_groups)), 
        "no_PCV13_strains_in_ipd_per_age_group":  tb.UInt32Col(pos=22,
                                                  shape=(self.no_age_groups)), 
        "no_PPSV23_strains_in_ipd_per_age_group":  tb.UInt32Col(pos=23,
                                                  shape=(self.no_age_groups)), 
        "no_PPSV23_notPCV13_strains_in_ipd_per_age_group":  \
                                            tb.UInt32Col(pos=24,
                                                  shape=(self.no_age_groups)), 
        }
       
        self.age_0_conversion = pl.DataFrame([
           pl.Series("age", [0] * 330),
           pl.Series("age_days", list(range(330))),
           pl.Series("age_coef_by_age", [0] * 30 + [1] * 60 + [2] * 60 + \
                               [3] * (180)), #+ [-1] * 34 ),
           ]) 
        
        """self.age1_conversion = pl.DataFrame([
            pl.Series("age", list(range(0,102))),
             pl.Series("age_coef", [4] * 4 + [5] * 10 + [6] * 10 + \
                                   [7] * 25 + [8] * 15 + [9] * 37 
                       ), #+ [-1] * 34 ),
             ])"""
            
        ppsv23 = pl.Series("ppsv23",
                    ["1","2", "3", "4", "5", "6B","7F","8","9N", "9V","10A",
                      "11A","12F","14","15B","17F", "18C","19A", "19F","20",
                      "22F", '23F', "33F"])
        indexes = (~ppsv23.is_in(
                        self.vaccines["pcv13_30"]["serotypes"])).to_list()
        indexes =[i for i, x in enumerate(indexes) if x]
        
        self.ppsv23_not_pcv13_strains = ppsv23[indexes]
        self.ppsv23_strains = ppsv23
        self.pcv13_strains = pl.Series("pcv13",
                    ["1", "3", "4", "5", "6A", "6B","7F", "9V", "14",
                                  "18C","19A", "19F", '23F' ])
        self.pcv15_strains = pl.Series("pcv15",
                    ["1", "3", "4", "5", "6A", 
                                  "6B","7F", "9V", "14","18C",
                                  "19A", "19F", "22F",'23F', "33F"])
        self.pcv20_strains = pl.Series("pcv20",
                    [ "1","10A","11A","12F","14",
                                    "15B","18C","19A","19F","22F",
                                    "23F","3","33F","4","5",
                                    "6A","6B","7F","8","9V"])
        super(DiseaseObserverByAge, self).__init__(h5file =h5file, \
                            label = 'disease_byage', description = desc,
                            title = 'DiseaseObserverByAge')
        

    
    def update(self, t, pop, **kwargs):
        
        if t % self.interval == self.observed_day:
            
            year = t / self.interval
            #agg data
            disease_counts = pop.disease_pop.group_by(
                                        "disease").agg(pl.count())
            pop.disease_pop = pop.disease_pop.join(self.age_0_conversion,
                        on= ["age", "age_days"],how="left").with_columns(
                                (pl.when(pl.col("age_coef_by_age").is_null())
                                .then((pl.col("age") + 4))
                                .otherwise(pl.col("age_coef_by_age")))
                                    .alias("age_coef_by_age")
                                )
            disease_age_groups = pl.DataFrame([
                pl.Series("age_coef_by_age", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                ]).update(
                   pop.disease_pop.group_by("age_coef_by_age").agg(pl.count()),
                  on = "age_coef_by_age", how= "left" ).sort("age_coef_by_age")
            cap_age_groups = pl.DataFrame([
                pl.Series("age_coef_by_age", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                ]).update(
                   pop.disease_pop.filter(pl.col("disease") == "cap")
                   .group_by("age_coef_by_age").agg(pl.count()),
                  on = "age_coef_by_age", how= "left" ).sort("age_coef_by_age")
            ipd_age_groups = pl.DataFrame([
                pl.Series("age_coef_by_age", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                pl.Series("pcv7_strains", [0] * self.no_age_groups),
                pl.Series("pcv13_strains", [0] * self.no_age_groups),
                pl.Series("ppsv23_strains", [0] * self.no_age_groups),
                pl.Series("ppsv23_notpcv13_strains", [0] * self.no_age_groups),
                ]).update(
                   pop.disease_pop.filter(pl.col("disease") == "ipd")
                   .with_columns(
                       (pl.col("exposed_strains")
                       .is_in(self.vaccines["pcv7"]["serotypes"])
                       .alias("pcv7_strains")),
                       (pl.col("exposed_strains")
                       .is_in(self.vaccines["pcv13_30"]["serotypes"])
                       .alias("pcv13_strains")),
                       (pl.col("exposed_strains")
                       .is_in(self.ppsv23_strains)
                       .alias("ppsv23_strains")),
                       (pl.col("exposed_strains")
                       .is_in(self.ppsv23_not_pcv13_strains)
                       .alias("ppsv23_notpcv13_strains")),
                       )
                   .group_by("age_coef_by_age").agg(
                       count = pl.count(),
                    pcv7_strains = pl.sum("pcv7_strains"),
                    pcv13_strains = pl.sum("pcv13_strains"),
                    ppsv23_strains = pl.sum("ppsv23_strains"),
                ppsv23_notpcv13_strains = pl.sum("ppsv23_notpcv13_strains")),
                  on = "age_coef_by_age", how= "left" ).sort("age_coef_by_age")
            
            """if pop.disease_pop.height and \
                pop.disease_pop["age_coef"].max() > 17:
                print("obs_disese: wrong age categories")"""
            
            #if pop.disease_pop.height:
            #    print("here")
            #create age groups for the pop   
            
            
            cur_pop = pop.I.join(self.age_0_conversion,
                        on= ["age", "age_days"],how="left").with_columns(
                                (pl.when(pl.col("age_coef_by_age").is_null())
                                .then((pl.col("age") + 4))
                                .otherwise(pl.col("age_coef_by_age")))
                                    .alias("age_coef_by_age")
                                )
            #cur_pop.filter(pl.col("age_coef1").is_null())
            #cur_pop = cur_pop.update(self.age1_conversion,
            #                     on= ["age"],how="left")
            
            #remove 11-12m group
            """cur_pop = cur_pop.filter(
                (pl.col("age_coef") >= 0) |
                ((pl.col("age") > 0)) |
                ((pl.col("age") == 0) &
                 (pl.col("age_days") <= 47)))"""
            
            
            
            #pop.I.filter(pl.col("age_coef") == 16)
            popsize_age_groups = pl.DataFrame([
                pl.Series("age_coef_by_age", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                ]).update(
                cur_pop.group_by("age_coef_by_age").agg(pl.count()),
                on = "age_coef_by_age", how= "left" ).sort("age_coef_by_age")
            
            frac_cases_age_groups = ((disease_age_groups["count"] / 
                     popsize_age_groups["count"]) * 100_000).round(2).to_list()
            
            ipd_cases = pop.disease_pop.filter(pl.col("disease") == "ipd")
            
            """ipd_cases = ipd_cases.with_columns(
                (pl.col("exposed_strains")
                .is_in(self.vaccines["pcv7"]["serotypes"])
                .alias("pcv7_strains")),
                (pl.col("exposed_strains")
                .is_in(self.vaccines["pcv13_30"]["serotypes"])
                .alias("pcv13_strains")),
                (pl.col("exposed_strains")
                .is_in(self.ppsv23_strains)
                .alias("ppsv23_strains")),
                (pl.col("exposed_strains")
                .is_in(self.ppsv23_not_pcv13_strains)
                .alias("ppsv23_notpcv13_strains")),
                )
            
            ipd_cases_per_agegroups = (ipd_cases.group_by("age_coef_by_age")
                                       .agg(
                pcv7_strains = pl.sum("pcv7_strains"),
                pcv13_strains = pl.sum("pcv13_strains"),
                ppsv23_strains = pl.sum("ppsv23_strains"),
                ppsv23_notpcv13_strains = pl.sum("ppsv23_notpcv13_strains"))
                                .sort("age_coef_by_age"))"""
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
            self.row["no_of_cap_per_age_group"] = \
                cap_age_groups["count"].to_list()
            self.row["no_of_ipd_per_age_group"] = \
                ipd_age_groups["count"].to_list()
            self.row["total_inds_per_age_group"] = \
                popsize_age_groups["count"].to_list()
            
            self.row["no_PPSV23_strains_in_ipd"] =  sum(
                        ipd_cases["exposed_strains"].unique()
                        .is_in(self.ppsv23_strains))
            self.row["no_PCV15_strains_in_ipd"] =  sum(
                        ipd_cases["exposed_strains"].unique()
                        .is_in(self.pcv15_strains))
            self.row["no_PCV20_strains_in_ipd"] =  sum(
                        ipd_cases["exposed_strains"].unique()
                        .is_in(self.pcv20_strains))
            
            
            
            self.row["no_PPSV23_notPCV13_strains_in_ipd"] = sum(
                        ipd_cases["exposed_strains"].unique()
                        .is_in(self.ppsv23_not_pcv13_strains))
            self.row["fraction_prevalence_PPSV23_strains_in_ipd"] =  \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.ppsv23_strains)) / \
                         ipd_cases.height if ipd_cases.height  else 0
                             
            self.row["fraction_prevalence_PPSV23_notPCV13_strains_in_ipd"] = \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.ppsv23_not_pcv13_strains)) / \
                         ipd_cases.height if ipd_cases.height  else 0
            
            self.row["fraction_prevalence_PCV15_strains_in_ipd"] = \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.pcv15_strains)) / \
                         ipd_cases.height if ipd_cases.height  else 0
            self.row["fraction_prevalence_PCV20_strains_in_ipd"] = \
                    sum(ipd_cases["exposed_strains"]
                    .is_in(self.pcv20_strains)) / \
                         ipd_cases.height if ipd_cases.height  else 0
                      
                         
            self.row["no_PCV7_strains_in_ipd_per_age_group"] = \
                ipd_age_groups["pcv7_strains"].to_list()
            self.row["no_PCV13_strains_in_ipd_per_age_group"] = \
                ipd_age_groups["pcv13_strains"].to_list()
            self.row["no_PPSV23_strains_in_ipd_per_age_group"] = \
                ipd_age_groups["ppsv23_strains"].to_list()
            self.row["no_PPSV23_notPCV13_strains_in_ipd_per_age_group"] = \
                ipd_age_groups["ppsv23_notpcv13_strains"].to_list()
            self.row.append()
            self.h5file.flush()
                

    
    
            
    