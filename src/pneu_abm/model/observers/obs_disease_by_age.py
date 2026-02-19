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
    def __init__(self, h5file, vaccines, interval, strains_df):
        #interval is in time ticks not days
        self.interval = interval
        self.sumdata = pl.DataFrame({})
        self.observed_day = self.interval - 1
        self.vaccines = vaccines
        self.no_age_groups = 106
        self.strains_df = strains_df
        #pl.Series("strain_list", list(np.sort(strains))))
        desc = {'t': tb.UInt32Col(pos=0),
                'run_no': tb.UInt32Col(pos=1),
                'cap_per_100k': tb.UInt32Col(pos=2),
                'ipd_per_100k': tb.UInt32Col(pos=3),
                'no_strains_in_ipd': tb.UInt32Col(pos=4),
                "no_PCV7_strains_in_ipd": tb.UInt32Col(pos=5),
                "no_PCV13_strains_in_ipd": tb.UInt32Col(pos=6),
                "no_PPSV23_strains_in_ipd": tb.UInt32Col(pos=7),
                "no_PPSV23_notPCV13_strains_in_ipd": tb.UInt32Col(pos=8),
            "fraction_prevalence_PCV7_strains_in_ipd": tb.Float64Col(pos=9),
            "fraction_prevalence_PCV13_strains_in_ipd": tb.Float64Col(pos=10),
            "fraction_prevalence_PPSV23_strains_in_ipd": tb.Float64Col(pos=11),
   "fraction_prevalence_PPSV23_notPCV13_strains_in_ipd": tb.Float64Col(pos=12),
        "no_of_cap_per_age_group": tb.UInt32Col(pos=13,
                                                  shape=(self.no_age_groups)), 
        "no_of_ipd_per_age_group": tb.UInt32Col(pos=14,
                                                  shape=(self.no_age_groups)), 
        "total_inds_per_age_group": tb.UInt32Col(pos=15,
                                                  shape=(self.no_age_groups)), 
        "cases_per_100k_per_age_group": tb.Float64Col(pos=16,
                                                    shape=(self.no_age_groups)),
        "no_of_cases_per_age_group": tb.Float64Col(pos=17,
                                                    shape=(self.no_age_groups)),
        "no_PCV15_strains_in_ipd": tb.UInt32Col(pos=18),
        "no_PCV20_strains_in_ipd": tb.UInt32Col(pos=19),
        "fraction_prevalence_PCV15_strains_in_ipd": tb.Float64Col(pos=20),
        "fraction_prevalence_PCV20_strains_in_ipd": tb.Float64Col(pos=21),
        "no_PCV7_strains_in_ipd_per_age_group":  tb.UInt32Col(pos=22,
                                                  shape=(self.no_age_groups)), 
        "no_PCV13_strains_in_ipd_per_age_group":  tb.UInt32Col(pos=23,
                                                  shape=(self.no_age_groups)), 
        "no_PPSV23_strains_in_ipd_per_age_group":  tb.UInt32Col(pos=24,
                                                  shape=(self.no_age_groups)), 
        "no_PPSV23_notPCV13_strains_in_ipd_per_age_group":  \
                                            tb.UInt32Col(pos=25,
                                                  shape=(self.no_age_groups)), 
        }
       
        self.age_0_conversion = pl.DataFrame([
           pl.Series("age", [0] * 365),
           pl.Series("age_days", list(range(365))),
           pl.Series("age_coef_by_age", [0] * 31 + [1] * 59 + [2] * 61 + \
                               [3] * (180 + 34)), #+ [-1] * 34 ),
           ]) 
        
           
        ppsv23 = pl.Series("ppsv23",
                    sorted(
                    set(self.strains_df.filter(pl.col("23vPPV"))['serotype'])))
        indexes = (~ppsv23.is_in(
                    sorted(
                    set(self.strains_df.filter(pl.col("13vPCV"))['serotype'])))
                ).to_list()
        indexes =[i for i, x in enumerate(indexes) if x]
        
        self.ppsv23_not_pcv13_strains = ppsv23[indexes]
        self.ppsv23_strains = ppsv23
        self.pcv13_strains = pl.Series("pcv13",
                    sorted(
                    set(self.strains_df.filter(pl.col("13vPCV"))['serotype'])))
        self.pcv15_strains = pl.Series("pcv15",
                    sorted(
                    set(self.strains_df.filter(pl.col("15vPCV"))['serotype'])))
        self.pcv20_strains = pl.Series("pcv20",
                    sorted(
                    set(self.strains_df.filter(pl.col("20vPCV"))['serotype'])))
        super(DiseaseObserverByAge, self).__init__(h5file =h5file, \
                            label = 'disease_byage', description = desc,
                            title = 'DiseaseObserverByAge')
        
    def update_age_conversion(self):
        self.age_0_conversion = pl.DataFrame([
           pl.Series("age", [0] * 365),
           pl.Series("age_days", list(range(365))),
           pl.Series("age_coef_by_age", [0] * 59 + [1] * 61 + [2] * 61 + \
                               [3] * (184)), 
           ]) 
    
    def update(self, t, pop, **kwargs):
        
        if t % self.interval == self.observed_day:
            
            year = t / self.interval
            #agg data
            run_nos = pop.disease_pop["clinical_model_no"].unique()
            
            for run_no in run_nos:
                cur_pop_disease = pop.disease_pop.filter(
                    pl.col("clinical_model_no") == run_no)
            
                disease_counts = cur_pop_disease.group_by(
                                            "disease").agg(pl.count())
                cur_pop_disease = cur_pop_disease.join(self.age_0_conversion,
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
                       cur_pop_disease.group_by("age_coef_by_age").agg(pl.count()),
                      on = "age_coef_by_age", how= "left" ).sort("age_coef_by_age")
                cap_age_groups = pl.DataFrame([
                    pl.Series("age_coef_by_age", list(range(self.no_age_groups))),
                    pl.Series("count", [0] * self.no_age_groups),
                    ]).update(
                       cur_pop_disease.filter(pl.col("disease") == "cap")
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
                       cur_pop_disease.filter(pl.col("disease") == "ipd")
                       .with_columns(
                           (pl.col("exposed_strains")
                           .is_in(sorted(
                           set(self.strains_df.filter(pl.col("7vPCV"))['serotype'])))
                           .alias("pcv7_strains")),
                           (pl.col("exposed_strains")
                           .is_in(sorted(
                           set(self.strains_df.filter(pl.col("13vPCV"))['serotype'])))
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
                
                cur_pop = pop.I.join(self.age_0_conversion,
                            on= ["age", "age_days"],how="left").with_columns(
                                    (pl.when(pl.col("age_coef_by_age").is_null())
                                    .then((pl.col("age") + 4))
                                    .otherwise(pl.col("age_coef_by_age")))
                                        .alias("age_coef_by_age")
                                    )
                
                #pop.I.filter(pl.col("age_coef") == 16)
                popsize_age_groups = pl.DataFrame([
                    pl.Series("age_coef_by_age", list(range(self.no_age_groups))),
                    pl.Series("count", [0] * self.no_age_groups),
                    ]).update(
                    cur_pop.group_by("age_coef_by_age").agg(pl.count()),
                    on = "age_coef_by_age", how= "left" ).sort("age_coef_by_age")
                
                frac_cases_age_groups = ((disease_age_groups["count"] / 
                         popsize_age_groups["count"]) * 100_000).round(2).to_list()
                
                ipd_cases = cur_pop_disease.filter(pl.col("disease") == "ipd")
                
                self.row['t'] = year
                self.row['run_no'] = run_no
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
                                    .is_in(sorted(
                                    set(self.strains_df.filter(pl.col("7vPCV"))['serotype']))))
                self.row['no_PCV13_strains_in_ipd'] = \
                                        sum(ipd_cases["exposed_strains"].unique()
                                    .is_in(sorted(
                                    set(self.strains_df.filter(pl.col("13vPCV"))['serotype']))))
                self.row['fraction_prevalence_PCV7_strains_in_ipd'] = \
                            sum(ipd_cases["exposed_strains"]
                        .is_in(sorted(
                        set(self.strains_df.filter(pl.col("7vPCV"))['serotype'])))) / \
                              ipd_cases.height  if ipd_cases.height  else 0
                self.row['fraction_prevalence_PCV13_strains_in_ipd'] = \
                        sum(ipd_cases["exposed_strains"]
                        .is_in(sorted(
                        set(self.strains_df.filter(pl.col("13vPCV"))['serotype'])))) / \
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
                

    
    
            
    