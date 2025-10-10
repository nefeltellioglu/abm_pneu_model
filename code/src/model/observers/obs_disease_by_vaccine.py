"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class DiseaseObserverByVaccineAtRisk(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.interval = interval
        self.sumdata = pl.DataFrame({})
        self.observed_day = self.interval - 1
        self.vaccines = vaccines
        self.no_vaccines = 30
        self.vacc_name_length = 30
        self.age_conversion = pl.DataFrame([
           pl.Series("age", list(range(102))),
           pl.Series("dis_age_groups", ["0-4"] * 5 + ["5-14"] * 10 + \
                     ["15-24"] * 10 + ["25-49"] * 25 + \
                     ["50-69"] * 20 + ["70+"] * 32), #+ [-1] * 34 ),
           ]) 
        self.no_age_groups = len(
                            self.age_conversion["dis_age_groups"].unique())
        self.age_groups = ["0-4", "5-14", "15-24", "25-49", "50-69", "70+"]
        self.age_groups_order = {val: idx 
                                 for idx, val in enumerate(self.age_groups)}
        
        #pl.Series("strain_list", list(np.sort(strains))))
        desc = {'t': tb.UInt32Col(pos=0),
                'cap_per_100k_per_vaccines': tb.Float64Col(pos=1,
                                              shape=(self.no_vaccines)), 
                'ipd_per_100k_per_vaccines': tb.Float64Col(pos=2,
                                              shape=(self.no_vaccines)),
                'no_of_cap_per_vaccines':tb.UInt32Col(pos=3,
                                              shape=(self.no_vaccines)),
                'no_of_ipd_per_vaccines': tb.UInt32Col(pos=4,
                                              shape=(self.no_vaccines)),
                'vaccine_names': tb.StringCol(pos=5,
                                              itemsize = self.vacc_name_length,
                                              shape=(self.no_vaccines)),
                'cap_per_100k_per_vaccines_last5_years': tb.Float64Col(pos=6,
                                              shape=(self.no_vaccines)), 
                'ipd_per_100k_per_vaccines_last5_years': tb.Float64Col(pos=7,
                                              shape=(self.no_vaccines)),
                'no_of_cap_per_vaccines_last5_years':tb.UInt32Col(pos=8,
                                              shape=(self.no_vaccines)),
                'no_of_ipd_per_vaccines_last5_years': tb.UInt32Col(pos=9,
                                              shape=(self.no_vaccines)),
                'cap_per_100k_per_vaccines_last1_years': tb.Float64Col(pos=10,
                                              shape=(self.no_vaccines)), 
                'ipd_per_100k_per_vaccines_last1_years': tb.Float64Col(pos=11,
                                              shape=(self.no_vaccines)),
                'no_of_cap_per_vaccines_last1_years':tb.UInt32Col(pos=12,
                                              shape=(self.no_vaccines)),
                'no_of_ipd_per_vaccines_last1_years': tb.UInt32Col(pos=13,
                                              shape=(self.no_vaccines)),
                'no_of_cap_per_vaccines_per_age_group':tb.UInt32Col(pos=14,
                              shape=(self.no_age_groups, self.no_vaccines)),
                'no_of_ipd_per_vaccines_per_age_group': tb.UInt32Col(pos=15,
                              shape=(self.no_age_groups, self.no_vaccines)),
                'no_of_inds_per_age_group':tb.UInt32Col(pos=16,
                              shape=(self.no_age_groups, self.no_vaccines)),
                'no_of_cap_per_vaccines_per_age_group_last5_years': \
                    tb.UInt32Col(pos=16,
                              shape=(self.no_age_groups, self.no_vaccines)),
                'no_of_ipd_per_vaccines_per_age_group_last5_years': \
                    tb.UInt32Col(pos=17,
                              shape=(self.no_age_groups, self.no_vaccines)),
                'no_of_inds_per_age_group_last5_years':tb.UInt32Col(pos=18,
                              shape=(self.no_age_groups, self.no_vaccines)),
                
         
        }
        """"
        "tier_1_no_of_at_risk_cap_per_vaccines": tb.UInt32Col(pos=16,
                                                  shape=(self.no_vaccines)), 
        "tier_1_no_of_at_risk_ipd_per_vaccines": tb.UInt32Col(pos=17,
                                                  shape=(self.no_vaccines)), 
        "tier_2_no_of_at_risk_cap_per_vaccines": tb.UInt32Col(pos=18,
                                                  shape=(self.no_vaccines)), 
        "tier_2_no_of_at_risk_ipd_per_vaccines": tb.UInt32Col(pos=19,
                                                  shape=(self.no_vaccines)), 
        
        
        tier_2_at_risk_no_PCV15_notPCV13_strains_in_ipd_per_age_group":  \
                                           tb.UInt32Col(pos=42,
                                                 shape=(self.no_age_groups)),
       "tier_2_at_risk_no_PCV20_notPCV15_strains_in_ipd_per_age_group":  \
                                           tb.UInt32Col(pos=43,
                                                 shape=(self.no_age_groups)),
       tier_1_at_risk_no_PCV15_notPCV13_strains_in_ipd_per_age_group":  \
                                           tb.UInt32Col(pos=37,
                                                 shape=(self.no_age_groups)),
       "tier_1_at_risk_no_PCV20_notPCV15_strains_in_ipd_per_age_group":  \
                                           tb.UInt32Col(pos=38,
                                                 shape=(self.no_age_groups)),
                                           """
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
        self.pcv15_not_pcv13 = pl.Series("pcv15_not_pcv13",
                    ["22F", "33F"])
        
        self.pcv20_not_pcv15 = pl.Series("pcv20_not_pcv15",
                    ["10A","11A","12F","15B", "8"])
        
        self.ppsv23_not_pcv205 = pl.Series("ppsv23_not_pcv20",
                    ["2","9N","17F","20" ])
        self.pcv20_strains = pl.Series("pcv20",
                    [ "1","10A","11A","12F","14",
                                    "15B","18C","19A","19F","22F",
                                    "23F","3","33F","4","5",
                                    "6A","6B","7F","8","9V"])
        super(DiseaseObserverByVaccineAtRisk, self).__init__(h5file =h5file, \
                            label = 'disease_byvaccine', description = desc,
                            title = 'DiseaseObserverByVaccineAtRisk')
        
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
            disease_pop1 = pop.disease_pop.drop("vaccine_type"
                                ).unnest("vaccines").with_columns(
                     (pl.when(pl.col("vaccine_type") == "")
                     .then(pl.lit("not_vaccinated"))
                     .otherwise(pl.col("vaccine_type"))).alias("vaccine_type"))
            
            
            disease_counts_per_vaccine = disease_pop1.group_by("disease", 
                            pl.col("vaccine_type")
                            ).agg(pl.count())
            vaccine_names = list(self.vaccines.keys()) + ["not_vaccinated"]
            vaccine_order = {val: idx for idx, val in enumerate(vaccine_names)}
             
            disease_counts_per_vaccine = disease_counts_per_vaccine.sort(
                pl.col("disease"),
                pl.col("vaccine_type").map_dict(vaccine_order))
            
            disease_counts_per_vaccine_df = pl.DataFrame([
                pl.Series("disease", ["ipd"] * len(vaccine_names) + \
                          ["cap"] * len(vaccine_names) ),
               pl.Series("vaccine_type", vaccine_names * 2),
                pl.Series("count", [0] * len(vaccine_names) * 2),
                ]).update(disease_counts_per_vaccine, on = ["disease", 
                                            "vaccine_type"], how = "left")
            disease_counts_per_vaccine_df_last5_years = pl.DataFrame([
                pl.Series("disease", ["ipd"] * len(vaccine_names) + \
                          ["cap"] * len(vaccine_names) ),
               pl.Series("vaccine_type", vaccine_names * 2),
                pl.Series("count", [0] * len(vaccine_names) * 2),
                ]).update(disease_pop1.filter(
                    pl.col("final_vaccine_time")/ (364/ self.interval) + \
                        self.interval * 5 >= t)
                .group_by("disease", pl.col("vaccine_type")).agg(pl.count()), 
                on = ["disease", "vaccine_type"], how = "left")
            
            disease_counts_per_vaccine_df_last1_years = pl.DataFrame([
                pl.Series("disease", ["ipd"] * len(vaccine_names) + \
                          ["cap"] * len(vaccine_names) ),
               pl.Series("vaccine_type", vaccine_names * 2),
                pl.Series("count", [0] * len(vaccine_names) * 2),
                ]).update(disease_pop1.filter(
                    pl.col("final_vaccine_time")/ (364/ self.interval) + \
                        self.interval * 1 >= t)
                .group_by("disease", pl.col("vaccine_type")).agg(pl.count()), 
                on = ["disease", "vaccine_type"], how = "left")
            
            
            #disease_pop1.filter(pl.col("vaccine_type") == "pcv7")["final_vaccine_time"]                                              
                                                            
            disease_pop1 = disease_pop1.join(self.age_conversion,
                        on= ["age"],how="left")
            
            
            
            cap_age_groups = pl.DataFrame([
                pl.Series("dis_age_groups", 
                          self.age_groups  * len(vaccine_names)),
                pl.Series("vaccine_type", [vacc for vacc in vaccine_names 
                                         for i in range(self.no_age_groups)]),
                pl.Series("count", 
                          [0] * self.no_age_groups * len(vaccine_names)),
                ]).update(
                   disease_pop1.filter(pl.col("disease") == "cap")
                   .group_by("dis_age_groups", "vaccine_type").agg(pl.count()),
                  on = ["dis_age_groups", "vaccine_type"], how= "left" ).sort(
                      pl.col("dis_age_groups").map_dict(self.age_groups_order), 
                      pl.col("vaccine_type").map_dict(vaccine_order)
                      ).group_by("dis_age_groups").agg("count").sort(
                             pl.col("dis_age_groups")
                             .map_dict(self.age_groups_order)
                             )["count"].to_list()
            ipd_age_groups =  pl.DataFrame([
                pl.Series("dis_age_groups", 
                          self.age_groups  * len(vaccine_names)),
                pl.Series("vaccine_type", [vacc for vacc in vaccine_names 
                                         for i in range(self.no_age_groups)]),
                pl.Series("count", 
                          [0] * self.no_age_groups * len(vaccine_names)),
                ]).update(
                   disease_pop1.filter(pl.col("disease") == "ipd")
                   .group_by("dis_age_groups", "vaccine_type").agg(pl.count()),
                  on = ["dis_age_groups", "vaccine_type"], how= "left" ).sort(
                      pl.col("dis_age_groups").map_dict(self.age_groups_order), 
                      pl.col("vaccine_type").map_dict(vaccine_order)
                      ).group_by("dis_age_groups").agg("count").sort(
                             pl.col("dis_age_groups")
                             .map_dict(self.age_groups_order)
                             )["count"].to_list()
            cap_age_groups_last5_years = pl.DataFrame([
                pl.Series("dis_age_groups", 
                          self.age_groups  * len(vaccine_names)),
                pl.Series("vaccine_type", [vacc for vacc in vaccine_names 
                                         for i in range(self.no_age_groups)]),
                pl.Series("count", 
                          [0] * self.no_age_groups * len(vaccine_names)),
                ]).update(
                   disease_pop1.filter((pl.col("disease") == "cap") &
                    (pl.col("final_vaccine_time")/ (364/ self.interval) + \
                                         self.interval * 1 >= t))
                   .group_by("dis_age_groups", "vaccine_type").agg(pl.count()),
                  on = ["dis_age_groups", "vaccine_type"], how= "left" ).sort(
                      pl.col("dis_age_groups").map_dict(self.age_groups_order), 
                      pl.col("vaccine_type").map_dict(vaccine_order)
                      ).group_by("dis_age_groups").agg("count").sort(
                             pl.col("dis_age_groups")
                             .map_dict(self.age_groups_order)
                             )["count"].to_list()
            ipd_age_groups_last5_years = pl.DataFrame([
                pl.Series("dis_age_groups", 
                          self.age_groups  * len(vaccine_names)),
                pl.Series("vaccine_type", [vacc for vacc in vaccine_names 
                                         for i in range(self.no_age_groups)]),
                pl.Series("count", 
                          [0] * self.no_age_groups * len(vaccine_names)),
                ]).update(
                   disease_pop1.filter((pl.col("disease") == "ipd") &
                    (pl.col("final_vaccine_time")/ (364/ self.interval) + \
                                         self.interval * 1 >= t))
                   .group_by("dis_age_groups", "vaccine_type").agg(pl.count()),
                  on = ["dis_age_groups", "vaccine_type"], how= "left" ).sort(
                      pl.col("dis_age_groups").map_dict(self.age_groups_order), 
                      pl.col("vaccine_type").map_dict(vaccine_order)
                      ).group_by("dis_age_groups").agg("count").sort(
                             pl.col("dis_age_groups")
                             .map_dict(self.age_groups_order)
                             )["count"].to_list()
                 
           
            
            
            cur_pop = pop.I.join(self.age_conversion,
                        on= ["age"],how="left")
            
                            
            cur_pop = cur_pop.unnest("vaccines").with_columns(
                     (pl.when(pl.col("vaccine_type") == "")
                     .then(pl.lit("not_vaccinated"))
                     .otherwise(pl.col("vaccine_type"))).alias("vaccine_type"))
            
            no_vacc_given_all_pop = pl.DataFrame([
                pl.Series("vaccine_type", vaccine_names ),
                pl.Series("count", [0] * len(vaccine_names) ),
                ]).update(cur_pop.group_by("vaccine_type").agg(pl.count())
                , on = ["vaccine_type"], how = "left")
            no_vacc_given_all_pop_last5_years = pl.DataFrame([
                pl.Series("vaccine_type", vaccine_names ),
                pl.Series("count", [0] * len(vaccine_names) ),
                ]).update(cur_pop.filter(
                    pl.col("final_vaccine_time")/ (364/ self.interval)  + \
            self.interval * 5 >= t).group_by("vaccine_type").agg(pl.count())
                , on = ["vaccine_type"], how = "left")
            
            no_vacc_given_all_pop_last1_years = pl.DataFrame([
                pl.Series("vaccine_type", vaccine_names ),
                pl.Series("count", [0] * len(vaccine_names) ),
                ]).update(cur_pop.filter(
                    pl.col("final_vaccine_time")/ (364/ self.interval)  + \
            self.interval * 1 >= t).group_by("vaccine_type").agg(pl.count())
                , on = ["vaccine_type"], how = "left")
            
            no_inds_age_groups = pl.DataFrame([
                pl.Series("dis_age_groups", 
                          self.age_groups  * len(vaccine_names)),
                pl.Series("vaccine_type", [vacc for vacc in vaccine_names 
                                         for i in range(self.no_age_groups)]),
                pl.Series("count", 
                          [0] * self.no_age_groups * len(vaccine_names)),
                ]).update(
                   cur_pop.group_by("dis_age_groups", 
                                    "vaccine_type").agg(pl.count()),
                  on = ["dis_age_groups", "vaccine_type"], how= "left").sort(
                      pl.col("dis_age_groups").map_dict(self.age_groups_order), 
                      pl.col("vaccine_type").map_dict(vaccine_order)
                      ).group_by("dis_age_groups").agg("count").sort(
                             pl.col("dis_age_groups")
                             .map_dict(self.age_groups_order)
                             )["count"].to_list()
            no_inds_age_groups_last5_years = pl.DataFrame([
                pl.Series("dis_age_groups", 
                          self.age_groups  * len(vaccine_names)),
                pl.Series("vaccine_type", [vacc for vacc in vaccine_names 
                                         for i in range(self.no_age_groups)]),
                pl.Series("count", 
                          [0] * self.no_age_groups * len(vaccine_names)),
                ]).update(
                   cur_pop.filter(
                       pl.col("final_vaccine_time")/ (364/ self.interval)  + \
               self.interval * 5 >= t).group_by("dis_age_groups", 
                                    "vaccine_type").agg(pl.count()),
                  on = ["dis_age_groups", "vaccine_type"], how= "left").sort(
                      pl.col("dis_age_groups").map_dict(self.age_groups_order), 
                      pl.col("vaccine_type").map_dict(vaccine_order)
                      ).group_by("dis_age_groups").agg("count").sort(
                             pl.col("dis_age_groups")
                             .map_dict(self.age_groups_order)
                             )["count"].to_list()
                          
                          
                          
            #cur_pop.filter(pl.col("age_coef1").is_null())
            #cur_pop = cur_pop.update(self.age1_conversion,
            #                     on= ["age"],how="left")
            
            #remove 11-12m group
            """cur_pop = cur_pop.filter(
                (pl.col("age_coef") >= 0) |
                ((pl.col("age") > 0)) |
                ((pl.col("age") == 0) &
                 (pl.col("age_days") <= 47)))"""
            
            
            
           
            ###########record the outputs 
            
            
            no_vaccines = len(vaccine_names)
            no_missing_rows = self.no_vaccines  - no_vaccines
            self.row['t'] = year
            
            
            self.row['cap_per_100k_per_vaccines'] =   \
                [100_000 * x/y  if y else 0 for (x,y) in zip(
                    (disease_counts_per_vaccine_df.filter(
                        pl.col("disease") == "cap").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()), \
                no_vacc_given_all_pop.sort(pl.col("vaccine_type").map_dict(
                        vaccine_order))["count"].to_list())] + \
                                                    [0] * no_missing_rows
            self.row['ipd_per_100k_per_vaccines'] =  \
                [100_000 * x/y  if y else 0 for (x,y) in zip(
                    (disease_counts_per_vaccine_df.filter(
                        pl.col("disease") == "ipd").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()), \
                no_vacc_given_all_pop.sort(pl.col("vaccine_type").map_dict(
                        vaccine_order))["count"].to_list())] + \
                                                    [0] * no_missing_rows
            
            
                    
                    
                    
           
            self.row['no_of_cap_per_vaccines'] = (
                        disease_counts_per_vaccine_df.filter(
                        pl.col("disease") == "cap").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()) + \
                                            [0] * no_missing_rows
            self.row['no_of_ipd_per_vaccines'] =   (
                        disease_counts_per_vaccine_df.filter(
                        pl.col("disease") == "ipd").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()) + \
                                            [0] * no_missing_rows
            self.row['vaccine_names'] =  vaccine_names + \
                                        [""] * no_missing_rows
            
            self.row['cap_per_100k_per_vaccines_last5_years'] =   \
                [100_000 * x/y if y else 0 for (x,y) in zip(
                    (disease_counts_per_vaccine_df_last5_years.filter(
                        pl.col("disease") == "cap").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()), \
                no_vacc_given_all_pop_last5_years.sort(
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list())] + \
                                            [0] * no_missing_rows
            self.row['ipd_per_100k_per_vaccines_last5_years'] =    \
                [100_000 * x/y if y else 0 for (x,y) in zip(
                    (disease_counts_per_vaccine_df_last5_years.filter(
                        pl.col("disease") == "ipd").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()), \
                no_vacc_given_all_pop_last5_years.sort(
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list())] + \
                                            [0] * no_missing_rows
            self.row['no_of_cap_per_vaccines_last5_years'] =  \
                        (disease_counts_per_vaccine_df_last5_years.filter(
                        pl.col("disease") == "cap").sort(pl.col("disease"),\
                            pl.col("vaccine_type").map_dict(vaccine_order)
                            )["count"].to_list()) + \
                                                        [0] * no_missing_rows
            self.row['no_of_ipd_per_vaccines_last5_years'] =   \
                        (disease_counts_per_vaccine_df_last5_years.filter(
                        pl.col("disease") == "ipd").sort(pl.col("disease"),\
                            pl.col("vaccine_type").map_dict(vaccine_order)
                            )["count"].to_list())+ \
                                                        [0] * no_missing_rows
                                                         
                                                         
                                                         
                                                         
                                                         
            self.row['cap_per_100k_per_vaccines_last1_years'] =    \
                [100_000 * x/y if y else 0 for (x,y) in zip(
                    (disease_counts_per_vaccine_df_last1_years.filter(
                        pl.col("disease") == "cap").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()), \
                no_vacc_given_all_pop_last1_years.sort(
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list())] + \
                                            [0] * no_missing_rows
            self.row['ipd_per_100k_per_vaccines_last1_years'] =   \
                [100_000 * x/y if y else 0 for (x,y) in zip(
                    (disease_counts_per_vaccine_df_last1_years.filter(
                        pl.col("disease") == "ipd").sort(pl.col("disease"),\
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list()), \
                no_vacc_given_all_pop_last1_years.sort(
                pl.col("vaccine_type").map_dict(vaccine_order)
                )["count"].to_list())] + \
                                            [0] * no_missing_rows
            self.row['no_of_cap_per_vaccines_last1_years'] = \
                        (disease_counts_per_vaccine_df_last1_years.filter(
                        pl.col("disease") == "cap").sort(pl.col("disease"),\
                            pl.col("vaccine_type").map_dict(vaccine_order)
                            )["count"].to_list()) + \
                                                        [0] * no_missing_rows
            self.row['no_of_ipd_per_vaccines_last1_years'] =   \
                        (disease_counts_per_vaccine_df_last1_years.filter(
                        pl.col("disease") == "ipd").sort(pl.col("disease"),\
                            pl.col("vaccine_type").map_dict(vaccine_order)
                            )["count"].to_list()) + \
                                                        [0] * no_missing_rows
                                                         
                                                         
            #TODO
            self.row['no_of_cap_per_vaccines_per_age_group'] =  \
                [i + [0] * no_missing_rows for i in cap_age_groups]
            self.row['no_of_ipd_per_vaccines_per_age_group'] =  \
                [i + [0] * no_missing_rows for i in ipd_age_groups]
            
            self.row['no_of_inds_per_age_group'] =  \
                [i + [0] * no_missing_rows for i in no_inds_age_groups]
            self.row['no_of_cap_per_vaccines_per_age_group_last5_years'] = \
                [i + [0] * no_missing_rows for i in cap_age_groups_last5_years]
            self.row['no_of_ipd_per_vaccines_per_age_group_last5_years'] = \
                [i + [0] * no_missing_rows for i in ipd_age_groups_last5_years]
            self.row['no_of_inds_per_age_group_last5_years'] = \
                [i + [0] * no_missing_rows 
                 for i in no_inds_age_groups_last5_years]
           
                         
          


            """
            self.row["tier_1_no_of_at_risk_cap_per_vaccines"] =   [0] * self.no_vaccines
            self.row["tier_1_no_of_at_risk_ipd_per_vaccines"] =  [0] * self.no_vaccines
            self.row["tier_2_no_of_at_risk_cap_per_vaccines"] =   [0] * self.no_vaccines
            self.row["tier_2_no_of_at_risk_ipd_per_vaccines"] =  [0] * self.no_vaccines
             """
            
            
            
            
            
            
            
                
            self.row.append()
            self.h5file.flush()
                

    
    
            
    