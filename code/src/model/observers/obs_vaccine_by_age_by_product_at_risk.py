"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class VaccineObserverByAgeByProduct(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.interval = interval
        self.sumdata = pl.DataFrame({})
        self.observed_day = self.interval - 1
        self.vaccines = vaccines
        self.no_age_groups = 106
        
        #pl.Series("strain_list", list(np.sort(strains))))
        desc = {'t': tb.UInt32Col(pos=0),
                'individual_type': tb.StringCol(pos=1, itemsize = 30),
                'vaccine_type': tb.StringCol(pos=2, itemsize = 30),
                'disease': tb.StringCol(pos=3, itemsize = 20),
                'vaccine_delivered': tb.StringCol(pos=4, itemsize = 20),
                "inds_per_age_group": tb.UInt32Col(pos=5,
                                                  shape=(self.no_age_groups)), 
        
        }
        
        self.individual_types = {"all": [0,1,2], "tier1": [1], "tier2": [2]}
        self.diseases = ["all_and_no_cases", "ipd", "cap"]
        
        self.vaccine_delivered = {"ever":99999999999, 
                                  "1":364, 
                                  "5": 364 * 5}
        
        self.age_0_conversion = pl.DataFrame([
           pl.Series("age", [0] * 330),
           pl.Series("age_days", list(range(330))),
           pl.Series("age_coef_by_age", [0] * 30 + [1] * 60 + [2] * 60 + \
                               [3] * (180)), #+ [-1] * 34 ),
           ]) 
        
        
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
        
        
        
        #v116 addition
        v116_strains = ["3", "6A", "7F", "8", "9N", "10A", "11A", "12F","15A",
                        "16F", "17F", "19A", "20", "22F", "23A", "23B", "24F", 
                        "31", "33F", "35B", "15C"]
        self.v116_strains = pl.Series("v116", v116_strains)
        self.v116_not_pcv13_strains = pl.Series("v116_not_pcv13", 
                 list(set(v116_strains) - set(self.pcv13_strains.to_list())))
        self.v116_not_pcv20_strains = pl.Series("v116_not_pcv20", 
                 list(set(v116_strains) - set(self.pcv20_strains.to_list())))
        self.v116_not_ppv23_strains = pl.Series("v116_not_ppv23", 
                 list(set(v116_strains) - set(self.ppsv23_strains.to_list())))
        
        self.v116_only_strains = pl.Series("v116_only", 
                 list(set(self.v116_not_ppv23_strains.to_list()) - 
                      set(self.pcv20_strains.to_list())))
        
        self.vt_strains =  pl.Series("vt_strains", 
                 list(set(self.v116_strains.to_list()).union(
                      set(self.pcv20_strains.to_list())).union( 
                       set(self.ppsv23_strains))
                      ))
        
        self.pcv15_not_pcv13_strains = pl.Series("pcv15_not_pcv13", 
                 list(set(self.pcv15_strains) - set(self.pcv13_strains.to_list())))
        
        
        super(VaccineObserverByAgeByProduct, self).__init__(h5file =h5file, \
                            label = 'vacc_byagebyproduct', description = desc,
                            title = 'VaccineObserverByAgeByProduct')
        
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
            
            delivered_vaccines = pop.I.unnest("vaccines")["vaccine_type"].unique().to_list() + ["all_no"]
            cur_disease = "all_and_no_cases"
            
            for cur_vaccine in delivered_vaccines:
                for individuals, at_risk_values in self.individual_types.items():
                    for vacc_delivered, time_since_vacc in self.vaccine_delivered.items():
                        """
                        cur_vaccine = "all_no"
                        individuals, at_risk_values = list(self.individual_types.items())[0]
                        vacc_delivered, time_since_vacc = list(self.vaccine_delivered.items())[0]
                        
                        """
                        if (cur_vaccine == "all_no") & (time_since_vacc < 999999):
                            continue
                        elif cur_vaccine == "all_no":
                            cur_pop = (pop.I.select("at_risk", "age", "age_days", "vaccines")
                                           .unnest("vaccines")
                                       .filter( (pl.col("at_risk").is_in(at_risk_values))
                                               )
                                       )
                        else:
                            cur_pop = (pop.I.select("at_risk", "age", "age_days", "vaccines")
                                       .unnest("vaccines")
                                       .filter((pl.col("vaccine_type") == cur_vaccine) &
                                              (pl.col("final_vaccine_time")/ (364/ self.interval) + \
                                                      self.interval * time_since_vacc >= t) &
                                                  (pl.col("at_risk").is_in(at_risk_values))
                                               )
                                       )
                        cur_pop = (cur_pop.join(self.age_0_conversion,
                                    on= ["age", "age_days"],how="left").with_columns(
                                            (pl.when(pl.col("age_coef_by_age").is_null())
                                            .then((pl.col("age") + 4))
                                            .otherwise(pl.col("age_coef_by_age")))
                                                .alias("age_coef_by_age")
                                            ))
                        cur_pop = (cur_pop
                                .groupby(['age_coef_by_age']) 
                                #.count()  #same as 
                                .agg(pl.count())
                                #.pivot('count','age_coef_by_age')
                                ).fill_null(0).sort("age_coef_by_age")                
                                        
                        cur_pop = pl.DataFrame(
                            pl.Series("age_coef_by_age", 
                                      range(self.no_age_groups))).join(
                                cur_pop, on = "age_coef_by_age", how = "left"
                                ).fill_null(0).sort("age_coef_by_age")
                        
                        self.row['t'] = year
                        self.row['vaccine_type'] = cur_vaccine
                        self.row['individual_type'] = individuals
                        self.row['disease'] = cur_disease
                        self.row['vaccine_delivered'] = vacc_delivered
                        self.row['inds_per_age_group'] = cur_pop["count"].to_list()
                        self.row.append()
                        self.h5file.flush()
            
            #delivered_vaccines = pop.I.unnest("vaccines")["vaccine_type"].unique().to_list() + ["all_no"]
            diseases = ["ipd", "cap"]
            
            for cur_disease in diseases:
                for cur_vaccine in delivered_vaccines:
                    for individuals, at_risk_values in self.individual_types.items():
                        for vacc_delivered, time_since_vacc in self.vaccine_delivered.items():
                            """
                            cur_vaccine = "all_no"
                            individuals, at_risk_values = list(self.individual_types.items())[0]
                            vacc_delivered, time_since_vacc = list(self.vaccine_delivered.items())[0]
                            cur_disease = diseases[0]
                            """
                            if (cur_vaccine == "all_no") & (time_since_vacc < 999999):
                                continue
                            elif cur_vaccine == "all_no":
                                cur_pop = (pop.disease_pop.select("at_risk", "age", "age_days", "vaccines", "disease")
                                               .unnest("vaccines")
                                           .filter((pl.col("at_risk").is_in(at_risk_values)) &
                                                    (pl.col("disease") == cur_disease)
                                                   )
                                           )
                            else:
                                cur_pop = (pop.disease_pop.select("at_risk", "age", "age_days", "vaccines", "disease")
                                           .unnest("vaccines")
                                           .filter((pl.col("vaccine_type") == cur_vaccine) &
                                                  (pl.col("final_vaccine_time")/ (364/ self.interval) + \
                                                          self.interval * time_since_vacc >= t) &
                                                      (pl.col("at_risk").is_in(at_risk_values)) &
                                                      (pl.col("disease") == cur_disease)
                                                   )
                                           )
                            cur_pop = (cur_pop.join(self.age_0_conversion,
                                        on= ["age", "age_days"],how="left").with_columns(
                                                (pl.when(pl.col("age_coef_by_age").is_null())
                                                .then((pl.col("age") + 4))
                                                .otherwise(pl.col("age_coef_by_age")))
                                                    .alias("age_coef_by_age")
                                                ))
                            cur_pop = (cur_pop
                                    .groupby(['age_coef_by_age']) 
                                    #.count()  #same as 
                                    .agg(pl.count())
                                    #.pivot('count','age_coef_by_age')
                                    ).fill_null(0).sort("age_coef_by_age")                
                                            
                            cur_pop = pl.DataFrame(
                                pl.Series("age_coef_by_age", 
                                          range(self.no_age_groups))).join(
                                    cur_pop, on = "age_coef_by_age", how = "left"
                                    ).fill_null(0).sort("age_coef_by_age")
                            
                            self.row['t'] = year
                            self.row['vaccine_type'] = cur_vaccine
                            self.row['individual_type'] = individuals
                            self.row['disease'] = cur_disease
                            self.row['vaccine_delivered'] = vacc_delivered
                            self.row['inds_per_age_group'] = cur_pop["count"].to_list()
                            self.row.append()
                            self.h5file.flush()



            
    