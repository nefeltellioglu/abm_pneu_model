"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class VaccinationEffDiseaseObserver(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.vaccines = vaccines
        self.interval = interval
        
        desc = {'t': tb.UInt32Col(pos=0),
                'age': tb.UInt32Col(pos=1),
                'age_coef': tb.UInt32Col(pos=2),
                'quantile': tb.Float64Col(pos=3),
                'final_vaccine_time':tb.UInt32Col(pos=4),
                'vaccine_type': tb.StringCol(pos=5, itemsize = 30),
                'no_of_doses': tb.UInt32Col(pos=6),
                'meanlog': tb.Float64Col(pos=7),
                'sdlog': tb.Float64Col(pos=8),
                'exposed_strains': tb.StringCol(pos=9, itemsize = 4),
                'log_antibodies': tb.Float64Col(pos=10),
                'prob_disease_logantibody': tb.Float64Col(pos=11),
                'disease': tb.StringCol(pos=12, itemsize = 10),
                }              
        super(VaccinationEffDiseaseObserver, self).__init__(h5file =h5file, \
                        label = 'vaccination_eff_disease', description = desc,
                            title = 'Vaccination Eff Disease Observer')
        

    
    def update(self, t, day, pop, **kwargs):
        
        if t % self.interval == 0:    
            if len(pop.vaccinated_disease_pop):
                for row in pop.vaccinated_disease_pop.rows(named=True):
                    self.row['t'] = day
                    self.row['age'] = row['age']
                    self.row['age_coef'] = row['age_coef']
                    self.row['quantile'] = row['quantile']
                    self.row['final_vaccine_time'] = row['final_vaccine_time']
                    self.row['vaccine_type'] = row['vaccine_type']
                    self.row['no_of_doses'] = row['no_of_doses']
                    self.row['meanlog'] = row['meanlog']
                    self.row['sdlog'] = row['sdlog']
                    self.row['exposed_strains'] = row['exposed_strains']
                    self.row['log_antibodies'] = \
                                            row["log_antibodies"]
                    self.row['prob_disease_logantibody'] = \
                                            row['prob_dis_logantibody_scale']
                    if row["will_develop_disease"]:
                        self.row['disease'] = row['disease']
                    else:
                        self.row['disease'] = ""
                                            
                    self.row.append()
                    self.h5file.flush()
                
        else:    
            diseased_pop = pop.vaccinated_disease_pop.filter(
                ~pl.col("disease").is_null()    
                )
            if len(diseased_pop):
                for row in diseased_pop.rows(named=True):
                    self.row['t'] = day
                    self.row['age'] = row['age']
                    self.row['age_coef'] = row['age_coef']
                    self.row['quantile'] = row['quantile']
                    self.row['final_vaccine_time'] = row['final_vaccine_time']
                    self.row['vaccine_type'] = row['vaccine_type']
                    self.row['no_of_doses'] = row['no_of_doses']
                    self.row['meanlog'] = row['meanlog']
                    self.row['sdlog'] = row['sdlog']
                    self.row['exposed_strains'] = row['exposed_strains']
                    self.row['wog_antibodies'] = \
                                            row["log_antibodies"]
                    self.row['prob_disease_logantibody'] = \
                                            row['prob_dis_logantibody_scale']
                    if row["will_develop_disease"]:
                        self.row['disease'] = row['disease']
                    else:
                        self.row['disease'] = ""
                                            
                    self.row.append()
                    self.h5file.flush()
                

   

    
            
    